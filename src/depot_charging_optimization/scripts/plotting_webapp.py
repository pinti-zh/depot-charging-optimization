import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import click
import plotly.graph_objs as go
import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.subplots import make_subplots
from rich.logging import RichHandler

from depot_charging_optimization.core import Solution

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Mount the static directory
STATIC = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

TRACE_COLORS = [
    "#58A7C5",  # Darker Soft Sky Blue
    "#63B88D",  # Darker Light Mint
    "#E58D87",  # Darker Pale Coral
    "#A987D6",  # Darker Lavender Mist
    "#C7A94F",  # Darker Misty Gold
    "#BA8B86",  # Darker Dusty Rose
    "#4F9792",  # Darker Muted Teal
    "#9889D4",  # Darker Cool Lilac
    "#E4A87E",  # Darker Soft Apricot
    "#89AEB0",  # Darker Silver Blue
    "#90B9A9",  # Darker Powder Green
    "#CDBE97",  # Darker Creamy Sand
]


logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)  # or DEBUG

logger = logging.getLogger("plot")

uvicorn_loggers = ["uvicorn", "uvicorn.error"]

for name in uvicorn_loggers:
    log = logging.getLogger(name)
    log.handlers = logger.handlers
    log.setLevel(logger.level)
    log.propagate = False  # Important: don't duplicate logs


def get_solution():
    with open(os.getenv("SOLUTION"), "r") as f:
        solution = Solution.model_validate(json.load(f))
    return solution


def convert_seconds_to_time(seconds):
    start = datetime(2025, 1, 1)  # this date is arbitrary
    x_times = [(start + timedelta(seconds=s)).isoformat() for s in seconds]
    return x_times


def update_layout(fig, yaxis_title=""):
    fig.update_layout(
        xaxis=dict(
            type="date",
            tickformatstops=[
                dict(dtickrange=[None, 1000], value="%H:%M:%S.%L"),  # < 1s
                dict(dtickrange=[1000, 60000], value="%H:%M:%S"),  # < 1min
                dict(dtickrange=[60000, 3600000], value="%H:%M"),  # < 1h
                dict(dtickrange=[3600000, 86400000], value="%H:%M"),  # < 1 day
                dict(dtickrange=[86400000, 604800000], value="%a %H:%M"),  # < 1 week
                dict(dtickrange=[604800000, None], value="%b %d"),  # > 1 week
            ],
        ),
        xaxis_title="Time of Day",
        yaxis_title=yaxis_title,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )


def soe_figure(vehicles: list | None = None):
    solution = get_solution()
    time = [0] + solution.input_data.time
    if vehicles is None:
        vehicles = range(solution.input_data.num_vehicles)

    fig = go.Figure()
    for vehicle in vehicles:
        color = TRACE_COLORS[vehicle % len(TRACE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=convert_seconds_to_time(time),
                y=[soe / 3.6e6 for soe in solution.state_of_energy[vehicle]],
                mode="lines",
                marker=dict(color=color),
                line=dict(color=color),
            )
        )
    update_layout(fig, yaxis_title="State of Energy [kWh]")
    return fig


def cp_figure(vehicles: Optional[list] = None):
    solution = get_solution()
    time = [solution.input_data.time[0] / 2]
    width = [solution.input_data.time[0]]
    for t1, t2 in zip(solution.input_data.time[:-1], solution.input_data.time[1:]):
        width.append(t2 - t1)
        time.append((t1 + t2) / 2)

    if vehicles is None:
        vehicles = range(solution.input_data.num_vehicles)

    total_charging_power = [0.0 for _ in range(len(solution.charging_power[0]))]
    fig = go.Figure()
    for vehicle in vehicles:
        color = TRACE_COLORS[vehicle % len(TRACE_COLORS)]
        fig.add_trace(
            go.Bar(
                x=convert_seconds_to_time(time),
                y=[cp / 1000 for cp in solution.charging_power[vehicle]],
                width=[1000 * w for w in width],  # when using datetime, width must be in milliseconds
                marker_color=color,
                marker=dict(line=dict(width=0)),
                opacity=0.8,
            )
        )
        for i, cp in enumerate(solution.charging_power[vehicle]):
            total_charging_power[i] += cp
    interval_time = [t for t in solution.input_data.time for _ in range(2)]
    interval_time = [0] + interval_time[:-1]
    interval_total_charging_power = [cp for cp in total_charging_power for _ in range(2)]
    fig.add_trace(
        go.Scatter(
            x=convert_seconds_to_time(interval_time),
            y=[cp / 1000 for cp in interval_total_charging_power],
            mode="lines",
            marker=dict(color="white"),
            line=dict(color="white"),
        )
    )
    fig.update_layout(
        barmode="relative",
    )
    update_layout(fig, yaxis_title="Charging Power [kW]")
    return fig


def energy_price_figure():
    solution = get_solution()
    time = [t for t in solution.input_data.time for _ in range(2)]
    time = [0] + time[:-1]
    energy_price_twice = [ep for ep in solution.input_data.energy_price for _ in range(2)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=convert_seconds_to_time(time),
            y=[ep * 3.6e6 for ep in energy_price_twice],
            mode="lines",
        )
    )
    fig.update_layout(
        barmode="stack",
    )
    update_layout(fig, yaxis_title="Energy Price [CHF/kWh]")
    return fig


def detail_figure(vehicle: int = -1):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if vehicle == -1:
        update_layout(fig)
        return fig

    solution = get_solution()
    time = [0] + solution.input_data.time
    color = TRACE_COLORS[vehicle % len(TRACE_COLORS)]

    # state of Energy
    fig.add_trace(
        go.Scatter(
            x=convert_seconds_to_time(time),
            y=[soe / 3.6e6 for soe in solution.state_of_energy[vehicle]],
            mode="lines",
            marker=dict(color=color),
            line=dict(color=color),
        )
    )
    for bound in [
        solution.input_data.soe_lb[vehicle] * solution.input_data.battery_capacity[vehicle],
        solution.input_data.soe_ub[vehicle] * solution.input_data.battery_capacity[vehicle],
    ]:
        fig.add_trace(
            go.Scatter(
                x=convert_seconds_to_time(time),
                y=[bound / 3.6e6 for _ in time],
                mode="lines",
                marker=dict(color=color),
                line=dict(color=color, dash="dash"),
            )
        )

    # charging power
    cp_time = [solution.input_data.time[0] / 2]
    width = [solution.input_data.time[0]]
    for t1, t2 in zip(solution.input_data.time[:-1], solution.input_data.time[1:]):
        width.append(t2 - t1)
        cp_time.append((t1 + t2) / 2)
    fig.add_trace(
        go.Bar(
            x=convert_seconds_to_time(cp_time),
            y=[cp / 1000 for cp in solution.charging_power[vehicle]],
            width=[w * 1000 for w in width],
            marker_color=color,
            marker=dict(line=dict(width=0)),
            opacity=0.4,
        ),
        secondary_y=True,
    )

    # depot charging intervals
    bands = []
    for t1, t2, dc in zip(
        convert_seconds_to_time(time), convert_seconds_to_time(time[1:]), solution.input_data.depot_charge[vehicle]
    ):
        if dc:
            bands.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=t1,
                    x1=t2,
                    y0=0,
                    y1=1,
                    fillcolor=color,
                    opacity=0.2,
                    line_width=0,
                    layer="below",
                )
            )
    fig.update_layout(
        shapes=bands,
        yaxis2_title="Charging Power [kW]",
    )
    update_layout(fig, yaxis_title="State of Energy [kWh]")
    return fig


@app.get("/")
async def index(request: Request):
    solution = get_solution()
    indices = list(range(solution.input_data.num_vehicles))
    names = []
    battery_index = 1
    bus_index = 1
    for dc in solution.input_data.depot_charge:
        if all(dc):
            names.append(f"Battery {battery_index}")
            battery_index += 1
        else:
            names.append(f"Bus {bus_index}")
            bus_index += 1

    # Initial plot data as JSON
    fig_soe = soe_figure([])
    fig_cp = cp_figure([])
    fig_detail = detail_figure(-1)

    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "soe_plot_json": fig_soe.to_json(),
            "cp_plot_json": fig_cp.to_json(),
            "detail_plot_json": fig_detail.to_json(),
            "indices": indices,
            "names": names,
            "colors": TRACE_COLORS,
        },
    )


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(STATIC, "favicon.ico"))


@app.get("/soe_plot")
async def get_empty_soe_plot():
    fig = soe_figure(vehicles=[])
    return JSONResponse(content=fig.to_dict())


@app.get("/soe_plot/{vehicles_str}")
async def get_soe_plot(vehicles_str):
    vehicles = [int(v) for v in vehicles_str.split(",")]
    fig = soe_figure(vehicles=vehicles)
    return JSONResponse(content=fig.to_dict())


@app.get("/cp_plot")
async def get_empty_cp_plot():
    fig = cp_figure(vehicles=[])
    return JSONResponse(content=fig.to_dict())


@app.get("/energy_price_plot")
async def get_energy_price_plot():
    fig = energy_price_figure()
    return JSONResponse(content=fig.to_dict())


@app.get("/cp_plot/{vehicles_str}")
async def get_cp_plot(vehicles_str):
    vehicles = [int(v) for v in vehicles_str.split(",")]
    fig = cp_figure(vehicles=vehicles)
    return JSONResponse(content=fig.to_dict())


@app.get("/detail_plot/{vehicle_str}")
async def get_detail_plot(vehicle_str):
    vehicle = int(vehicle_str)
    fig = detail_figure(vehicle=vehicle)
    return JSONResponse(content=fig.to_dict())


@click.command()
@click.option("--reload", "-r", is_flag=True, default=False, help="reload the app")
@click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")
def serve(reload, solution_file):
    logger.info(f"Loaded solution from [cyan3]{solution_file}")
    os.environ["SOLUTION"] = solution_file
    uvicorn.run("depot_charging_optimization.scripts.plotting_webapp:app", host="0.0.0.0", port=8000, reload=reload)
