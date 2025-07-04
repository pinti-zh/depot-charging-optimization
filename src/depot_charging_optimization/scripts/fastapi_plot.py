import json
import logging
import os
from typing import Optional

import click
import plotly.graph_objs as go
import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from rich.logging import RichHandler

from depot_charging_optimization.core import Solution

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

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
        solution = Solution.from_dict(json.load(f))
    return solution


def update_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )


def soe_figure(vehicles: Optional[list] = None):
    solution = get_solution()
    time = [i * solution.optimization_input.dt for i in range(solution.optimization_input.num + 1)]
    if vehicles is None:
        vehicles = range(solution.optimization_input.num_vehicles)

    fig = go.Figure()
    for vehicle in vehicles:
        color = TRACE_COLORS[vehicle % len(TRACE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=time,
                y=list(solution.state_of_energy[vehicle] / (3600 * 1000)),
                mode="lines",
                marker=dict(color=color),
                line=dict(color=color),
            )
        )
    update_layout(fig)
    return fig


def cp_figure(vehicles: Optional[list] = None):
    solution = get_solution()
    time = [(i + 1) * solution.optimization_input.dt for i in range(solution.optimization_input.num)]
    if vehicles is None:
        vehicles = range(solution.optimization_input.num_vehicles)

    fig = go.Figure()
    for vehicle in vehicles:
        color = TRACE_COLORS[vehicle % len(TRACE_COLORS)]
        fig.add_trace(
            go.Bar(
                x=time,
                y=list(solution.charging_power[vehicle] / 1000),
                marker_color=color,
                marker=dict(line=dict(width=0)),
                opacity=0.8,
            )
        )
    fig.update_layout(barmode="stack")
    update_layout(fig)
    return fig


def detail_figure(vehicle: int = -1):
    fig = go.Figure()
    if vehicle == -1:
        update_layout(fig)
        return fig

    solution = get_solution()
    time = [i * solution.optimization_input.dt for i in range(solution.optimization_input.num + 1)]
    color = TRACE_COLORS[vehicle % len(TRACE_COLORS)]
    fig.add_trace(
        go.Scatter(
            x=time,
            y=list(solution.state_of_energy[vehicle] / (3600 * 1000)),
            mode="lines",
            marker=dict(color=color),
            line=dict(color=color),
        )
    )

    update_layout(fig)
    return fig


@app.get("/")
async def index(request: Request):
    vehicles = list(range(get_solution().optimization_input.num_vehicles))

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
            "vehicles": vehicles,
            "colors": TRACE_COLORS,
        },
    )


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
    uvicorn.run("depot_charging_optimization.scripts.fastapi_plot:app", host="0.0.0.0", port=8000, reload=reload)
