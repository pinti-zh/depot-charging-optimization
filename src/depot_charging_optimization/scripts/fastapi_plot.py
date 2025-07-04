import json
import logging
import os

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


@app.get("/")
async def index(request: Request):
    with open(os.getenv("SOLUTION"), "r") as f:
        solution = Solution.from_dict(json.load(f))
    time = [i * solution.optimization_input.dt for i in range(solution.optimization_input.num + 1)]

    # Initial plot data as JSON
    fig = go.Figure()
    for vehicle in range(solution.optimization_input.num_vehicles):
        fig.add_trace(go.Scatter(x=time, y=list(solution.state_of_energy[vehicle] / (3600 * 1000))))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )

    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "plot_json": fig.to_json(),
            "vehicles": list(i + 1 for i in range(solution.optimization_input.num_vehicles)),
        },
    )


@app.get("/plot")
async def get_empty_plot():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return JSONResponse(content=fig.to_dict())


@app.get("/plot/{vehicles_str}")
async def get_plot(vehicles_str):
    vehicles = [int(v) - 1 for v in vehicles_str.split(",")]

    with open(os.getenv("SOLUTION"), "r") as f:
        solution = Solution.from_dict(json.load(f))
    time = [i * solution.optimization_input.dt for i in range(solution.optimization_input.num + 1)]

    fig = go.Figure()
    for vehicle in vehicles:
        fig.add_trace(go.Scatter(x=time, y=list(solution.state_of_energy[vehicle] / (3600 * 1000))))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )

    return JSONResponse(content=fig.to_dict())


@click.command()
@click.option("--reload", "-r", is_flag=True, default=False, help="reload the app")
@click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")
def serve(reload, solution_file):
    logger.info(f"Loaded solution from [cyan3]{solution_file}")
    os.environ["SOLUTION"] = solution_file
    uvicorn.run("depot_charging_optimization.scripts.fastapi_plot:app", host="0.0.0.0", port=8000, reload=reload)
