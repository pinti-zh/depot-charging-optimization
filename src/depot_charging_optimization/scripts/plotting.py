import json
import logging
import os
import sys

import click
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import polars as pl
from dash import Dash, Input, Output, dcc, html
from dash_iconify import DashIconify
from rich.console import Console
from rich.logging import RichHandler

from depot_charging_optimization.core import Solution

# Basic Rich logging setup
rich_handler = RichHandler(console=Console(stderr=True), markup=True)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich_handler],
)  # or DEBUG

# use rich handler for all the logs we want to see, and write to stderr
logger = logging.getLogger("plot")
logger.addHandler(rich_handler)
logger.propagate = False
logging.getLogger("dash").addHandler(rich_handler)

logging.getLogger("dash").addHandler(rich_handler)
logging.getLogger("dash").propagate = False

# redirect stdout to /dev/null for unwanted logs
sys.stdout = open(os.devnull, "w")


def stack_df_for_area_plot(df: pl.DataFrame) -> pl.DataFrame:
    stacked_dict = {
        "time": ([0] + [int(t) for t in df["time"].to_list() for _ in range(2)][:-1]) * (len(df.columns) - 1),
        "value": [],
        "group": [],
    }
    for col in df.columns:
        if col != "time":
            stacked_dict["value"] += [v for v in df[col].to_list() for _ in range(2)]
            stacked_dict["group"] += [col] * len(df[col]) * 2
    return pl.DataFrame(stacked_dict)


def create_app(solution):
    soe_dict = {
        "time": np.linspace(
            0, solution.optimization_input.num * solution.optimization_input.dt, solution.optimization_input.num + 1
        )
    }
    charging_power_dict = {
        "time": np.linspace(
            solution.optimization_input.dt,
            solution.optimization_input.num * solution.optimization_input.dt,
            solution.optimization_input.num,
        )
    }
    for vehicle in range(solution.optimization_input.num_vehicles):
        soe_dict[str(vehicle + 1)] = solution.state_of_energy[vehicle] / 3.6e6
        charging_power_dict[str(vehicle + 1)] = solution.charging_power[vehicle] / 1e3
    soe_df = pl.DataFrame(soe_dict)
    charging_power_df = pl.DataFrame(charging_power_dict)

    heading = html.Div(
        "⚡ Depot Charging Optimization",
        style={
            "fontSize": "30px",
            "fontWeight": "bold",
            "padding": "20px",
            "textAlign": "center",
            "backgroundColor": "#222",
            "color": "#f0f0f0",
            "borderBottom": "1px solid #444",
            "fontFamily": "Segoe UI, sans-serif",
        },
    )
    soe_heading = html.Div(
        "State of Energy",
        style={
            "fontSize": "20px",
            "fontWeight": "bold",
            "padding": "20px",
            "textAlign": "center",
            "backgroundColor": "#222",
            "color": "#f0f0f0",
            "fontFamily": "Segoe UI, sans-serif",
        },
    )
    cp_heading = html.Div(
        "Charging Power",
        style={
            "fontSize": "20px",
            "fontWeight": "bold",
            "padding": "20px",
            "textAlign": "center",
            "backgroundColor": "#222",
            "color": "#f0f0f0",
            "fontFamily": "Segoe UI, sans-serif",
        },
    )
    checklist = dcc.Checklist(
        options=[
            {
                "label": html.Span([" ", DashIconify(icon="mdi:bus", style={"color": "#ffcc00"}), f" V{i+1}"]),
                "value": str(i + 1),
            }
            for i in range(solution.optimization_input.num_vehicles)
        ],
        value=[str(i + 1) for i in range(solution.optimization_input.num_vehicles)],
        id="checklist",
        className="ms-5",
    )

    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    # app.layout = dbc.Container([heading, checklist, dcc.Graph(id="graph")], fluid=True)
    app.layout = dbc.Container(
        [
            heading,
            dbc.Row(id="top-padding", style={"height": "30px"}),
            soe_heading,
            dbc.Row(
                [
                    dbc.Col(checklist, width=2),  # 3/12 columns wide = ~25% of row width
                    dbc.Col(
                        dcc.Graph(id="state_of_energy_graph"),
                        width=8,
                        style={
                            "border": "2px solid #444",  # dark gray border
                            "borderRadius": "10px",  # rounded corners
                            "padding": "10px",  # inner spacing
                        },
                    ),
                ],
                style={"height": "50vh"},
            ),
            dbc.Row(id="middle-padding", style={"height": "30px"}),
            cp_heading,
            dbc.Row(
                [
                    dbc.Col(width=2),
                    dbc.Col(
                        dcc.Graph(id="charging_power_graph"),
                        width=8,
                        style={
                            "border": "2px solid #444",  # dark gray border
                            "borderRadius": "10px",  # rounded corners
                            "padding": "10px",  # inner spacing
                        },
                    ),
                ],
                style={"height": "50vh"},
            ),
            dbc.Row(id="bottom-padding", style={"height": "30px"}),
        ],
        fluid=True,
    )

    @app.callback(
        Output("state_of_energy_graph", "figure"),
        Input("checklist", "value"),
    )
    def update_state_of_energy(checklist):
        fig = px.line(
            soe_df,
            x="time",
            y=checklist,
            labels={
                "time": "Time [s]",
                "value": "State of Energy [kWh]",
            },
        )
        fig.update_layout(legend_title="Vehicles", margin=dict(t=10), template="plotly_dark")
        return fig

    @app.callback(
        Output("charging_power_graph", "figure"),
        Input("checklist", "value"),
    )
    def update_charging_power(checklist):
        area_charging_power_df = stack_df_for_area_plot(charging_power_df.select(["time"] + checklist))
        fig = px.area(
            area_charging_power_df,
            x="time",
            y="value",
            color="group",
            labels={
                "time": "Time [s]",
                "value": "Charging Power [kW]",
            },
        )
        fig.update_layout(legend_title="Vehicles", margin=dict(t=10), template="plotly_dark")
        fig.update_traces(line=dict(width=0))
        return fig

    return app


@click.command()
@click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")
@click.option("--public", "-p", is_flag=True, default=False, help="publish the app")
def plot_solution(solution_file, public):
    with open(solution_file, "r") as f:
        solution = Solution.from_dict(json.load(f))
    logger.info(f"Loaded solution from [cyan3]{solution_file}")

    app = create_app(solution)

    if public:
        app.run(host="0.0.0.0", port=80)
    else:
        app.run(debug=True)
