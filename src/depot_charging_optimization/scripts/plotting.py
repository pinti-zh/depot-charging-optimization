import json
import logging
import os
import sys
from itertools import groupby

import click
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, dcc, html
from dash_iconify import DashIconify
from rich.console import Console
from rich.logging import RichHandler

from depot_charging_optimization.core import Solution
from depot_charging_optimization.utils import partial_sums

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


def get_interval_time_series(time):
    intervals = [0]
    for t in time[1:]:
        intervals += [t, t]
    intervals.append(time[-1])
    return intervals


def get_axes_shape(n):
    h, w = 0, 0
    while w * h < n:
        w += 1
        if w * h >= n:
            return h, w
        h += 1
    return h, w


def get_axes_indices(n, shape):
    if shape[1] <= 0:
        return 0, 0
    i = n // shape[1]
    j = n % shape[1]
    return (i, j)


def plot_state_of_energy(ax, time, state_of_energy, lb=None, ub=None, color="black", label=None):
    ax.plot(time, state_of_energy, color=color, label=label)
    if lb is not None:
        ax.plot(time, [lb] * len(time), color=color, linestyle="dashed")
    if ub is not None:
        ax.plot(time, [ub] * len(time), color=color, linestyle="dashed")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("SoE [kWh]")
    ax.set_xlim(0, max(time))
    ax.legend()


def plot_charging_power(ax, time, charging_power, color="black", label=None, bottom=None):
    dt = time[1] - time[0]
    ax.bar(
        [t - dt / 2 for t in time],
        charging_power,
        bottom=bottom,
        width=dt,
        label=label,
        color=color,
        edgecolor="none",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Charging Power [kW]")
    ax.set_xlim(0, max(time))
    ax.legend()


def plot_depot_charging_intervals(ax, time, depot_charge, color="black", alpha=1.0, label=None):
    dt = time[1] - time[0]
    dc_dups = [sum(1 for _ in group) for _, group in groupby(depot_charge)]
    dup_intervals = [(i * dt, j * dt) for i, j in zip(partial_sums([0] + dc_dups[:-1]), partial_sums(dc_dups))]
    if depot_charge[0]:
        dup_intervals = dup_intervals[::2]
    else:
        dup_intervals = dup_intervals[1::2]
    for t1, t2 in dup_intervals:
        ax.axvspan(t1, t2, color=color, alpha=alpha, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_xlim(0, max(time))
    if label is not None:
        ax.legend()


def plot_energy_price(ax, time, energy_price, color="black", label=None, f=1.0):
    energy_price_twice = []
    for ep in energy_price:
        energy_price_twice += [ep * f, ep * f]
    ax.plot(
        get_interval_time_series(time),
        energy_price_twice,
        c=color,
        label=label,
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy Price [$/kWh]")
    ax.set_xlim(0, max(time))
    ax.set_ylim(min(energy_price_twice) * 0.9, max(energy_price_twice) * 1.1)
    ax.legend()


def display_string(v):
    if isinstance(v, np.ndarray):
        return f"Shape = {v.shape}, Range = [{v.min():.3f}, {v.max():.3f}]"
    else:
        return str(v)


def create_app(solution):
    soe_dict = {
        "time": np.linspace(
            0, solution.optimization_input.num * solution.optimization_input.dt, solution.optimization_input.num + 1
        )
    }
    charging_power_dict = {
        "time": np.linspace(
            solution.optimization_input.dt / 2,
            solution.optimization_input.num * solution.optimization_input.dt - solution.optimization_input.dt / 2,
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
        fig = go.Figure(
            data=[
                go.Bar(
                    name=f"V{i+1}",
                    x=charging_power_df["time"],
                    y=charging_power_df[str(i + 1)],
                    offsetgroup=0,
                    width=solution.optimization_input.dt,
                )
                for i in range(solution.optimization_input.num_vehicles)
            ],
        )
        # fig = px.line(
        # charging_power_df,
        # x="time",
        # y=checklist,
        # labels={
        #    "time": "Time [s]",
        #    "value": "Charging Power [kW]",
        # },
        # )
        fig.update_layout(legend_title="Vehicles", margin=dict(t=10), template="plotly_dark")
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


@click.command()
@click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")
def plot_solution_old(solution_file):
    with open(solution_file, "r") as f:
        solution = Solution.from_dict(json.load(f))
    logger.info(f"Loaded solution from [cyan3]{solution_file}")

    sns.set_style("darkgrid")
    colors = ["navy", "gold", "orchid", "orangered", "mediumseagreen", "saddlebrown", "cornflowerblue"]
    _, axes = plt.subplots(3, figsize=(12, 8))

    time = np.array([solution.optimization_input.dt * i for i in range(solution.optimization_input.num + 1)])
    joule_to_kwh = 1.0 / 3.6e6

    soe = solution.state_of_energy * joule_to_kwh
    charging_power = solution.charging_power / 1000.0

    # plot state of energy
    for i, soe_i in enumerate(soe):
        plot_state_of_energy(axes[0], time, soe_i, color=colors[i % len(colors)], label=f"SoE V{i+1}")

    # plot charging power
    plot_charging_power(axes[1], time[1:], charging_power[0], color=colors[0], label="Charging Power V1")
    for vehicle in range(1, solution.optimization_input.num_vehicles):
        plot_charging_power(
            axes[1],
            time[1:],
            charging_power[vehicle],
            color=colors[vehicle % len(colors)],
            label=f"Charging Power V{vehicle+1}",
            bottom=np.sum(charging_power[:vehicle], axis=0),
        )

    # plot energy price
    plot_energy_price(
        axes[2],
        time[1:],
        solution.optimization_input.energy_price,
        color="firebrick",
        label="Energy Price",
        f=1 / joule_to_kwh,
    )

    axes[0].set_title("Optimization Result")
    plt.show(block=False)

    # plot depot charge intervals
    plot_shape = get_axes_shape(solution.optimization_input.num_vehicles)
    plot_shape = (max(plot_shape[0], 2), max(plot_shape[0], 2))
    _, axes = plt.subplots(*plot_shape, figsize=(12, 8))
    for vehicle in range(solution.optimization_input.num_vehicles):
        ax_i, ax_j = get_axes_indices(vehicle, plot_shape)
        plot_depot_charging_intervals(
            axes[ax_i, ax_j],
            time[1:],
            solution.optimization_input.depot_charge[vehicle],
            color=colors[vehicle % len(colors)],
            alpha=0.2,
        )
        plot_charging_power(
            axes[ax_i, ax_j],
            time[1:],
            charging_power[vehicle],
            color=colors[vehicle % len(colors)],
            label=f"Charging Power V{vehicle+1}",
        )
        plot_state_of_energy(
            axes[ax_i, ax_j],
            time,
            soe[vehicle],
            lb=solution.optimization_input.soe_lb[vehicle] * joule_to_kwh,
            ub=solution.optimization_input.soe_ub[vehicle] * joule_to_kwh,
        )

    # remove axes that are not used
    for index in range(solution.optimization_input.num_vehicles, plot_shape[0] * plot_shape[1]):
        ax_i, ax_j = get_axes_indices(index, plot_shape)
        axes[ax_i, ax_j].remove()
    plt.show()
