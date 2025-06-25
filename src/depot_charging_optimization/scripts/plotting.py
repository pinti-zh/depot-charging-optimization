import json
import logging
from itertools import groupby

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.logging import RichHandler

from depot_charging_optimization.core import Solution
from depot_charging_optimization.utils import partial_sums

# Basic Rich logging setup
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)  # or DEBUG

logger = logging.getLogger("plot")


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


@click.command()
@click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")
def plot_solution(solution_file):
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
