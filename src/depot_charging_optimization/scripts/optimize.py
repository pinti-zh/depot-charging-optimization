from functools import reduce
from itertools import groupby
from math import gcd
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import click
from rich import print as printr

from depot_charging_optimization.core import (
    GreedyOptimizationModel,
    OptimizationInput,
    OptimizationModel,
)
from depot_charging_optimization.utils import expand_values, partial_sums


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
    ax.plot([0] + time, state_of_energy, color=color, label=label)
    if lb is not None:
        ax.plot([0] + time, [lb] * (len(time) + 1), color=color, linestyle="dashed")
    if ub is not None:
        ax.plot([0] + time, [ub] * (len(time) + 1), color=color, linestyle="dashed")
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
@click.argument("data_files", type=str, nargs=-1)
@click.option("energy_price_file", "-epf", type=str, default="data/energy_price.csv", help="energy price file")
@click.option("--ce_function", "-cef", type=click.Choice(["constant", "quadratic", "one"], case_sensitive=False), default="one")
@click.option("--alpha", "-a", type=float, default=1.0, help="constant for charging efficiency function")
def optimize(data_files, energy_price_file, ce_function, alpha):
    printr("loading the following files:")
    for i, file in enumerate(data_files):
        printr(f"  {i+1}. [cyan3]{file}[/cyan3]")
    data = [pl.read_csv(data_file) for data_file in data_files]
    energy_price = pl.read_csv(energy_price_file)
    energy_price = energy_price.with_columns(pl.col("energy_price").truediv(3.6e6))

    all_timestamps = []
    for df in data:
        all_timestamps += list(df["time"])
    all_timestamps += list(energy_price["time"])
    dt = reduce(gcd, all_timestamps)

    expanded_data = []
    for df in data:
        expanded_data.append(
            pl.DataFrame(
                {
                    "time": expand_values(df["time"], df["time"], dt, interpolation="linear"),
                    "energy_demand": expand_values(df["time"], df["energy_demand"], dt, interpolation="split"),
                    "depot_charge": expand_values(df["time"], df["depot_charge"], dt),
                    "charge_amount": expand_values(df["time"], df["charge_amount"], dt, interpolation="split"),
                    "battery_capacity": expand_values(df["time"], df["battery_capacity"], dt),
                    "max_charging_power": expand_values(df["time"], df["max_charging_power"], dt),
                    "cycle": expand_values(df["time"], df["cycle"], dt),
                }
            )
        )
    energy_price = pl.DataFrame(
        {
            "time": expand_values(energy_price["time"], energy_price["time"], dt, interpolation="linear"),
            "energy_price": expand_values(energy_price["time"], energy_price["energy_price"], dt),
        }
    )

    # optimization
    opt_input = OptimizationInput.from_dataframes(expanded_data, energy_price, 0.2e-3)
    start = perf_counter()
    ok, reasons = opt_input.is_feasible()
    printr(f"[grey69]feasibility check in {perf_counter() - start:.4f} seconds")
    if not ok:
        printr("[gold1]optimization input is not feasible:")
        for description, vehicles in reasons.items():
            for vehicle in vehicles:
                printr(f"    [light_goldenrod2]{description} in [light_salmon1]{data_files[vehicle]}")

    start = perf_counter()
    opt_model = OptimizationModel(opt_input)
    opt_model.set_variables()
    opt_model.set_constraints(ce_function_type=ce_function, alpha=alpha)
    opt_model.set_objective()

    # solve
    solution = opt_model.solve()
    optimization_time = perf_counter() - start

    if solution is None:
        printr("[orange1]no solution found")
    else:
        printr(f"[green]found solution in {optimization_time:.4f} seconds with objective value: {solution}")
        # greedy solutions for comparison
        best_max_power = opt_model.get_max_charging_power_used()
        for adjusted_max_power in [None, best_max_power]:
            greedy_opt_model = GreedyOptimizationModel(opt_input)
            greedy_opt_model.set_variables()
            greedy_opt_model.set_constraints(
                ce_function_type=ce_function, alpha=alpha, adjusted_max_power=adjusted_max_power
            )
            greedy_opt_model.set_objective()
            greedy_solution = greedy_opt_model.solve()
            if adjusted_max_power is None:
                printr(f"[grey69]                              greedy solution (naive): {greedy_solution}")
            else:
                printr(f"[grey69]                 greedy solution (max power adjusted): {greedy_solution}")

    if solution is not None and False:
        sns.set_style("darkgrid")
        colors = ["navy", "gold", "orchid", "orangered", "mediumseagreen", "saddlebrown", "cornflowerblue"]
        _, axes = plt.subplots(3, figsize=(12, 8))

        time = expanded_data[0]["time"].to_list()
        joule_to_kwh = 1.0 / 3.6e6

        soe = opt_model.get_state_of_energy() * joule_to_kwh
        charging_power = opt_model.get_charging_power() / 1000.0

        # plot state of energy
        for i, soe_i in enumerate(soe):
            plot_state_of_energy(axes[0], time, soe_i, color=colors[i % len(colors)], label=f"SoE V{i+1}")

        # plot charging power
        plot_charging_power(axes[1], time, charging_power[0], color=colors[0], label="Charging Power V1")
        for vehicle in range(1, opt_input.num_vehicles):
            plot_charging_power(
                axes[1],
                time,
                charging_power[vehicle],
                color=colors[vehicle % len(colors)],
                label=f"Charging Power V{vehicle+1}",
                bottom=np.sum(charging_power[:vehicle], axis=0),
            )

        # plot energy price
        plot_energy_price(
            axes[2], time, energy_price["energy_price"], color="firebrick", label="Energy Price", f=1 / joule_to_kwh
        )

        axes[0].set_title("Optimization Result")
        plt.show(block=False)

        # plot depot charge intervals
        plot_shape = get_axes_shape(opt_input.num_vehicles)
        plot_shape = (max(plot_shape[0], 2), max(plot_shape[0], 2))
        _, axes = plt.subplots(*plot_shape, figsize=(12, 8))
        for vehicle in range(opt_input.num_vehicles):
            ax_i, ax_j = get_axes_indices(vehicle, plot_shape)
            plot_depot_charging_intervals(
                axes[ax_i, ax_j], time, opt_input.depot_charge[vehicle], color=colors[vehicle % len(colors)], alpha=0.2
            )
            plot_charging_power(
                axes[ax_i, ax_j],
                time,
                charging_power[vehicle],
                color=colors[vehicle % len(colors)],
                label=f"Charging Power V{vehicle+1}",
            )
            plot_state_of_energy(
                axes[ax_i, ax_j],
                time,
                soe[vehicle],
                lb=opt_input.soe_lb[vehicle] * joule_to_kwh,
                ub=opt_input.soe_ub[vehicle] * joule_to_kwh,
            )

        # remove axes that are not used
        for index in range(opt_input.num_vehicles, plot_shape[0] * plot_shape[1]):
            ax_i, ax_j = get_axes_indices(index, plot_shape)
            axes[ax_i, ax_j].remove()
        plt.show()
