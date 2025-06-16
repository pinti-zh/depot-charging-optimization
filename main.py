import argparse
from functools import reduce
from itertools import groupby
from math import gcd

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from rich import print as printr

from optimization.optimization import OptimizationInput, OptimizationModel
from optimization.utils import expand_values, partial_sums


def get_interval_time_series(time):
    intervals = [0]
    for t in time[1:]:
        intervals += [t, t]
    intervals.append(time[-1])
    return intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, nargs="+", required=True, help="path to data file")
    parser.add_argument("--energy_price", "-ep", type=str, default="data/energy_price.csv", help="energy price file")
    parser.add_argument("--ce_function", "-cef", type=str, choices=["constant", "quadratic", "one"], default="one")
    parser.add_argument("--plot", "-p", action="store_true", help="plot results")
    args = parser.parse_args()

    printr("loading the following files:")
    for i, file in enumerate(args.data):
        printr(f"  {i+1}. [cyan3]{file}[/cyan3]")
    data = [pl.read_csv(data_file) for data_file in args.data]
    energy_price = pl.read_csv(args.energy_price)
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
    opt_input = OptimizationInput(expanded_data, energy_price, 0.2e-3)
    ok, reasons = opt_input.is_feasible()
    if not ok:
        printr(f"[gold1]optimization input is not feasible: {reasons}")
    opt_model = OptimizationModel(opt_input)
    opt_model.set_variables()
    opt_model.set_constraints(ce_function_type=args.ce_function)
    opt_model.set_objective()

    # solve
    solution = opt_model.solve()

    if solution is None:
        printr("[orange1]no solution found")
    else:
        printr(f"[green]found solution with objective value: {solution}")

    if solution is not None and args.plot:
        sns.set_style("darkgrid")
        colors = ["navy", "gold", "orchid", "orangered", "mediumseagreen", "saddlebrown", "cornflowerblue"]
        _, axes = plt.subplots(3, figsize=(12, 8))
        time = expanded_data[0]["time"].to_list()

        joule_to_kwh = 1.0 / 3.6e6

        soe = opt_model.get_state_of_energy() * joule_to_kwh
        charging_power = opt_model.get_charging_power() / 1000.0
        energy_price_twice = []
        for ep in energy_price["energy_price"]:
            energy_price_twice += [ep / joule_to_kwh, ep / joule_to_kwh]

        # plot state of energy
        for i, soe_i in enumerate(soe):
            axes[0].plot([0] + time, soe_i, color=colors[i % len(colors)], label=f"Vehicle {i+1}")
        axes[0].set_ylabel("SoE [kWh]")

        # plot charging power
        axes[1].bar(
            [t - opt_input.dt / 2 for t in time],
            charging_power[0],
            width=opt_input.dt,
            label="Vehicle 1",
            color=colors[0],
            edgecolor="none",
        )
        for vehicle in range(1, opt_input.num_vehicles):
            axes[1].bar(
                [t - opt_input.dt / 2 for t in time],
                charging_power[vehicle],
                bottom=charging_power[vehicle - 1],
                width=opt_input.dt,
                label=f"Vehicle {vehicle + 1}",
                color=colors[vehicle % len(colors)],
                edgecolor="none",
            )
        axes[1].set_ylabel("Charging Power [kW]")

        # plot energy price
        axes[2].plot(
            get_interval_time_series([(i + 1) * dt for i in range(opt_input.num)]),
            energy_price_twice,
            c="firebrick",
            label="Energy Price",
        )
        axes[2].set_ylabel("Energy Price [$/kWh]")
        axes[2].set_ylim(min(energy_price_twice) * 0.9, max(energy_price_twice) * 1.1)

        # plot depot charge intervals
        """
        dc_dups = [sum(1 for _ in group) for _, group in groupby(data["depot_charge"])]
        dup_intervals = [
            (i * opt_input.dt, j * opt_input.dt)
            for i, j in zip(partial_sums([0] + dc_dups[:-1]), partial_sums(dc_dups))
        ]
        if data["depot_charge"][0]:
            dup_intervals = dup_intervals[::2]
        else:
            dup_intervals = dup_intervals[1::2]
        for t1, t2 in dup_intervals:
            for ax in axes:
                ax.axvspan(t1, t2, color="forestgreen", alpha=0.2)
        """

        for ax in axes:
            ax.legend()
            ax.set_xlim(0, 86400)
        axes[0].set_title("Optimization Result")
        plt.show()


if __name__ == "__main__":
    main()
