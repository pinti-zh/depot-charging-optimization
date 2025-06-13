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
    parser.add_argument("--data", "-d", type=str, required=True, help="path to data file")
    parser.add_argument("--energy_price", "-ep", type=str, default="data/energy_price.csv", help="energy price file")
    parser.add_argument("--ce_function", "-cef", type=str, choices=["constant", "quadratic", "one"], default="one")
    parser.add_argument("--plot", "-p", action="store_true", help="plot results")
    args = parser.parse_args()

    data = pl.read_csv(args.data)
    energy_price = pl.read_csv(args.energy_price)
    energy_price = energy_price.with_columns(pl.col("energy_price").truediv(3.6e6))

    dt = reduce(gcd, list(data["time"]) + list(energy_price["time"]))
    data = pl.DataFrame(
        {
            "time": expand_values(data["time"], data["time"], dt, interpolation="linear"),
            "energy_demand": expand_values(data["time"], data["energy_demand"], dt, interpolation="split"),
            "depot_charge": expand_values(data["time"], data["depot_charge"], dt),
            "charge_amount": expand_values(data["time"], data["charge_amount"], dt, interpolation="split"),
            "battery_capacity": expand_values(data["time"], data["battery_capacity"], dt),
            "max_charging_power": expand_values(data["time"], data["max_charging_power"], dt),
            "cycle": expand_values(data["time"], data["cycle"], dt),
        }
    )
    energy_price = pl.DataFrame(
        {
            "time": expand_values(energy_price["time"], energy_price["time"], dt, interpolation="linear"),
            "energy_price": expand_values(energy_price["time"], energy_price["energy_price"], dt),
        }
    )

    # optimization
    opt_input = OptimizationInput(data, energy_price, 0.2e-3)
    ok, reason = opt_input.is_feasible()
    if not ok:
        printr(f"[gold1]optimization input is not feasible: {reason}")
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
        printr(f"              [navajo_white3]naive greedy solution: {opt_input.naive_greedy_solution()}")

    if solution is not None and args.plot:
        sns.set_style("darkgrid")
        _, axes = plt.subplots(3)

        joule_to_kwh = 1.0 / 3.6e6

        soe = opt_model.get_state_of_energy() * joule_to_kwh
        charging_power = opt_model.get_charging_power() / 1000.0
        energy_price_twice = []
        for ep in energy_price["energy_price"]:
            energy_price_twice += [ep / joule_to_kwh, ep / joule_to_kwh]

        # plot state of energy
        axes[0].plot([0] + list(data["time"]), soe, c="navy", label="SoE")
        axes[0].plot(
            [0, 86400], 2 * [opt_input.soe_lb * joule_to_kwh], c="navy", linestyle="dashed", label="SoE Lower Bound"
        )
        axes[0].plot(
            [0, 86400], 2 * [opt_input.soe_ub * joule_to_kwh], c="navy", linestyle="dashed", label="SoE Upper Bound"
        )
        axes[0].set_ylabel("SoE [kWh]")
        axes[0].set_ylim(0, opt_input.battery_capacity * joule_to_kwh)

        # plot charging power
        axes[1].bar(
            [t - opt_input.dt / 2 for t in data["time"]],
            charging_power,
            width=opt_input.dt,
            color="forestgreen",
            label="Charging Power",
            edgecolor="none",
        )
        axes[1].set_ylabel("Charging Power [kW]")
        axes[1].set_ylim(
            0, max([mcp for dc, mcp in zip(opt_input.depot_charge, opt_input.max_charging_power) if dc]) / 1000.0
        )

        # plot energy price
        axes[2].plot(get_interval_time_series(data["time"]), energy_price_twice, c="maroon", label="Energy Price")
        axes[2].set_ylabel("Energy Price [$/kWh]")
        axes[2].set_ylim(min(energy_price_twice) * 0.9, max(energy_price_twice) * 1.1)

        # plot depot charge intervals
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

        for ax in axes:
            ax.legend()
            ax.set_xlim(0, 86400)
        axes[0].set_title("Optimization Result")
        plt.show()


if __name__ == "__main__":
    main()
