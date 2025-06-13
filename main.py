import argparse
from functools import reduce
from math import gcd

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from rich import print as printr

from optimization.optimization import OptimizationInput, OptimizationModel
from optimization.utils import expand_values


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
    args = parser.parse_args()

    data = pl.read_csv(args.data)
    energy_price = pl.read_csv(args.energy_price)

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
    opt_input = OptimizationInput(data, energy_price, 1.0)
    opt_model = OptimizationModel(opt_input)
    opt_model.set_variables()
    opt_model.set_constraints(ce_function_type=args.ce_function)
    opt_model.set_objective()

    # solve
    solution = opt_model.solve()

    if solution is None:
        printr("[gold1]no solution found")
    else:
        printr(f"[green]found solution with objective value: {solution:.2f}")

    if solution is not None:
        sns.set_style("darkgrid")
        _, axes = plt.subplots(3)
        soe = opt_model.get_state_of_energy()
        charging_power = opt_model.get_charging_power()
        energy_price_twice = []
        for ep in energy_price["energy_price"]:
            energy_price_twice += [ep, ep]
        axes[0].plot([0] + list(data["time"]), soe, c="navy", label="SoE")
        axes[0].plot(
            [0, 86400], 2 * [opt_input.battery_capacity * 0.0], c="navy", linestyle="dashed", label="SoE Lower Bound"
        )
        axes[0].plot(
            [0, 86400], 2 * [opt_input.battery_capacity * 1.0], c="navy", linestyle="dashed", label="SoE Upper Bound"
        )
        axes[1].bar(
            [t - opt_input.dt / 2 for t in data["time"]],
            charging_power,
            width=opt_input.dt,
            color="forestgreen",
            label="Charging Power",
            edgecolor="none",
        )
        axes[2].plot(get_interval_time_series(data["time"]), energy_price_twice, c="maroon", label="Energy Price")
        for ax in axes:
            ax.legend()
            ax.set_xlim(0, 86400)
        axes[0].set_title("Optimization Result")
        plt.show()


if __name__ == "__main__":
    main()
