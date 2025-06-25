import contextlib
import json
import logging
import os
import sys
from functools import reduce
from math import gcd
from time import perf_counter

import click
import polars as pl
from rich.logging import RichHandler

from depot_charging_optimization.core import OptimizationInput, OptimizationModel
from depot_charging_optimization.utils import expand_values


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# Basic Rich logging setup
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)  # or DEBUG

logger = logging.getLogger("optimize")


@click.command()
@click.argument("data_files", type=str, nargs=-1)
@click.option("energy_price_file", "-epf", type=str, default="data/energy_price.csv", help="energy price file")
@click.option(
    "--ce_function", "-cef", type=click.Choice(["constant", "quadratic", "one"], case_sensitive=False), default="one"
)
@click.option("--alpha", "-a", type=float, default=1.0, help="constant for charging efficiency function")
@click.option("--time_limit", "-tl", type=int, default=5, help="solver time limit in seconds")
@click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")
def optimize(data_files, energy_price_file, ce_function, alpha, time_limit, solution_file):
    logger.info("Loading the following files:")
    for i, file in enumerate(data_files):
        logger.info(f"  {i+1}. [cyan3]{file}")
    logger.info("")
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
    opt_input = OptimizationInput.from_dataframes(expanded_data, energy_price, 0.2e-4)
    start = perf_counter()
    with suppress_stdout_stderr():
        opt_model = OptimizationModel(opt_input)
    opt_model.model.setParam("LogToConsole", 0)
    opt_model.model.setParam("OutputFlag", 1)
    opt_model.model.setParam("TimeLimit", time_limit)
    opt_model.set_variables()
    opt_model.set_constraints(ce_function_type=ce_function, alpha=alpha)
    opt_model.set_objective()

    # solve
    solution = opt_model.solve()
    optimization_time = perf_counter() - start

    if solution is None:
        logger.error("No solution found")
    else:
        logger.info(f"Found solution in {optimization_time:.4f} seconds")
        total_cost = f"{solution.total_cost:.3f} $"
        energy_cost = f"{solution.energy_cost:.3f} $"
        power_cost = f"{solution.power_cost:.3f} $"
        max_cost_string_length = max(map(len, [total_cost, energy_cost, power_cost]))
        logger.info(f"Total cost of solution:   {' ' * (max_cost_string_length - len(total_cost))}{total_cost}")
        logger.info(f"Energy cost of solution:  {' ' * (max_cost_string_length - len(energy_cost))}{energy_cost}")
        logger.info(f"Power cost of solution:   {' ' * (max_cost_string_length - len(power_cost))}{power_cost}")

        solution_dir = os.path.dirname(solution_file)
        os.makedirs(solution_dir, exist_ok=True)
        with open(solution_file, "w") as f:
            json.dump(solution.to_dict(), f)
        logger.info(f"Saved solution to [cyan3]{solution_file}")
