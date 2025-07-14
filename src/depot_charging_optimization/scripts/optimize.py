import contextlib
import json
import logging
import os
import sys
from time import perf_counter

import click
import pandas as pd
from rich.logging import RichHandler

from depot_charging_optimization.core import CasadiOptimizer, GurobiOptimizer
from depot_charging_optimization.data_models import Input


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
@click.option("--greedy", "-g", is_flag=True, default=False, help="use a greedy algorithm")
@click.option("--charging_power_throttle", "-cpt", type=float, default=1.0, help="throttle for charging power")
@click.option("--use_casadi", "-uc", is_flag=True, default=False, help="use casadi instead of gurobi")
def optimize(
    data_files,
    energy_price_file,
    ce_function,
    alpha,
    time_limit,
    solution_file,
    greedy,
    charging_power_throttle,
    use_casadi,
):
    data = []
    for i, file in enumerate(data_files):
        logger.info(f"{i+1}. Loading [cyan3]{file}")
        with open(file) as f:
            data.append(Input.model_validate(json.load(f)))
    logger.info("")

    data_input = Input.combine(data)

    energy_price = pd.read_csv(energy_price_file)
    energy_price["energy_price"] /= 3.6e6

    data_input = data_input.add_energy_price(energy_price["time"].to_list(), energy_price["energy_price"].to_list())
    data_input = data_input.add_grid_tariff(1.2e-4)

    # optimization
    start = perf_counter()
    if use_casadi:
        opt_model = CasadiOptimizer(data_input, greedy=greedy)
    else:
        with suppress_stdout_stderr():
            opt_model = GurobiOptimizer(data_input, greedy=greedy)
            opt_model.model.setParam("LogToConsole", 0)
            opt_model.model.setParam("OutputFlag", 1)
            opt_model.model.setParam("TimeLimit", time_limit)
    opt_model.set_variables()
    opt_model.set_constraints(ce_function_type=ce_function, alpha=alpha, cp_throttle=charging_power_throttle)
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
            f.write(solution.model_dump_json(indent=4))
        logger.info(f"Saved solution to [cyan3]{solution_file}")
