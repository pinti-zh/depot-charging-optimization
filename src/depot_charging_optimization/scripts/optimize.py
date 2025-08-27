import json
import os
from pathlib import Path
from time import perf_counter

import click
import pandas as pd

from depot_charging_optimization.core import CasadiOptimizer, GurobiOptimizer
from depot_charging_optimization.data_models import Input
from depot_charging_optimization.logging import get_logger, log_stdout
from depot_charging_optimization.result_store import ResultStore


@click.command()
@click.argument("data_files", type=str, nargs=-1)
@click.option("energy_price_file", "-epf", type=str, default="data/energy_price.csv", help="energy price file")
@click.option(
    "--ce_function", "-cef", type=click.Choice(["constant", "quadratic", "one"], case_sensitive=False), default="one"
)
@click.option("--alpha", "-a", type=float, default=1.0, help="constant for charging efficiency function")
@click.option("--time_limit", "-tl", type=int, default=5, help="solver time limit in seconds")
@click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")
@click.option("--use_casadi", "-uc", is_flag=True, default=False, help="use casadi instead of gurobi")
@click.option("--bidirectional_charging", "-bc", is_flag=True, default=False, help="allow no bidirectional charging")
@click.option("--debug", "-d", is_flag=True, default=False, help="print debug messages")
@click.option("--confidence_level", "-cl", type=float, default=0.0, help="confidence level for stochastic robustness")
@click.option(
    "--energy_std_dev", "-esd", type=float, default=0.0, help="energy standard deviation for stochastic robustness"
)
def optimize(
    data_files,
    energy_price_file,
    ce_function,
    alpha,
    time_limit,
    solution_file,
    use_casadi,
    bidirectional_charging,
    debug,
    confidence_level,
    energy_std_dev,
):
    if debug:
        logger = get_logger(name="optimize", level="debug")
    else:
        logger = get_logger(name="optimize", level="info")
    data = []
    logger.info("Reading files:")
    for i, file in enumerate(data_files):
        with open(file) as f:
            data.append(Input.model_validate(json.load(f)))
        logger.info(f"  {i + 1}. [cyan]{file}")
    logger.info("")

    data_input = Input.combine(data)

    energy_price = pd.read_csv(energy_price_file)
    energy_price["energy_price"] /= 3.6e6

    data_input = data_input.add_energy_price(energy_price["time"].to_list(), energy_price["energy_price"].to_list())
    data_input = data_input.add_grid_tariff(1.2e-4)
    # data_input = data_input.add_grid_tariff((17.0 / 30) * 1e-3)

    # optimization
    start = perf_counter()
    if use_casadi:
        optimizer = CasadiOptimizer(
            data_input,
            bidirectional_charging=bidirectional_charging,
            confidence_level=confidence_level,
            energy_std_dev=energy_std_dev,
        )
    else:
        optimizer = GurobiOptimizer(
            data_input,
            bidirectional_charging=bidirectional_charging,
            time_limit=time_limit,
            confidence_level=confidence_level,
            energy_std_dev=energy_std_dev,
        )

    optimizer.build(ce_function_type=ce_function, alpha=alpha)

    # solve
    with log_stdout(logger, level="debug"):
        solution = optimizer.solve()
    optimization_time = perf_counter() - start

    if solution is None:
        logger.error("No solution found")
    else:
        # slack information
        max_slack = 0
        max_slack_location = None
        for sublist in optimizer.slack["state_of_energy"]:
            if max(sublist) > max_slack:
                max_slack = max(sublist)
                max_slack_location = "State of Energy"
        for sublist in optimizer.slack["charging_power"]:
            if max(sublist) > max_slack:
                max_slack = max(sublist)
                max_slack_location = "Charging Power"
        max_slack = max(max_slack, optimizer.slack["max_charging_power"])
        if optimizer.slack["max_charging_power"] > max_slack:
            max_slack = optimizer.slack["max_charging_power"]
            max_slack_location = "Max Charging Power"
        logger.debug(f"Maximum slack: {max_slack:.3e} (found in [{max_slack_location}] constraints)")

        # solution information
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

        result_store = ResultStore(Path("logs/optimize.log"))
        result_store.write(
            {
                "solution_total_cost": solution.total_cost,
                "input": {
                    "data_files": data_files,
                    "energy_price_file": energy_price_file,
                    "ce_function": ce_function,
                    "alpha": alpha,
                    "time_limit": time_limit,
                    "solution_file": solution_file,
                    "use_casadi": use_casadi,
                    "bidirectional_charging": bidirectional_charging,
                    "debug": debug,
                },
            }
        )
