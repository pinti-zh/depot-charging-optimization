import json
import os
from time import perf_counter

import click
import pandas as pd

from depot_charging_optimization.config import FileConfig, OptimizerConfig
from depot_charging_optimization.data_models import Input
from depot_charging_optimization.logging import get_logger, log_stdout
from depot_charging_optimization.optimizer.base import Optimizer
from depot_charging_optimization.optimizer.casadi import CasadiOptimizer
from depot_charging_optimization.optimizer.gurobi import GurobiOptimizer


def run_main(
    debug: bool,
    file_config_cli_arguments: dict,
    optimizer_config_cli_arguments: dict,
):
    if debug:
        logger = get_logger(name="optimize", level="debug")
    else:
        logger = get_logger(name="optimize", level="info")

    if not file_config_cli_arguments["config_file"].exists():
        logger.warning(f"File config file {file_config_cli_arguments['config_file']} not found")

    if not optimizer_config_cli_arguments["config_file"].exists():
        logger.warning(f"Optimizer config file {optimizer_config_cli_arguments['config_file']} not found")

    file_config = FileConfig.load_from_dict(file_config_cli_arguments)
    optimizer_config = OptimizerConfig.load_from_dict(optimizer_config_cli_arguments)

    # log config
    logger.debug("File Config:")
    logger.debug(file_config)
    logger.debug("Optimizer Config:")
    logger.debug(optimizer_config)

    data = []

    if len(file_config.data_files) < 1:
        logger.error("No data files specified.")
        return

    logger.info("Reading files:")
    for i, file in enumerate(file_config.data_files):
        with open(file) as f:
            data.append(Input.model_validate(json.load(f)))
        logger.info(f"  {i + 1}. [cyan]{file}")
    logger.info("")

    input_data = Input.combine(data)

    energy_price = pd.read_csv(file_config.energy_price_file)
    energy_price["energy_price"] /= 3.6e6  # convert to CHF / Joule

    grid_tariff = pd.read_csv(file_config.grid_tariff_file)
    grid_tariff["grid_tariff"] /= 365 * 1.0e6  # convert to CHF / Watt

    input_data = input_data.add_energy_price(energy_price["time"].to_list(), energy_price["energy_price"].to_list())
    input_data = input_data.add_grid_tariff(grid_tariff["grid_tariff"][0])

    # optimization
    optimizer: Optimizer | None = None
    match optimizer_config.optimizer_type:
        case "casadi":
            optimizer = CasadiOptimizer(input_data, config=optimizer_config)
        case "gurobi":
            optimizer = GurobiOptimizer(input_data, config=optimizer_config)
        case _:
            logger.error(f"Unknown optimizer type: {optimizer_config.optimizer_type}")
            return

    assert optimizer is not None

    build_start = perf_counter()
    optimizer.build()
    build_time = perf_counter() - build_start

    with log_stdout(logger, level="debug"):
        solve_start = perf_counter()
        solution = optimizer.solve()
        solve_time = perf_counter() - solve_start

    logger.info(f"Optimizer build time: {build_time:.4f} seconds")
    logger.info(f"Optimizer solve time: {solve_time:.4f} seconds")

    if solution is None:
        logger.error("No feasible solution found")
    else:
        total_cost = f"{solution.total_cost:.3f} $"
        energy_cost = f"{solution.energy_cost:.3f} $"
        power_cost = f"{solution.power_cost:.3f} $"

        max_cost_string_length = max(map(len, [total_cost, energy_cost, power_cost]))
        logger.info(f"Total cost of solution:   {' ' * (max_cost_string_length - len(total_cost))}{total_cost}")
        logger.info(f"Energy cost of solution:  {' ' * (max_cost_string_length - len(energy_cost))}{energy_cost}")
        logger.info(f"Power cost of solution:   {' ' * (max_cost_string_length - len(power_cost))}{power_cost}")

        solution_dir = os.path.dirname(file_config.solution_file)
        os.makedirs(solution_dir, exist_ok=True)
        with open(file_config.solution_file, "w") as f:
            f.write(solution.model_dump_json(indent=4))
        logger.info(f"Saved solution to [cyan3]{file_config.solution_file}")


@click.command()
@click.option("--debug", is_flag=True, default=False, help="print debug messages")
@FileConfig.as_click_options
@OptimizerConfig.as_click_options
def main(
    debug: bool,
    file_config_cli_arguments: dict,
    optimizer_config_cli_arguments: dict,
):
    return run_main(debug, file_config_cli_arguments, optimizer_config_cli_arguments)
