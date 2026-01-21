import json
import os

import click
import pandas as pd

from depot_charging_optimization.config import EnvironmentConfig, FileConfig, HeuristicConfig
from depot_charging_optimization.data_models import Input
from depot_charging_optimization.environment import Environment
from depot_charging_optimization.logging import get_logger
from depot_charging_optimization.simulator import (
    HeuristicFunction,
    charge_on_arrival,
)


def run_main(
    debug: bool,
    file_config: FileConfig,
    env_config: EnvironmentConfig,
    heuristic_config: HeuristicConfig,
):
    if debug:
        logger = get_logger(name="simulate", level="debug")
    else:
        logger = get_logger(name="simulate", level="info")

    # log config
    logger.debug("File Config:")
    logger.debug(file_config)
    logger.debug("Environment Config:")
    logger.debug(env_config)
    logger.debug("Heuristic Config:")
    logger.debug(heuristic_config)

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

    data_input = Input.combine(data)

    energy_price = pd.read_csv(file_config.energy_price_file)
    energy_price["energy_price"] /= 3.6e6  # convert to CHF / Joule

    grid_tariff = pd.read_csv(file_config.grid_tariff_file)
    grid_tariff["grid_tariff"] /= 365 * 1.0e6  # convert to CHF / Watt

    data_input = data_input.add_energy_price(energy_price["time"].to_list(), energy_price["energy_price"].to_list())
    data_input = data_input.add_grid_tariff(grid_tariff["grid_tariff"][0])

    env = Environment(data_input, config=env_config)

    # heuristic algorithm
    heuristic: HeuristicFunction = charge_on_arrival

    env.reset(data_input.battery_capacity)

    # initial try with full battery start
    logger.info("running simulation once with initially full battery")
    for t in range(env.plan.num_timesteps):
        policy = heuristic(env)
        env.step(policy)

    # simulation
    assert env.state is not None
    logger.info(f"running simulation with initial state: {env.state}")
    env.reset(env.state.state_of_energy)
    for t in range(env.plan.num_timesteps):
        logger.debug(f"Step {t}")
        logger.debug(env.state)
        policy = heuristic(env)
        env.step(policy)

    solution = env.get_solution()

    # print solution
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
@EnvironmentConfig.as_click_options
@HeuristicConfig.as_click_options
def main(
    debug: bool,
    file_config_cli_arguments: dict,
    env_config_cli_arguments: dict,
    heuristic_config_cli_arguments: dict,
):
    file_config = FileConfig.load_from_dict(file_config_cli_arguments)
    env_config = EnvironmentConfig.load_from_dict(env_config_cli_arguments)
    heuristic_config = HeuristicConfig.load_from_dict(heuristic_config_cli_arguments)
    return run_main(debug, file_config, env_config, heuristic_config)
