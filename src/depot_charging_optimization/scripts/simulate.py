import json
import logging
import os

import click
import pandas as pd
from rich.logging import RichHandler

from depot_charging_optimization.config import FileConfig
from depot_charging_optimization.data_models import Input
from depot_charging_optimization.simulator import GreedySimulator, PeakShavingSimulator

# Basic Rich logging setup
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)  # or DEBUG

logger = logging.getLogger("simulate")


@click.command()
@click.option("energy_price_file", "-epf", type=str, default="data/energy_price.csv", help="energy price file")
@click.option(
    "--ce_function",
    "-cef",
    type=click.Choice(["constant", "quadratic", "one"], case_sensitive=False),
    default="quadratic",
)
@click.option(
    "--simulation_algorithm",
    "-sa",
    type=click.Choice(["greedy", "peak_shaving"], case_sensitive=False),
    default="greedy",
)
@click.option("--alpha", "-a", type=float, default=1.0, help="constant for charging efficiency function")
@click.option("--debug", "-d", is_flag=True, default=False, help="print debug messages")
@click.option(
    "--max_charging_power",
    "-mcp",
    type=float,
    default=1.0,
    help="maximum charging power for peak shaving (between 0 and 1)",
)
@FileConfig.as_click_options
def main(
    ce_function,
    simulation_algorithm,
    alpha,
    debug,
    max_charging_power,
    file_config,
):
    if debug:
        logger.setLevel(logging.DEBUG)
    data = []
    logger.info("Reading files:")
    for i, file in enumerate(file_config.data_files):
        with open(file) as f:
            data.append(Input.model_validate(json.load(f)))
        logger.info(f"  {i + 1}. [cyan]{file}")
    logger.info("")

    data_input = Input.combine(data)

    energy_price = pd.read_csv(file_config.energy_price_file)
    energy_price["energy_price"] /= 3.6e6

    data_input = data_input.add_energy_price(energy_price["time"].to_list(), energy_price["energy_price"].to_list())
    data_input = data_input.add_grid_tariff(1.3e-4)
    # data_input = data_input.add_grid_tariff(0.57 * 1e-3)

    # simulation
    if simulation_algorithm == "greedy":
        simulator = GreedySimulator(data_input)
    elif simulation_algorithm == "peak_shaving":
        simulator = PeakShavingSimulator(data_input, max_charging_power * data_input.max_charging_power)
    else:
        logger.error(f"Unknown simulator [{simulation_algorithm}]")

    solution = simulator.run(ce_function_type=ce_function, alpha=alpha)

    if solution is None:
        logger.error("No solution found")
    else:
        # solution information
        logger.info("Found solution")
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
