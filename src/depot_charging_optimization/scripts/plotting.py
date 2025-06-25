import json
import logging

import click
from rich.logging import RichHandler

from depot_charging_optimization.core import Solution

# Basic Rich logging setup
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)  # or DEBUG

logger = logging.getLogger("plot")


@click.command()
@click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")
def plot_solution(solution_file):
    with open(solution_file, "r") as f:
        solution = Solution.from_dict(json.load(f))
    logger.info(f"Loaded solution from [cyan3]{solution_file}")
    logger.info(solution.total_cost)
