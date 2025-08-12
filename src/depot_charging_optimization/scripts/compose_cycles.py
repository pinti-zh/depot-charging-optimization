import json
import logging

import click
from rich.logging import RichHandler

from depot_charging_optimization.data_models import Input

# Basic Rich logging setup
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)  # or DEBUG

logger = logging.getLogger("list_cycles")


def composable(cycles):
    if len(cycles) < 2:
        return True
    if len(cycles) > 2:
        for i, cycle_1 in enumerate(cycles):
            for cycle_2 in cycles[i + 1 :]:
                if not composable([cycle_1, cycle_2]):
                    return False
        return True
    cycle_1, cycle_2 = cycles[0], cycles[1]
    time_index_1, time_index_2 = 0, 0
    while (time_index_1 < cycle_1.num_timesteps) and (time_index_2 < cycle_2.num_timesteps):
        time_1 = cycle_1.time[time_index_1]
        time_2 = cycle_2.time[time_index_2]
        depot_charge_1 = cycle_1.depot_charge[0][time_index_1]
        depot_charge_2 = cycle_2.depot_charge[0][time_index_2]
        if not (depot_charge_1 or depot_charge_2):
            return False
        if time_1 <= time_2:
            time_index_1 += 1
        if time_2 <= time_1:
            time_index_2 += 1
    return True


def partitions(elements):
    if len(elements) < 1:
        yield []
        return
    for smaller in partitions(elements[1:]):
        # add first element to each subset
        for i, subset in enumerate(smaller):
            yield smaller[:i] + [[elements[0]] + subset] + smaller[i + 1 :]
        # add first element as its own subset
        yield [[elements[0]]] + smaller


@click.command()
@click.argument("data_files", type=str, nargs=-1, required=True)
def compose_cycles(data_files):
    cycles = []
    logger.info("Reading files:")
    for i, file in enumerate(data_files):
        with open(file) as f:
            cycles.append(Input.model_validate(json.load(f)))
        logger.info(f"  {i + 1}. [cyan]{file}")
    logger.info("")

    number_of_partitions = 0
    max_length_of_partition_string = len(" ".join(str(i + 1) for i in range(len(cycles))))
    composable_partitions = []

    for partition in partitions(list(range(len(cycles)))):
        number_of_partitions += 1
        logger.info(f"Number of subsets in partition: {len(partition)}")
        subset_is_composable = []
        for subset in partition:
            subset_is_composable.append(composable([cycles[i] for i in subset]))
            info_string = " ".join(str(i + 1) for i in subset)
            info_string += " " * (max_length_of_partition_string - len(info_string))
            info_string = "    " + info_string
            if subset_is_composable[-1]:
                info_string += "  [green]composable"
            else:
                info_string += "  [red]not composable"
            logger.info(info_string)
        if all(subset_is_composable):
            composable_partitions.append(partition)
            logger.info("[green]Ok")
        else:
            logger.info("[red]Not ok")
        logger.info("")

    logger.info(f"There are {len(composable_partitions)}/{number_of_partitions} partitions composable:")
    for index, partition in enumerate(composable_partitions):
        subset_strings = ["(" + " ".join(str(i + 1) for i in subset) + ")" for subset in partition]
        logger.info(f"  {index + 1}.\t|vehicles: {len(partition)}|\t" + " - ".join(subset_strings))
