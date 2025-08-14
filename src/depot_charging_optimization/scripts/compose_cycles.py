import json

import click

from depot_charging_optimization.data_models import Input
from depot_charging_optimization.logging import get_logger

logger = get_logger(name="compose")


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
            subset_is_composable.append(Input.concatenatable([cycles[i] for i in subset]))
            info_string = " ".join(str(i + 1) for i in subset)
            info_string += " " * (max_length_of_partition_string - len(info_string))
            info_string = "    " + info_string
            if subset_is_composable[-1]:
                info_string += "  [green]can be concatenated"
            else:
                info_string += "  [red]cannot be concatenated"
            logger.info(info_string)
        if all(subset_is_composable):
            composable_partitions.append(partition)
            logger.info("[green]Composable")
        else:
            logger.info("[red]Not composable")
        logger.info("")

    logger.info(f"There are {len(composable_partitions)}/{number_of_partitions} partitions composable:")
    for index, partition in enumerate(composable_partitions):
        subset_strings = ["(" + " ".join(str(i + 1) for i in subset) + ")" for subset in partition]
        logger.info(f"  {index + 1}.\t|vehicles: {len(partition)}|\t" + " - ".join(subset_strings))
