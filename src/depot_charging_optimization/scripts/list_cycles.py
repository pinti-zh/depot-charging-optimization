import logging

import click
import pandas as pd
from rich.logging import RichHandler

# Basic Rich logging setup
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)  # or DEBUG

logger = logging.getLogger("list_cycles")


@click.command()
@click.option("--file", "-f", type=str, default="data/clean/vehicle_cycle_energy.csv", help="input csv file")
@click.option("--cycle_id_column", "-cid", type=str, default="UM_UID", help="column containing cycle ids")
@click.option("--vehicle_id_column", "-vid", type=str, default="FahrzeugLaufID", help="column containing vehicle ids")
def list_cycles(file, cycle_id_column, vehicle_id_column):
    data = pd.read_csv(file)
    vehicle_ids = data[vehicle_id_column].unique()
    depot_cycles = {}
    for vehicle_id in vehicle_ids:
        depot_id = vehicle_id.split("_")[0]
        cycles = data[data[vehicle_id_column] == vehicle_id][cycle_id_column].unique().tolist()
        if depot_id in depot_cycles.keys():
            depot_cycles[depot_id] += cycles
        else:
            depot_cycles[depot_id] = cycles

    for depot_id, cycle_ids in depot_cycles.items():
        logger.info(f"Depot ID: {depot_id} ({len(cycle_ids)} cycles)")
        for cycle_id in cycle_ids:
            logger.info(f"  - {cycle_id}")
