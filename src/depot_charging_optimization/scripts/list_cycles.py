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

    for vehicle_id in vehicle_ids:
        cycle_ids = data[data[vehicle_id_column] == vehicle_id][cycle_id_column].unique()
        logger.info(f"Vehicle ID: {vehicle_id} ({len(cycle_ids)} cycles)")
        for cycle_id in cycle_ids:
            logger.info(f"  - {cycle_id}")
