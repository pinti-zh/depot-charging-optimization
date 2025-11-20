import logging
import os
from dataclasses import dataclass

import click
import pandas as pd
from pydantic import ValidationError
from rich.logging import RichHandler

from depot_charging_optimization.data_models import Input

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)  # or DEBUG

logger = logging.getLogger("optimize")


@dataclass
class ColumnArguments:
    time_column: str
    type_column: str
    demand_column: str
    capacity_column: str
    max_charging_power_column: str


def select_relevant_columns(df, columns):
    # check battery capacity
    if not all(cap == df[columns.capacity_column].iloc[0] for cap in df[columns.capacity_column]):
        raise ValueError("Battery capacity columns do not match")

    # check depot charging power
    max_charging_power = df[df[columns.type_column] == "LadungDepot"][columns.max_charging_power_column].max()
    depot_charging_power = df[df[columns.type_column] == "LadungDepot"][columns.max_charging_power_column]
    if not all(cp == max_charging_power or cp == 0 for cp in depot_charging_power):
        raise ValueError("Max charging power columns do not match")

    depot_charge = df[columns.type_column] == "LadungDepot"
    battery_capacity = df[columns.capacity_column].iloc[0]
    selection = pd.DataFrame(
        {
            "time": df[columns.time_column],
            "energy_demand": df[columns.demand_column],
            "depot_charge": depot_charge,
            "battery_capacity": [battery_capacity] * len(df),
            "max_charging_power": df[columns.max_charging_power_column].where(
                df[columns.type_column] == "LadungDepot", 0
            ),
        }
    )
    return selection


def remove_dummies(df):
    rows_to_drop = []
    for i in range(len(df) - 1):
        dt = df["time"].iloc[i + 1] - df["time"].iloc[i]
        if dt == 1:
            if df["depot_charge"].iloc[i] and not df["depot_charge"].iloc[i + 1]:
                df.loc[i, "time"] += 1
                rows_to_drop.append(i + 1)
    return df.drop(index=rows_to_drop).reset_index(drop=True)


def split_intervals_over_midnight(df):
    day = 60 * 60 * 24
    last_ts = df["time"].max()
    if last_ts < day:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "time": [day],
                        "energy_demand": [0.0],
                        "depot_charge": [True],
                        "battery_capacity": [df["battery_capacity"].iloc[0]],
                        "max_charging_power": [df["max_charging_power"].where(df["depot_charge"], 0).max()],
                    }
                ),
            ],
            ignore_index=True,
        )
    elif last_ts > day:
        for i in range(len(df) - 1):
            if df["time"].iloc[i] < day and df["time"].iloc[i + 1] > day:
                split_time = [day, df["time"].iloc[i + 1]]
                split_depot_charge = [df["depot_charge"].iloc[i + 1]] * 2
                split_battery_capacity = [df["battery_capacity"].iloc[i + 1]] * 2
                split_max_charging_power = [df["max_charging_power"].iloc[i + 1]] * 2
                dt1 = day - df["time"].iloc[i]
                dt2 = df["time"].iloc[i + 1] - day
                ed1 = df["energy_demand"].iloc[i + 1] * dt1 / (dt1 + dt2)
                ed2 = df["energy_demand"].iloc[i + 1] * dt2 / (dt1 + dt2)
                split_energy_demand = [ed1, ed2]
                df = pd.concat(
                    [
                        df.iloc[: i + 1],
                        pd.DataFrame(
                            {
                                "time": split_time,
                                "energy_demand": split_energy_demand,
                                "depot_charge": split_depot_charge,
                                "battery_capacity": split_battery_capacity,
                                "max_charging_power": split_max_charging_power,
                            }
                        ),
                        df.iloc[i + 1 :],
                    ],
                    ignore_index=True,
                )
                break
    return df


def preprocess_group(group, column_arguments):
    df = select_relevant_columns(group, column_arguments)
    df.loc[0, "depot_charge"] = True
    mcp = df["max_charging_power"].where(df["depot_charge"], 0).max()
    df.loc[0, "max_charging_power"] = mcp if mcp > 0 else 150
    df = remove_dummies(df)
    df = split_intervals_over_midnight(df)
    df["time"] = df["time"].map(lambda x: (x - 1) % (60 * 60 * 24) + 1)
    df = df.sort_values(by="time", ascending=True)
    df = df.drop_duplicates()

    # convert to SI units
    df["energy_demand"] = df["energy_demand"] * 3.6e6
    df["battery_capacity"] = df["battery_capacity"] * 3.6e6
    df["max_charging_power"] = df["max_charging_power"] * 1.0e3
    return df


@click.command()
@click.argument("source", type=str)
@click.option("--target", "-t", type=str, default="data/processed")
@click.option("--group_by", "-g", type=str, default="FahrzeugLaufID")
@click.option("--time_column", "-tc", type=str, default="zeit")
@click.option("--type_column", "-tc", type=str, default="Typ")
@click.option("--demand_column", "-dc", type=str, default="Energie.total")
@click.option("--capacity_column", "-cc", type=str, default="Batteriekapazitaet")
@click.option("--max_charging_power_column", "-mc", type=str, default="Ladegeschwindigkeit")
def main(
    source,
    target,
    group_by,
    time_column,
    type_column,
    demand_column,
    capacity_column,
    max_charging_power_column,
):
    column_arguments = ColumnArguments(
        time_column,
        type_column,
        demand_column,
        capacity_column,
        max_charging_power_column,
    )

    df = pd.read_csv(source)

    logger.info(f"loaded [magenta]{source}[/magenta]\n  {len(df.columns)} columns and {len(df)} rows")

    num_groups = 0
    num_successes = 0
    for id, group in df.groupby(group_by, sort=False):
        num_groups += 1
        group = group.copy().reset_index(drop=True)  # ensure indexing starts at 0 and is not reference to original
        logger.info(f"processing [cyan]{group_by}[/cyan] [dark_cyan]{id}")
        processed = preprocess_group(group, column_arguments)
        try:
            input_data = Input.from_dataframe(processed)

            os.makedirs(target, exist_ok=True)
            file_name = os.path.join(target, f"{id}.json")
            with open(file_name, "w") as f:
                f.write(input_data.model_dump_json(indent=4))
            logger.info(f"  [bold green]ok[/bold green]  written data to [cyan]{file_name}")
            num_successes += 1

        except ValidationError:
            logger.info("  [bold red]not ok[/bold red]  |  [orange1]skipping")

    logger.info(f"[bold]processed {num_successes}/{num_groups} groups successfully")
