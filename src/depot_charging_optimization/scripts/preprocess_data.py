import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import click
import polars as pl
import seaborn as sns
from matplotlib.patches import Rectangle
from rich import print as printr

DAY = 60 * 60 * 24


@dataclass
class ColumnArguments:
    cycle_id_column: str
    time_column: str
    type_column: str
    demand_column: str
    capacity_column: str
    charge_amount_column: str
    max_charging_power_column: str


def process_group(group, args):
    global DAY

    # assert length is at least 2
    assert len(group) >= 2

    # assert data at most a full day
    assert group[args.time_column].max() - group[args.time_column].min() <= DAY

    # assert equal battery capacity for all rows
    assert all(
        group[args.capacity_column].map_elements(
            lambda x: x == group[args.capacity_column][0], return_dtype=pl.Boolean
        )
    )
    battery_capacity = group[args.capacity_column][0]

    # assert chronological order
    for t1, t2 in zip(group[args.time_column], group[args.time_column][1:]):
        assert t1 <= t2

    times = []
    energy_demands = []
    depot_charge = []
    charge_amounts = []
    max_charging_powers = []
    uid_switches = []

    try:
        max_depot_charging_power = max(
            [
                mcp
                for dc, mcp in zip(group[args.type_column], group[args.max_charging_power_column])
                if dc == "LadungDepot"
            ]
        )
    except ValueError:
        max_depot_charging_power = 150

    # remove unnecessary rows at uid switches
    current_uid = None
    for time, uid, energy_demand, step_type, charge_amount, max_charging_power in zip(
        group[args.time_column],
        group[args.cycle_id_column],
        group[args.demand_column],
        group[args.type_column],
        group[args.charge_amount_column],
        group[args.max_charging_power_column],
    ):
        if uid != current_uid:
            current_uid = uid
            uid_switches.append(time)
            if len(times):
                times = times[:-1] + [time]
            else:
                times = [time]
                energy_demands = [0.0]
                depot_charge = [True]
                charge_amounts = [charge_amount]
                max_charging_powers = [max_depot_charging_power]
        else:
            times.append(time)
            energy_demands.append(energy_demand)
            depot_charge.append(step_type == "LadungDepot")
            charge_amounts.append(charge_amount)
            max_charging_powers.append(max_charging_power)

    assert len(times) == len(energy_demands) == len(depot_charge)
    assert times[0] < DAY

    # fill up rest of day not covered by data and split steps going over midnight
    for i in range(len(times)):
        if times[i] > DAY:

            # proportional divison of energy demand
            dt = times[i] - times[i - 1]
            ed_proportion = ((DAY - times[i - 1]) / dt, (times[i] - DAY) / dt)
            ed_1 = energy_demands[i] * ed_proportion[0]
            ed_2 = energy_demands[i] * ed_proportion[1]
            energy_demands = energy_demands[:i] + [ed_1, ed_2] + energy_demands[i + 1:]

            # adjust charge amounts
            ca_1 = charge_amounts[i] * ed_proportion[0]
            ca_2 = charge_amounts[i] * ed_proportion[1]
            charge_amounts = charge_amounts[:i] + [ca_1, ca_2] + charge_amounts[i + 1:]

            # adjust max charging powers
            max_charging_powers = max_charging_powers[:i] + [max_charging_powers[i]] + max_charging_powers[i:]

            # adjust depot charge
            depot_charge = depot_charge[:i] + [depot_charge[i]] + depot_charge[i:]

            # adjust times
            times = times[:i] + [DAY] + times[i:]
            break
    else:
        times.append(DAY)
        energy_demands.append(0.0)
        depot_charge.append(True)
        max_charging_powers.append(max_depot_charging_power)
        left_over_charge = sum(energy_demands) - sum(charge_amounts)
        charge_amounts.append(left_over_charge)

    # modulo but keeps DAY instead of 0
    times = list(map(lambda x: (x - 1) % DAY + 1, times))
    uid_switches = list(map(lambda x: (x - 1) % DAY + 1, uid_switches))
    uid_switches = sorted(uid_switches)

    # cut at midnight and swap
    for i in range(len(times) - 1):
        if times[i + 1] < times[i]:
            times = times[i + 1:] + times[: i + 1]
            energy_demands = energy_demands[i + 1:] + energy_demands[: i + 1]
            depot_charge = depot_charge[i + 1:] + depot_charge[: i + 1]
            charge_amounts = charge_amounts[i + 1:] + charge_amounts[: i + 1]
            max_charging_powers = max_charging_powers[i + 1:] + max_charging_powers[: i + 1]
            break

    return {
        "time": times,
        "energy_demand": [ed * 3.6e6 for ed in energy_demands],  # convert from kWh to J
        "depot_charge": depot_charge,
        "charge_amount": [ca * 3.6e6 for ca in charge_amounts],  # convert from kWh to J
        "max_charging_power": [mcp * 1.0e3 for mcp in max_charging_powers],  # convert from kW to W
        "battery_capacity": [battery_capacity * 3.6e6] * len(times),  # convert from kWh to J
        "uid_switches": uid_switches,
    }


def save_processed_group_to_csv(file_name, data_dict):
    cycles = []
    cycle = 0
    for time in data_dict["time"]:
        cycles.append(cycle)
        if len(data_dict["uid_switches"]) > cycle and time == data_dict["uid_switches"][cycle]:
            cycle += 1
    cycles = list(map(lambda x: x % max(cycles), cycles))

    del data_dict["uid_switches"]
    data_dict["cycle"] = cycles

    df = pl.DataFrame(data_dict)
    df.write_csv(file_name)
    printr(f"    saved processed data to [green]{file_name}")


@click.command()
@click.argument("source", type=str)
@click.option("--target", "-t", type=str, default="data/processed")
@click.option("--group_by", "-g", type=str, default="FahrzeugLaufID")
@click.option("--cycle_id_column", "-ic", type=str, default="UM_UID")
@click.option("--time_column", "-tc", type=str, default="zeit")
@click.option("--type_column", "-tc", type=str, default="Typ")
@click.option("--demand_column", "-dc", type=str, default="Energie.total")
@click.option("--capacity_column", "-cc", type=str, default="Batteriekapazitaet")
@click.option("--charge_amount_column", "-ac", type=str, default="EffektiveLademenge")
@click.option("--max_charging_power_column", "-mc", type=str, default="Ladegeschwindigkeit")
def preprocess_data(
    source,
    target,
    group_by,
    cycle_id_column,
    time_column,
    type_column,
    demand_column,
    capacity_column,
    charge_amount_column,
    max_charging_power_column,
):
    global DAY

    column_arguments = ColumnArguments(
        cycle_id_column,
        time_column,
        type_column,
        demand_column,
        capacity_column,
        charge_amount_column,
        max_charging_power_column,
    )

    data = pl.read_csv(source)

    printr(f"loaded [magenta]{source}[/magenta]\n  {len(data.columns)} columns and {len(data)} rows")
    printr("  head", data.head())

    num_groups = 0
    num_successes = 0
    for id, group in data.group_by(group_by, maintain_order=True):
        num_groups += 1
        printr(f"processing [cyan]{group_by}[/cyan] [dark_cyan]{id[0]}", end="")
        data_dict = process_group(group, column_arguments)
        try:
            eps = 1e-6
            assert abs(group[column_arguments.demand_column].sum() * 3.6e6 - sum(data_dict["energy_demand"])) < eps
            assert data_dict["time"][-1] == DAY
            assert len(set(data_dict["time"])) == len(data_dict["time"])
            assert abs(sum(data_dict["energy_demand"]) - sum(data_dict["charge_amount"])) < eps
            for key, values in data_dict.items():
                if key == "uid_switches":
                    continue
                assert len(data_dict["time"]) == len(values)
            num_successes += 1
            printr("  [bold green]ok")
            os.makedirs(target, exist_ok=True)
            save_processed_group_to_csv(f"{target}/{id[0]}.csv", data_dict)
        except AssertionError:
            printr("  [bold red]not ok [/bold red] | [orange1]skipping")

    printr(f"[bold]processed {num_successes}/{num_groups} groups successfully")
