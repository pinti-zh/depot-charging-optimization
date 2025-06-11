import argparse
import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.patches import Rectangle
from rich import print as printr

DAY = 60 * 60 * 24


def process_group(group, args):
    global DAY

    # assert length is at least 2
    assert len(group) >= 2

    # assert data at most a full day
    DAY = 60 * 60 * 24
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

    # remove unnecessary rows at uid switches
    current_uid = None
    for time, uid, energy_demand, step_type, charge_amount, max_charging_power in zip(
        group[args.time_column],
        group[args.cycle_id],
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
                max_charging_powers = [max_charging_power]
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
            energy_demands = energy_demands[:i] + [ed_1, ed_2] + energy_demands[i + 1 :]

            # adjust charge amounts
            ca_1 = charge_amounts[i] * ed_proportion[0]
            ca_2 = charge_amounts[i] * ed_proportion[1]
            charge_amounts = charge_amounts[:i] + [ca_1, ca_2] + charge_amounts[i + 1 :]

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
        max_charging_powers.append(max(max_charging_powers))
        left_over_charge = sum(energy_demands) - sum(charge_amounts)
        charge_amounts.append(left_over_charge)

    # modulo but keeps DAY instead of 0
    times = list(map(lambda x: (x - 1) % DAY + 1, times))
    uid_switches = list(map(lambda x: (x - 1) % DAY + 1, uid_switches))
    uid_switches = sorted(uid_switches)

    # cut at midnight and swap
    for i in range(len(times) - 1):
        if times[i + 1] < times[i]:
            times = times[i + 1 :] + times[: i + 1]
            energy_demands = energy_demands[i + 1 :] + energy_demands[: i + 1]
            depot_charge = depot_charge[i + 1 :] + depot_charge[: i + 1]
            charge_amounts = charge_amounts[i + 1 :] + charge_amounts[: i + 1]
            max_charging_powers = max_charging_powers[i + 1 :] + max_charging_powers[: i + 1]
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


def plot_processed_group(times, energy_demands, depot_charge, uid_switches, name="blank", plot_dir=""):

    midpoints = [(t1 + t2) / 2 for t1, t2 in zip([0.0] + times[:-1], times)]
    width = [t2 - t1 for t1, t2 in zip([0.0] + times[:-1], times)]
    height = [3600 * ed / (t2 - t1) for ed, t1, t2 in zip(energy_demands, [0.0] + times[:-1], times)]

    sns.set_style("darkgrid")
    _, ax = plt.subplots(figsize=(12, 8))
    ax.bar(midpoints, height, width=width, color="firebrick", label="Energy Demand")
    max_height = max(height)
    min_height = min(height)
    buffer = (max_height - min_height) / 10
    ylim = (min_height - buffer, max_height + buffer)
    for uid_switch in uid_switches:
        ax.plot([uid_switch, uid_switch], [ylim[0], ylim[1]], color="black", linestyle="dashed")
    for t1, t2, dc in zip([0.0] + times, times, depot_charge):
        if dc:
            ax.add_patch(
                Rectangle(
                    (t1, ylim[0]),
                    t2 - t1,
                    ylim[1] - ylim[0],
                    color="forestgreen",
                    alpha=0.2,
                )
            )
    # label patch
    ax.add_patch(
        Rectangle(
            (t1, ylim[0]),
            0,
            0,
            color="forestgreen",
            alpha=0.2,
            label="Depot Charge",
        )
    )
    ax.set_ylim(ylim)
    ax.set_xlim(0.0, DAY)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power Demand [kW]")
    plt.legend()
    if len(uid_switches) == 1:
        plt.title(f"{name} | 1 cycle")
    else:
        plt.title(f"{name} | {len(uid_switches)} cycles")
    if plot_dir:
        plt.savefig(f"{plot_dir}/{name}.png")
        printr(f"    saved plot to [magenta]{plot_dir}/{name}.png")
    plt.show()


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


def main():
    global DAY

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="path to source csv file")
    parser.add_argument("--target", type=str, default="data/processed", help="target directory")
    parser.add_argument("--plot_dir", type=str, default="plots/data_visualization", help="directory to save plots")
    parser.add_argument("--plot", action="store_true", help="create plots for visualization")
    parser.add_argument("--group_by", type=str, default="FahrzeugLaufID", help="column to group by")
    parser.add_argument("--cycle_id", type=str, default="UM_UID", help="column containing cycle id")
    parser.add_argument("--time_column", type=str, default="zeit", help="column containing time")
    parser.add_argument("--type_column", type=str, default="Typ", help="column containing row type")
    parser.add_argument("--demand_column", type=str, default="Energie.total", help="column containing energy demand")
    parser.add_argument(
        "--capacity_column", type=str, default="Batteriekapazitaet", help="column containing battery capacity"
    )
    parser.add_argument(
        "--charge_amount_column", type=str, default="EffektiveLademenge", help="column containing charge amount"
    )
    parser.add_argument(
        "--max_charging_power_column",
        type=str,
        default="Ladegeschwindigkeit",
        help="column containing max charging power",
    )
    args = parser.parse_args()

    data = pl.read_csv(args.source)

    printr(f"loaded [magenta]{args.source}[/magenta]\n  {len(data.columns)} columns and {len(data)} rows")
    printr("  head", data.head())

    num_groups = 0
    num_successes = 0
    for id, group in data.group_by(args.group_by, maintain_order=True):
        num_groups += 1
        printr(f"processing [cyan]{args.group_by}[/cyan] [dark_cyan]{id[0]}", end="")
        data_dict = process_group(group, args)
        try:
            eps = 1e-6
            assert abs(group[args.demand_column].sum() * 3.6e6 - sum(data_dict["energy_demand"])) < eps
            assert data_dict["time"][-1] == DAY
            assert len(set(data_dict["time"])) == len(data_dict["time"])
            assert abs(sum(data_dict["energy_demand"]) - sum(data_dict["charge_amount"])) < eps
            for key, values in data_dict.items():
                if key == "uid_switches":
                    continue
                assert len(data_dict["time"]) == len(values)
            num_successes += 1
            printr("  [bold green]ok")
            os.makedirs(args.target, exist_ok=True)
            save_processed_group_to_csv(f"{args.target}/{id[0]}.csv", data_dict)
            if args.plot:
                os.makedirs(args.plot_dir, exist_ok=True)
                plot_processed_group(
                    data_dict["time"],
                    data_dict["energy_demand"],
                    data_dict["depot_charge"],
                    data_dict["uid_switches"],
                    name=id[0],
                    plot_dir=args.plot_dir,
                )
        except AssertionError:
            printr("  [bold red]not ok [/bold red] | [orange1]skipping")

    printr(f"[bold]processed {num_successes}/{num_groups} groups successfully")


if __name__ == "__main__":
    main()
