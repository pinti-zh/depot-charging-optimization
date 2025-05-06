import argparse
import json

import gurobipy as gp
from gurobipy import GRB
from rich import print

from utils import plot_result, print_model_summary, print_params


def charging_efficiency(cp, function_type, alpha, max_charging_power):
    match function_type:
        case "constant":
            return 1
        case "quadratic":
            return 1 - alpha * (cp / max_charging_power) ** 2
        case _:
            raise ValueError(f"Unknown charging efficiency function type: {function_type}")


def get_charging_indices(num_timesteps, energy_demands):
    non_charging_indices = []
    for energy_demand in energy_demands:
        for i in range(energy_demand["start"], energy_demand["end"]):
            non_charging_indices.append(i)
    return [i for i in range(num_timesteps) if i not in non_charging_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, required=True, help="path to data file")
    parser.add_argument(
        "--verbosity", "-v", choices=[0, 1, 2], type=int, default=1, help="0 = only output, 1 = verbose, 2 = debug"
    )
    parser.add_argument(
        "--charging_efficiency", "-ce", type=str, choices=["constant", "quadratic"], default="constant"
    )
    args = parser.parse_args()

    with open(args.data, "r") as f:
        data = json.load(f)

    # optimization
    model = gp.Model("DepotCharge")

    # charging mask
    charging_indices = get_charging_indices(data["numTimeSteps"], data["energyDemand"])

    # decision variables
    charging_power = []
    for i in charging_indices:
        charging_power.append(
            model.addVar(name=f"chargingPower_{i}", vtype=GRB.CONTINUOUS, lb=0, ub=data["maxChargingPower"])
        )

    state_of_energy = []
    for i in range(data["numTimeSteps"] + 1):
        state_of_energy.append(
            model.addVar(
                name=f"stateOfEnergy_{i}",
                vtype=GRB.CONTINUOUS,
                lb=data["stateOfEnergyLowerBound"],
                ub=data["stateOfEnergyUpperBound"],
            )
        )

    # aux variables
    max_charging_power = model.addVar(name="maxChargingPower", vtype=GRB.CONTINUOUS, lb=0)
    for i, cp in zip(charging_indices, charging_power):
        model.addConstr(max_charging_power >= cp, f"maxPower_{i}")  # relaxed max constraint

    # constraints
    for i, energy_demand in enumerate(data["energyDemand"]):
        model.addConstr(
            state_of_energy[energy_demand["end"]] == state_of_energy[energy_demand["start"]] - energy_demand["value"],
            f"energyDemand_{i}",
        )
    for i, cp in zip(charging_indices, charging_power):
        model.addConstr(
            state_of_energy[i + 1]
            == state_of_energy[i]
            + charging_efficiency(cp, args.charging_efficiency, 0.25, data["maxChargingPower"])
            * cp
            * data["timeStepDuration"],
            f"charging_{i}",
        )

    model.addConstr(state_of_energy[0] <= state_of_energy[data["numTimeSteps"]], "energyLoop")

    # objective
    model.setObjective(
        gp.quicksum(
            data["energyPrice"][i] * cp * data["timeStepDuration"] for i, cp in zip(charging_indices, charging_power)
        )
        + max_charging_power * data["powerGridTariff"],
        GRB.MINIMIZE,
    )

    # display model
    print_model_summary(model, args.verbosity)

    # print params
    if args.verbosity >= 1:
        print_params(data)

    # solve
    model.optimize()

    # print solution
    try:
        if args.verbosity >= 2:
            soe_strings = list(map(lambda x: f"{x.X:.2f}", state_of_energy))
            cp_strings = []
            cp_index = 0
            for i in range(data["numTimeSteps"]):
                if i in charging_indices:
                    cp_strings.append(f"{charging_power[cp_index].X:.2f}")
                    cp_index += 1
                else:
                    cp_strings.append(" ")
            max_string_len = max(max(map(len, soe_strings)), max(map(len, cp_strings)))
            max_string_len = max(max(map(len, soe_strings)), max(map(len, cp_strings)))
            for soe_string in soe_strings:
                buffer = " " * (max_string_len - len(soe_string))
                print(f"[bold blue]{buffer}{soe_string}{' ' * max_string_len}", end="")
            print()
            for cp_string in cp_strings:
                buffer = " " * (max_string_len - len(cp_string))
                print(f"[bold red]{' ' * max_string_len}{buffer}{cp_string}", end="")
            print(f"[bold red]{' ' * max_string_len} -> {sum([cp.X for cp in charging_power]):.2f}")
            print("[bold]" + "-" * 100)
        if args.verbosity >= 1:
            grid_cost = max_charging_power.X * data["powerGridTariff"]
            energy_cost = sum(
                data["energyPrice"][i] * cp.X * data["timeStepDuration"]
                for i, cp in zip(charging_indices, charging_power)
            )
            print(f"Energy Cost:        [bold green]{energy_cost:.2f}$")
            print(f"Grid Cost:        + [bold green]{grid_cost:.2f}$")
            print("-------------------------")
            print(f"Total Optimal Cost: [bold green]{model.ObjVal:.2f}$")
        else:
            print(model.ObjVal)
    except AttributeError:
        print("[red]No solution found")
        return

    plot_result(charging_indices, charging_power, state_of_energy, data)


if __name__ == "__main__":
    main()
