import argparse
import json

import gurobipy as gp
from gurobipy import GRB
from rich import print

from utils import plot_result, print_model_summary


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
        "--verbosity", "-v", choices=[0, 1, 2], type=int, default=0, help="0 = only output, 1 = verbose, 2 = debug"
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
        model.addConstr(max_charging_power >= cp, f"max_power_{i}")  # relaxed max constraint

    # constraints
    for i, energy_demand in enumerate(data["energyDemand"]):
        model.addConstr(
            state_of_energy[energy_demand["end"]] == state_of_energy[energy_demand["start"]] - energy_demand["value"],
            f"energyDemand_{i}",
        )
    for i, cp in zip(charging_indices, charging_power):
        model.addConstr(state_of_energy[i + 1] == state_of_energy[i] + cp * data["timeStepDuration"], f"charging_{i}")

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

    # solve
    model.optimize()

    # print solution
    try:
        print(f"Optimal Energy Cost: [bold green]{model.ObjVal:.0f}$")
    except AttributeError:
        print("[red]No solution found")
        return

    plot_result(charging_indices, charging_power, state_of_energy, data)


if __name__ == "__main__":
    main()
