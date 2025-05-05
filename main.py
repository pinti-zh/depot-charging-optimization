import argparse

import gurobipy as gp
import polars as pl
from gurobipy import GRB

from utils import plot_result, print_model_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, required=True, help="path to data file")
    parser.add_argument(
        "--verbosity", "-v", choices=[0, 1, 2], type=int, default=0, help="0 = only output, 1 = verbose, 2 = debug"
    )
    parser.add_argument("--max_power", "-mp", type=float, help="maximum possible charging power")
    parser.add_argument("--energy_lb", "-elb", type=float, help="lower bound of SoE")
    parser.add_argument("--energy_ub", "-eub", type=float, help="upper bound of SoE")
    parser.add_argument("--tariff", "-t", type=float, help="power grid tariff")
    args = parser.parse_args()

    # input variables
    DT = 1
    MAX_CHARGING_POWER = args.max_power or 40.0
    STATE_OF_ENERGY_LOWER_BOUND = args.energy_lb or 20.0
    STATE_OF_ENERGY_UPPER_BOUND = args.energy_ub or 80.0
    POWER_GRID_TARIFF = args.tariff or 2.0

    df = pl.read_csv(args.data)
    num_timesteps = len(df)

    # optimization
    model = gp.Model("DepotCharge")

    # decision variables
    power = model.addVars(num_timesteps, vtype=GRB.CONTINUOUS, name="power", lb=-GRB.INFINITY, ub=MAX_CHARGING_POWER)
    state_of_energy = model.addVars(
        num_timesteps + 1,
        vtype=GRB.CONTINUOUS,
        name="state_of_energy",
        lb=STATE_OF_ENERGY_LOWER_BOUND,
        ub=STATE_OF_ENERGY_UPPER_BOUND,
    )

    # aux variables
    max_power = model.addVar(name="max_power")
    for p in power:
        model.addConstr(max_power >= power[p], f"max_power_{p}")

    # constraints
    for i, data in enumerate(df.iter_rows(named=True)):
        if data["Depot"]:
            model.addConstr(power[i] >= 0, f"power_non_negative_{i}")
        else:
            model.addConstr(power[i] == -data["EnergyDemand"] / DT, f"power_{i}_usage")
        model.addConstr(state_of_energy[i + 1] == state_of_energy[i] + power[i] * DT, f"energy_{i}_usage")

    model.addConstr(state_of_energy[0] <= state_of_energy[num_timesteps], "energy_loop")

    # objective
    model.setObjective(
        gp.quicksum(df["EnergyPrice"][i] * df["Depot"][i] * power[i] * DT for i in range(num_timesteps))
        + max_power * POWER_GRID_TARIFF,
        GRB.MINIMIZE,
    )

    # display model
    print_model_summary(model, args.verbosity)

    # solve
    model.optimize()

    # print solution
    try:
        print(f"Optimal Energy Cost: {model.ObjVal:.0f}$")
    except AttributeError:
        print("No solution found =(")
        return

    plot_result(
        power,
        state_of_energy,
        df,
        STATE_OF_ENERGY_LOWER_BOUND,
        STATE_OF_ENERGY_UPPER_BOUND,
    )


if __name__ == "__main__":
    main()
