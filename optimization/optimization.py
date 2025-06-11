from functools import reduce
from math import gcd

import gurobipy as gp
from gurobipy import GRB

from optimization.utils import expand_df


class OptimizationModel:
    def __init__(self, data, energy_price, name, granularity="auto"):
        self.data = data
        self.energy_price = energy_price
        self.name = name
        self.granularity = granularity
        self.model = gp.Model(self.name)
        self.charing_indices = None
        self.charging_power = None
        self.state_of_energy = None
        self.dt = None
        self.power_grid_tariff = 0.01
        self.solution = None
        self.vars_initialized = False
        self.constraints_initialized = False
        self.objective_initialized = False

    def set_variables(self):
        all_times = list(self.data["time"]) + list(self.energy_price["time"])
        if self.granularity == "auto":
            self.dt = gcd_of_list(all_times)
        else:
            for t in all_times:
                assert t % self.granularity == 0
            self.dt = self.granularity

        self.data = expand_df(self.data, "time", self.dt)
        self.energy_price = expand_df(self.energy_price, "time", self.dt, no_interpolation=True)

        # charging mask
        self.charging_indices = [i for i in range(len(self.data)) if self.data["depot_charge"][i]]

        # decision variables
        self.charging_power = []
        for i in self.charging_indices:
            self.charging_power.append(
                self.model.addVar(
                    name=f"chargingPower_{i}", vtype=GRB.CONTINUOUS, lb=0, ub=self.data["max_charging_power"][i] / 3600
                )
            )

        self.state_of_energy = []
        for i in range(len(self.data) + 1):
            self.state_of_energy.append(
                self.model.addVar(
                    name=f"stateOfEnergy_{i}",
                    vtype=GRB.CONTINUOUS,
                    lb=self.data["battery_capacity"][0] * 0.2,
                    ub=self.data["battery_capacity"][0] * 0.8,
                )
            )

        # aux variables
        self.max_charging_power = self.model.addVar(name="maxChargingPower", vtype=GRB.CONTINUOUS, lb=0)
        for i, cp in zip(self.charging_indices, self.charging_power):
            self.model.addConstr(self.max_charging_power >= cp, f"maxPower_{i}")  # relaxed max constraint

        self.vars_initialized = True

    def set_constraints(self, ce_function_type="constant", alpha=0.0):
        if not self.vars_initialized:
            raise ValueError("Variables must be initialized before constraints")

        for i, v in enumerate(zip(self.data["depot_charge"], self.data["energy_demand"])):
            depot_charge, energy_demand = v
            if not depot_charge:
                self.model.addConstr(
                    self.state_of_energy[i + 1] == self.state_of_energy[i] - energy_demand,
                    f"energyDemand_{i}",
                )
        for i, cp in zip(self.charging_indices, self.charging_power):
            self.model.addConstr(
                self.state_of_energy[i + 1]
                == self.state_of_energy[i]
                + charging_efficiency(cp, ce_function_type, alpha, self.data["max_charging_power"][i] / 3600)
                * cp
                * self.dt,
                f"charging_{i}",
            )

        self.model.addConstr(self.state_of_energy[0] <= self.state_of_energy[len(self.data)], "energyLoop")

        self.constraints_initialized = True

    def set_objective(self):
        if not self.constraints_initialized:
            raise ValueError("Constraints must be initialized before objective")
        self.model.setObjective(
            gp.quicksum(
                self.energy_price["energy_price"][i] * cp * self.dt
                for i, cp in zip(self.charging_indices, self.charging_power)
            )
            + self.max_charging_power * self.power_grid_tariff,
            GRB.MINIMIZE,
        )

        self.objective_initialized = True

    def optimize(self):
        if not self.objective_initialized:
            raise ValueError("Objective must be initialized before optimization")
        self.model.optimize()
        try:
            self.solution = self.model.ObjVal
        except AttributeError:
            raise ValueError("No solution found")


def charging_efficiency(cp, function_type, alpha, max_charging_power):
    match function_type:
        case "one":
            return 1
        case "constant":
            return alpha
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


def gcd_of_list(values):
    assert len(values) >= 2
    return reduce(gcd, values)
