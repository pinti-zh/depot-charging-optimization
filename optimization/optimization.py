import gurobipy as gp
from gurobipy import GRB


class OptimizationModel:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.model = gp.Model(self.name)
        self.charing_indices = None
        self.charging_power = None
        self.state_of_energy = None
        self.max_charging_power = None
        self.solution = None
        self.vars_initialized = False
        self.constraints_initialized = False
        self.objective_initialized = False

    def set_variables(self):
        # charging mask
        self.charging_indices = get_charging_indices(self.data["numTimeSteps"], self.data["energyDemand"])

        # decision variables
        self.charging_power = []
        for i in self.charging_indices:
            self.charging_power.append(
                self.model.addVar(
                    name=f"chargingPower_{i}", vtype=GRB.CONTINUOUS, lb=0, ub=self.data["maxChargingPower"]
                )
            )

        self.state_of_energy = []
        for i in range(self.data["numTimeSteps"] + 1):
            self.state_of_energy.append(
                self.model.addVar(
                    name=f"stateOfEnergy_{i}",
                    vtype=GRB.CONTINUOUS,
                    lb=self.data["stateOfEnergyLowerBound"],
                    ub=self.data["stateOfEnergyUpperBound"],
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

        for i, energy_demand in enumerate(self.data["energyDemand"]):
            self.model.addConstr(
                self.state_of_energy[energy_demand["end"]]
                == self.state_of_energy[energy_demand["start"]] - energy_demand["value"],
                f"energyDemand_{i}",
            )
        for i, cp in zip(self.charging_indices, self.charging_power):
            self.model.addConstr(
                self.state_of_energy[i + 1]
                == self.state_of_energy[i]
                + charging_efficiency(cp, ce_function_type, alpha, self.data["maxChargingPower"])
                * cp
                * self.data["timeStepDuration"],
                f"charging_{i}",
            )

        self.model.addConstr(self.state_of_energy[0] <= self.state_of_energy[self.data["numTimeSteps"]], "energyLoop")

        self.constraints_initialized = True

    def set_objective(self):
        if not self.constraints_initialized:
            raise ValueError("Constraints must be initialized before objective")
        self.model.setObjective(
            gp.quicksum(
                self.data["energyPrice"][i] * cp * self.data["timeStepDuration"]
                for i, cp in zip(self.charging_indices, self.charging_power)
            )
            + self.max_charging_power * self.data["powerGridTariff"],
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
