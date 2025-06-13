from typing import Optional

import gurobipy as gp
import numpy as np
import polars as pl
from gurobipy import GRB

from optimization.utils import list_start_string


class OptimizationInput:
    def __init__(self, data: pl.DataFrame, energy_price: pl.DataFrame, grid_tariff: float):
        data_columns = ["time", "energy_demand", "depot_charge", "battery_capacity", "max_charging_power"]
        energy_price_columns = ["time", "energy_price"]

        assert set(data_columns) <= set(data.columns)
        assert set(energy_price_columns) <= set(energy_price.columns)
        assert len(data) == len(energy_price) > 0
        assert all(cap == data["battery_capacity"][0] for cap in data["battery_capacity"])

        self.num: int = len(data)
        self.dt: int = data["time"][0]
        self.battery_capacity: float = float(data["battery_capacity"][0])
        self.grid_tariff: float = grid_tariff

        self.energy_demand: np.ndarray[np.float64] = data["energy_demand"].to_numpy()
        self.depot_charge: np.ndarray[np.bool] = data["depot_charge"].to_numpy()
        self.max_charging_power: np.ndarray[np.float64] = data["max_charging_power"].to_numpy().astype(np.float64)
        self.energy_price: np.ndarray[np.float64] = energy_price["energy_price"].to_numpy()

    def __repr__(self) -> str:
        return (
            f"OptimizationInput(\n"
            f"    num: {self.num},\n"
            f"    dt: {self.dt},\n"
            f"    battery_capacity: {self.battery_capacity},\n"
            f"    grid_tariff: {self.grid_tariff},\n"
            f"    energy_demand: {list_start_string(self.energy_demand, 5)},\n"
            f"    depot_charge: {list_start_string(self.depot_charge, 5)},\n"
            f"    max_charging_power: {list_start_string(self.max_charging_power, 5)},\n"
            f"    energy_price: {list_start_string(self.energy_price, 5)}\n"
            f")"
        )

    def __str__(self) -> str:
        return self.__repr__()


class OptimizationModel:
    def __init__(self, opt_input: OptimizationInput, name: str = "OptimizationModel"):
        self.opt_input: OptimizationInput = opt_input
        self.name: str = name
        self.model: gp.Model = gp.Model(self.name)

        self.charging_power: list[gp.Var] = []
        self.state_of_energy: list[gp.Var] = []

        self.charging_indices: Optional[np.ndarray[np.int64]] = None
        self.mcp: Optional[gp.Var] = None

        self.solution: Optional[float] = None
        self.vars_initialized: bool = False
        self.constraints_initialized: bool = False
        self.objective_initialized: bool = False

    def set_variables(self):

        # charging mask
        self.charging_indices = [i for i, d in enumerate(self.opt_input.depot_charge) if d]

        # decision variables
        for i in self.charging_indices:
            self.charging_power.append(
                self.model.addVar(
                    name=f"chargingPower_{i}",
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    ub=self.opt_input.max_charging_power[i],
                )
            )

        for i in range(self.opt_input.num + 1):
            self.state_of_energy.append(
                self.model.addVar(
                    name=f"stateOfEnergy_{i}",
                    vtype=GRB.CONTINUOUS,
                    lb=self.opt_input.battery_capacity * 0.2,
                    ub=self.opt_input.battery_capacity * 0.8,
                )
            )

        # aux variables
        self.mcp = self.model.addVar(name="maxChargingPower", vtype=GRB.CONTINUOUS, lb=0)
        for i, cp in zip(self.charging_indices, self.charging_power):
            self.model.addConstr(self.mcp >= cp, f"maxPower_{i}")  # relaxed max constraint

        self.vars_initialized = True

    def set_constraints(self, ce_function_type: str = "one", alpha: float = 0.0):
        if not self.vars_initialized:
            raise ValueError("Variables must be initialized before constraints")

        # energy demand
        for i, v in enumerate(zip(self.opt_input.depot_charge, self.opt_input.energy_demand)):
            depot_charge, energy_demand = v
            if not depot_charge:
                self.model.addConstr(
                    self.state_of_energy[i + 1] == self.state_of_energy[i] - energy_demand,
                    f"energyDemand_{i}",
                )

        # charging
        for i, cp in zip(self.charging_indices, self.charging_power):
            self.model.addConstr(
                self.state_of_energy[i + 1]
                == self.state_of_energy[i]
                + charging_efficiency(cp, self.opt_input.max_charging_power, ce_function_type, alpha)
                * cp
                * self.opt_input.dt,
                f"charging_{i}",
            )

        # energy loop
        self.model.addConstr(self.state_of_energy[0] <= self.state_of_energy[self.opt_input.num], "energyLoop")

        self.constraints_initialized = True

    def set_objective(self):
        if not self.constraints_initialized:
            raise ValueError("Constraints must be initialized before objective")

        self.model.setObjective(
            gp.quicksum(
                self.opt_input.energy_price[i] * cp * self.opt_input.dt
                for i, cp in zip(self.charging_indices, self.charging_power)
            )
            + self.mcp * self.opt_input.grid_tariff,
            GRB.MINIMIZE,
        )

        self.objective_initialized = True

    def solve(self) -> Optional[float]:
        if not self.objective_initialized:
            raise ValueError("Objective must be initialized before optimization")
        self.model.optimize()
        try:
            self.solution = self.model.ObjVal
        except AttributeError:
            self.solution = None
        return self.solution

    def get_charging_power(self) -> np.ndarray[np.float64]:
        charging_power = np.zeros(self.opt_input.num)
        for i, cp in zip(self.charging_indices, self.charging_power):
            charging_power[i] = cp.X
        return charging_power

    def get_state_of_energy(self) -> np.ndarray[np.float64]:
        return np.array([soe.X for soe in self.state_of_energy])


def charging_efficiency(cp: float, mcp: float, function_type: str, alpha: float) -> float:
    match function_type:
        case "one":
            return 1.0
        case "constant":
            return alpha
        case "quadratic":
            return 1 - alpha * (cp / mcp) ** 2
        case _:
            raise ValueError(f"Unknown charging efficiency function type: {function_type}")
