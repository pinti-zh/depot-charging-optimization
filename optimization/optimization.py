from dataclasses import dataclass
from typing import Optional, Tuple

import gurobipy as gp
import numpy as np
import polars as pl
from gurobipy import GRB

from optimization.utils import list_start_string, partial_sums


@dataclass
class OptimizationResult:
    total_cost: float
    energy_cost: float
    power_cost: int = 0

    def __repr__(self):
        return f"{self.total_cost:.3f}$ ({self.energy_cost:.3f}$ + {self.power_cost:.3f}$)"

    def __str__(self):
        return self.__repr__()


class OptimizationInput:
    def __init__(self, data: list[pl.DataFrame], energy_price: pl.DataFrame, grid_tariff: float):
        data_columns = ["time", "energy_demand", "depot_charge", "battery_capacity", "max_charging_power"]
        energy_price_columns = ["time", "energy_price"]

        assert len(data) > 0
        for df in data:
            assert set(data_columns) <= set(df.columns)
            assert all(cap == df["battery_capacity"][0] for cap in df["battery_capacity"])
            assert len(df) == len(energy_price) > 0
        assert set(energy_price_columns) <= set(energy_price.columns)

        self.num: int = len(data[0])
        self.num_vehicles = len(data)
        self.dt: int = data[0]["time"][0]
        self.battery_capacity: np.ndarray[np.float64] = np.array([df["battery_capacity"][0] for df in data])
        self.soe_lb: np.ndarray[np.float64] = self.battery_capacity * 0.2
        self.soe_ub: np.ndarray[np.float64] = self.battery_capacity * 0.8
        self.grid_tariff: float = grid_tariff

        self.energy_demand: np.ndarray[np.float64] = np.array([df["energy_demand"].to_numpy() for df in data])
        self.depot_charge: np.ndarray[np.bool] = np.array([df["depot_charge"].to_numpy() for df in data])
        self.max_charging_power: np.ndarray[np.float64] = np.array(
            [df["max_charging_power"].to_numpy().astype(np.float64) for df in data]
        )
        self.energy_price: np.ndarray[np.float64] = energy_price["energy_price"].to_numpy()

    def __repr__(self) -> str:
        return (
            f"OptimizationInput\n"
            f"    num: {self.num}\n"
            f"    number of vehicles: {self.num_vehicles}\n"
            f"    dt: {self.dt}\n"
            f"    battery_capacity: {list_start_string(self.battery_capacity, 5)}\n"
            f"    grid_tariff: {self.grid_tariff}\n"
            f"    energy_price: {list_start_string(self.energy_price, 3)}\n"
            f"    energy_demand:\n"
            f"      {"\n      ".join([list_start_string(energy_demand_i, 5) for energy_demand_i in self.energy_demand])}\n"
            f"    depot_charge:\n"
            f"      {"\n      ".join([list_start_string(depot_charge_i, 5) for depot_charge_i in self.depot_charge])}\n"
            f"    max_charging_power:\n"
            f"      {"\n      ".join([list_start_string(max_charging_power_i, 5) for max_charging_power_i in self.max_charging_power])}\n"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def is_feasible(self) -> Tuple[bool, str]:
        energy_deltas = []
        for dc, mcp, ed in zip(self.depot_charge, self.max_charging_power, self.energy_demand):
            if dc:
                energy_deltas.append(mcp * self.dt)
            else:
                energy_deltas.append(-ed)

        if sum(energy_deltas) < 0.0:
            return False, "not enough time to charge"

        demand_indices = [i for i, dc in enumerate(self.depot_charge) if not dc]
        for i, start in enumerate(demand_indices):
            for stop in demand_indices[i:]:
                if any(x < -(self.soe_ub - self.soe_lb) for x in partial_sums(energy_deltas[start : stop + 1])):
                    return False, "not enough battery capacity"

        return True, ""

    def naive_greedy_solution(self) -> OptimizationResult:
        current_dc = self.depot_charge[0]
        start = None
        if current_dc:
            switches = 1
        else:
            switches = 0
        for i, dc in enumerate(self.depot_charge):
            if dc != current_dc:
                current_dc = dc
                switches += 1
            if switches >= 2:
                start = i
                break
        assert start is not None

        depot_charge = np.concatenate([self.depot_charge[start:], self.depot_charge[:start]])
        max_charging_power = np.concatenate([self.max_charging_power[start:], self.max_charging_power[:start]])
        energy_demand = np.concatenate([self.energy_demand[start:], self.energy_demand[:start]])
        energy_price = np.concatenate([self.energy_price[start:], self.energy_price[:start]])

        relative_soe = 0.0
        max_depot_charging_power = 0.0
        cumulative_ep = 0.0

        for dc, mcp, ed, ep in zip(depot_charge, max_charging_power, energy_demand, energy_price):
            if dc:
                if relative_soe < 0.0:
                    max_depot_charging_power = max(mcp, max_depot_charging_power)
                    cumulative_ep += min(-relative_soe * ep, mcp * self.dt * ep)
                    relative_soe = min(0.0, relative_soe + mcp * self.dt)
            else:
                relative_soe -= ed

        return OptimizationResult(
            total_cost=cumulative_ep + max_depot_charging_power * self.grid_tariff,
            energy_cost=cumulative_ep,
            power_cost=max_depot_charging_power * self.grid_tariff,
        )


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
                    name=f"stateOfEnergy_{i}", vtype=GRB.CONTINUOUS, lb=self.opt_input.soe_lb, ub=self.opt_input.soe_ub
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

    def solve(self) -> Optional[OptimizationResult]:
        if not self.objective_initialized:
            raise ValueError("Objective must be initialized before optimization")
        self.model.optimize()
        try:
            self.solution = self.model.ObjVal
            power_cost = self.mcp.X * self.opt_input.grid_tariff
            energy_cost = sum(
                self.opt_input.energy_price[i] * cp.X * self.opt_input.dt
                for i, cp in zip(self.charging_indices, self.charging_power)
            )
        except AttributeError:
            return None
        return OptimizationResult(self.solution, energy_cost, power_cost)

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
