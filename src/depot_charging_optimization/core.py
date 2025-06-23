from dataclasses import dataclass
from itertools import product
from typing import Optional, Tuple

import gurobipy as gp
import numpy as np
import polars as pl
from gurobipy import GRB

from depot_charging_optimization.utils import (
    find_continuos_blocks,
    group_vehicles_by_index,
    list_start_string,
    minimum_joint_chain_range,
    partial_sums,
)


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

    def is_feasible(self) -> Tuple[bool, dict]:
        reasons = {
            "not enough time to charge": [],
            "not enough battery capacity": [],
        }
        for vehicle in range(self.num_vehicles):
            if np.sum(self.max_charging_power[vehicle] * self.depot_charge[vehicle] * self.dt) < np.sum(
                self.energy_demand[vehicle]
            ):
                reasons["not enough time to charge"].append(vehicle)

            continuous_blocks = find_continuos_blocks(self.depot_charge[vehicle])
            chain_values = []
            for i, j, value in continuous_blocks:
                if value:
                    if i < j:
                        chain_values.append(np.sum(self.max_charging_power[vehicle][i:j]) * self.dt)
                    else:
                        chain_values.append(
                            np.sum(
                                np.concatenate(
                                    [self.max_charging_power[vehicle][i:], self.max_charging_power[vehicle][:j]]
                                )
                            )
                            * self.dt
                        )
                else:
                    if i < j:
                        cum_sum = np.cumsum(self.energy_demand[vehicle][i:j])
                    else:
                        cum_sum = np.cumsum(
                            np.concatenate(self.energy_demand[vehicle][i:], self.energy_demand[vehicle][:j])
                        )
                    chain_values.append(np.max(cum_sum) - min(np.min(cum_sum), 0))
            if continuous_blocks[0][2]:
                chain_blocks = chain_values[1::2]
                joints = chain_values[2::2]
            else:
                chain_blocks = chain_values[::2]
                joints = chain_values[1:-1:2]

            min_capacity_needed = minimum_joint_chain_range(chain_blocks, joints)
            if min_capacity_needed > self.soe_ub[vehicle] - self.soe_lb[vehicle]:
                reasons["not enough battery capacity"].append(vehicle)

        if any(map(lambda x: len(x) > 0, reasons.values())):
            return False, reasons
        return True, reasons

    def is_feasible_slow(self) -> Tuple[bool, dict]:
        reasons = {
            "not enough time to charge": [],
            "not enough battery capacity": [],
        }
        for vehicle in range(self.num_vehicles):
            energy_deltas = []
            for dc, mcp, ed in zip(
                self.depot_charge[vehicle], self.max_charging_power[vehicle], self.energy_demand[vehicle]
            ):
                if dc:
                    energy_deltas.append(mcp * self.dt)
                else:
                    energy_deltas.append(-ed)

            if sum(energy_deltas) < 0.0:
                reasons["not enough time to charge"].append(vehicle)
                break

            demand_indices = [i for i, dc in enumerate(self.depot_charge[vehicle]) if not dc]
            for i, start in enumerate(demand_indices):
                found_reason = False
                for stop in demand_indices[i:]:
                    if any(
                        x < -(self.soe_ub[vehicle] - self.soe_lb[vehicle])
                        for x in partial_sums(energy_deltas[start: stop + 1])
                    ):
                        reasons["not enough battery capacity"].append(vehicle)
                        found_reason = True
                        break
                if found_reason:
                    break

        if any(map(lambda x: len(x) > 0, reasons.values())):
            return False, reasons

        return True, reasons


class OptimizationModel:
    def __init__(self, opt_input: OptimizationInput, name: str = "OptimizationModel"):
        self.opt_input: OptimizationInput = opt_input
        self.name: str = name
        self.model: gp.Model = gp.Model(self.name)

        self.charging_power: list[np.ndarray[gp.Var]] = []
        self.state_of_energy: np.ndarray[gp.Var] = np.empty((opt_input.num_vehicles, opt_input.num + 1), dtype=gp.Var)
        self.charging_efficiency: list[np.ndarray[gp.Var]] = []

        self.charging_indices: Optional[list[np.ndarray[np.int64]]] = None
        self.mcp: Optional[gp.Var] = None

        self.solution: Optional[OptimizationResult] = None
        self.vars_initialized: bool = False
        self.constraints_initialized: bool = False
        self.objective_initialized: bool = False

    def set_variables(self):

        # charging mask
        self.charging_indices = [np.where(depot_charge_i)[0] for depot_charge_i in self.opt_input.depot_charge]

        # decision variables
        for vehicle, vehicle_charging_indices in enumerate(self.charging_indices):
            vehicle_charging_power = np.array(
                [
                    self.model.addVar(
                        name=f"chargingPower_v{vehicle}_{i}",
                        vtype=GRB.CONTINUOUS,
                        lb=0,
                        ub=self.opt_input.max_charging_power[vehicle, i],
                    )
                    for i in vehicle_charging_indices
                ]
            )
            vehicle_charging_efficiency = np.array(
                [
                    self.model.addVar(
                        name=f"chargingEfficiency_v{vehicle}_{i}",
                        vtype=GRB.CONTINUOUS,
                        lb=0,
                        ub=1,
                    )
                    for i in vehicle_charging_indices
                ]
            )
            self.charging_power.append(vehicle_charging_power)
            self.charging_efficiency.append(vehicle_charging_efficiency)

        for vehicle, i in product(range(self.opt_input.num_vehicles), range(self.opt_input.num + 1)):
            self.state_of_energy[vehicle, i] = self.model.addVar(
                name=f"stateOfEnergy_v{vehicle}_{i}",
                vtype=GRB.CONTINUOUS,
                lb=self.opt_input.soe_lb[vehicle],
                ub=self.opt_input.soe_ub[vehicle],
            )

        # aux variables
        self.mcp = self.model.addVar(name="maxChargingPower", vtype=GRB.CONTINUOUS, lb=0)
        for index, vehicles in group_vehicles_by_index(self.charging_indices).items():
            indices = [np.where(self.charging_indices[vehicle] == index)[0][0] for vehicle in vehicles]
            self.model.addConstr(
                self.mcp >= gp.quicksum(self.charging_power[vehicle][i] for i, vehicle in zip(indices, vehicles)),
                f"maxChargingPower_{index}",
            )

        self.vars_initialized = True

    def set_constraints(self, ce_function_type: str = "one", alpha: float = 1.0):
        if not self.vars_initialized:
            raise ValueError("Variables must be initialized before constraints")

        # energy demand
        for vehicle in range(self.opt_input.num_vehicles):
            for i, v in enumerate(zip(self.opt_input.depot_charge[vehicle], self.opt_input.energy_demand[vehicle])):
                depot_charge, energy_demand = v
                if not depot_charge:
                    self.model.addConstr(
                        self.state_of_energy[vehicle, i + 1] == self.state_of_energy[vehicle, i] - energy_demand,
                        f"energyDemand_v{vehicle}_{i}",
                    )

        # charging
        for vehicle in range(self.opt_input.num_vehicles):
            for i, cp, ce in zip(
                self.charging_indices[vehicle], self.charging_power[vehicle], self.charging_efficiency[vehicle]
            ):
                if ce_function_type == "one":
                    self.model.addConstr(ce == 1.0, f"chargingEfficiency_v{vehicle}_{i}")
                elif ce_function_type == "constant":
                    self.model.addConstr(ce == alpha, f"chargingEfficiency_v{vehicle}_{i}")
                elif ce_function_type == "quadratic":
                    q_constant = (1 - alpha) / (3 * self.opt_input.max_charging_power[vehicle, i] ** 2)
                    self.model.addQConstr(ce <= 1 - q_constant * cp * cp, f"chargingEfficiency_v{vehicle}_{i}")
                self.model.addConstr(
                    self.state_of_energy[vehicle, i + 1]
                    == self.state_of_energy[vehicle, i] + cp * self.opt_input.dt * ce,
                    f"charging_v{vehicle}_{i}",
                )

        # energy loop
        for vehicle in range(self.opt_input.num_vehicles):
            self.model.addConstr(
                self.state_of_energy[vehicle, 0] <= self.state_of_energy[vehicle, self.opt_input.num], "energyLoop"
            )

        self.constraints_initialized = True

    def set_objective(self):
        if not self.constraints_initialized:
            raise ValueError("Constraints must be initialized before objective")

        self.model.setObjective(
            gp.quicksum(
                gp.quicksum(
                    self.opt_input.energy_price[i] * cp * self.opt_input.dt
                    for i, cp in zip(self.charging_indices[vehicle], self.charging_power[vehicle])
                )
                for vehicle in range(self.opt_input.num_vehicles)
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
                sum(
                    self.opt_input.energy_price[i] * cp.X * self.opt_input.dt
                    for i, cp in zip(self.charging_indices[vehicle], self.charging_power[vehicle])
                )
                for vehicle in range(self.opt_input.num_vehicles)
            )
        except AttributeError:
            return None
        return OptimizationResult(self.solution, energy_cost, power_cost)

    def get_charging_power(self) -> np.ndarray[np.float64]:
        charging_power = np.zeros((self.opt_input.num_vehicles, self.opt_input.num))
        for vehicle in range(self.opt_input.num_vehicles):
            for i, cp in zip(self.charging_indices[vehicle], self.charging_power[vehicle]):
                charging_power[vehicle, i] = cp.X
        return charging_power

    def get_charging_efficiency(self) -> np.ndarray[np.float64]:
        charging_efficiency = np.zeros((self.opt_input.num_vehicles, self.opt_input.num))
        for vehicle in range(self.opt_input.num_vehicles):
            for i, cp in zip(self.charging_indices[vehicle], self.charging_efficiency[vehicle]):
                charging_efficiency[vehicle, i] = cp.X
        return charging_efficiency

    def get_state_of_energy(self) -> np.ndarray[np.float64]:
        soe = np.empty((self.opt_input.num_vehicles, self.opt_input.num + 1))
        for vehicle in range(self.opt_input.num_vehicles):
            soe[vehicle] = np.array([soe.X for soe in self.state_of_energy[vehicle]])
        return soe

    def get_max_charging_power_used(self) -> float:
        return self.mcp.X


class GreedyOptimizationModel(OptimizationModel):
    def __init__(self, opt_input: OptimizationInput, name: str = "GreedyOptimizationModel"):
        super().__init__(opt_input, name)

    def set_constraints(
        self, ce_function_type: str = "one", alpha: float = 1.0, adjusted_max_power: Optional[float] = None
    ):
        super().set_constraints(ce_function_type=ce_function_type, alpha=alpha)
        if adjusted_max_power is not None:
            self.model.addConstr(self.mcp <= adjusted_max_power, "adjustedMaxChargingPower")

    def set_objective(self):
        if not self.constraints_initialized:
            raise ValueError("Constraints must be initialized before objective")

        self.model.setObjective(
            gp.quicksum(
                gp.quicksum(soe for soe in self.state_of_energy[vehicle])
                for vehicle in range(self.opt_input.num_vehicles)
            ),
            GRB.MAXIMIZE,
        )

        self.objective_initialized = True

    def solve(self):
        if not self.objective_initialized:
            raise ValueError("Objective must be initialized before optimization")
        self.model.optimize()
        try:
            self.solution = self.model.ObjVal
            energy_cost = 0.0
            for vehicle in range(self.opt_input.num_vehicles):
                for i, cp in zip(self.charging_indices[vehicle], self.charging_power[vehicle]):
                    energy_cost += self.opt_input.energy_price[i] * cp.X * self.opt_input.dt
            power_cost = np.max(np.sum(self.get_charging_power(), axis=0)) * self.opt_input.grid_tariff
        except AttributeError:
            return None
        return OptimizationResult(energy_cost + power_cost, energy_cost, power_cost)
