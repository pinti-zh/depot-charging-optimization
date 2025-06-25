from dataclasses import dataclass
from itertools import product
from typing import Optional, Tuple

import gurobipy as gp
import numpy as np
import polars as pl
from dataclasses_json import dataclass_json
from gurobipy import GRB

from depot_charging_optimization.utils import (
    find_continuos_blocks,
    group_vehicles_by_index,
    list_start_string,
    minimum_joint_chain_range,
)


@dataclass_json
@dataclass
class OptimizationInput:
    num: int
    num_vehicles: int
    dt: int
    soe_lb: float
    soe_ub: float
    grid_tariff: float
    max_charging_power: float
    battery_capacity: np.ndarray[float]
    energy_demand: np.ndarray[float]
    energy_price: np.ndarray[float]
    depot_charge: np.ndarray[bool]

    @classmethod
    def from_dataframes(cls, data: list[pl.DataFrame], energy_price: pl.DataFrame, grid_tariff: float):
        data_columns = ["time", "energy_demand", "depot_charge", "battery_capacity", "max_charging_power"]
        energy_price_columns = ["time", "energy_price"]

        assert len(data) > 0
        for df in data:
            assert set(data_columns) <= set(df.columns)
            assert all(cap == df["battery_capacity"][0] for cap in df["battery_capacity"])
            assert len(df) == len(energy_price) > 0
        assert set(energy_price_columns) <= set(energy_price.columns)

        max_charging_power = max(df.filter(pl.col("depot_charge"))["max_charging_power"].max() for df in data)
        for df in data:
            assert all(
                mcp == max_charging_power or mcp == 0
                for mcp in df.filter(pl.col("depot_charge"))["max_charging_power"]
            )

        num = len(data[0])
        num_vehicles = len(data)
        dt = data[0]["time"][0]
        battery_capacity = np.array([df["battery_capacity"][0] for df in data])
        soe_lb = battery_capacity * 0.2
        soe_ub = battery_capacity * 0.8
        grid_tariff = grid_tariff

        energy_demand = np.array([df["energy_demand"].to_numpy() for df in data])
        depot_charge = np.array([df["depot_charge"].to_numpy() for df in data])
        energy_price = energy_price["energy_price"].to_numpy()
        return cls(
            num,
            num_vehicles,
            dt,
            soe_lb,
            soe_ub,
            grid_tariff,
            max_charging_power,
            battery_capacity,
            energy_demand,
            energy_price,
            depot_charge,
        )

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
            f"      {"\n      ".join(
                [list_start_string(energy_demand_i, 5) for energy_demand_i in self.energy_demand]
                                )}\n"
            f"    depot_charge:\n"
            f"      {"\n      ".join(
                [list_start_string(depot_charge_i, 5) for depot_charge_i in self.depot_charge]
                                )}\n"
            f"    max_charging_power:\n"
            f"      {"\n      ".join(
                [list_start_string(max_charging_power_i, 5) for max_charging_power_i in self.max_charging_power]
                                )}\n"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> dict:
        data_dict = super().__to_dict__()
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                data_dict[key] = value.tolist()
        return data_dict

    def from_dict(self, data_dict) -> dict:
        for key, value in data_dict.items():
            if isinstance(value, list):
                data_dict[key] = np.array(value)
        return super().__from_dict__(data_dict)

    def is_feasible(self) -> Tuple[bool, dict]:
        reasons = {
            "not enough time to charge": [],
            "not enough battery capacity": [],
        }
        for vehicle in range(self.num_vehicles):
            if np.sum(self.max_charging_power * self.depot_charge[vehicle] * self.dt) < np.sum(
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


@dataclass_json
@dataclass
class Solution:
    optimization_input: OptimizationInput
    total_cost: float
    energy_cost: float
    power_cost: float
    gap: float
    max_charging_power_used: float
    charging_power: np.ndarray[float]
    charging_efficiency: np.ndarray[float]
    state_of_energy: np.ndarray[float]

    def __repr__(self):
        return f"{self.total_cost:.3f}$ ({self.energy_cost:.3f}$ + {self.power_cost:.3f}$)"

    def __str__(self):
        return self.__repr__()

    def to_dict(self) -> dict:
        data_dict = super().__to_dict__()
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                data_dict[key] = value.tolist()
        return data_dict

    def from_dict(self, data_dict) -> dict:
        for key, value in data_dict.items():
            if isinstance(value, list):
                data_dict[key] = np.array(value)
        return super().__from_dict__(data_dict)


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

        self.objective_value: Optional[float] = None
        self.vars_initialized: bool = False
        self.constraints_initialized: bool = False
        self.objective_initialized: bool = False

        self.energy_price_scale_factor: float = 1 / max(opt_input.energy_price)
        self.scaled_energy_price: np.ndarray[float] = opt_input.energy_price * self.energy_price_scale_factor
        self.charge_unit: float = opt_input.max_charging_power * opt_input.dt
        self.storable_charge_units: np.ndarray[float] = (opt_input.soe_ub - opt_input.soe_lb) / self.charge_unit
        self.scaled_energy_demands: np.ndarray[float] = opt_input.energy_demand / self.charge_unit

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
                        ub=1,
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
                lb=0,
                ub=self.storable_charge_units[vehicle],
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
            for i, v in enumerate(zip(self.opt_input.depot_charge[vehicle], self.scaled_energy_demands[vehicle])):
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
                    self.model.addConstr(
                        ce == 1 - (1 - alpha) * cp / 2,
                        f"chargingEfficiency_v{vehicle}_{i}",
                    )
                self.model.addConstr(
                    self.state_of_energy[vehicle, i + 1] == self.state_of_energy[vehicle, i] + cp * ce,
                    f"charging_v{vehicle}_{i}",
                )

        # energy loop
        for vehicle in range(self.opt_input.num_vehicles):
            self.model.addConstr(
                self.state_of_energy[vehicle, 0] <= self.state_of_energy[vehicle, self.opt_input.num],
                f"energyLoop_v{vehicle}",
            )

        self.constraints_initialized = True

    def set_objective(self):
        if not self.constraints_initialized:
            raise ValueError("Constraints must be initialized before objective")

        self.model.setObjective(
            gp.quicksum(
                gp.quicksum(
                    self.scaled_energy_price[i] * cp
                    for i, cp in zip(self.charging_indices[vehicle], self.charging_power[vehicle])
                )
                for vehicle in range(self.opt_input.num_vehicles)
            )
            + self.mcp * self.opt_input.grid_tariff * self.energy_price_scale_factor / self.opt_input.dt,
            GRB.MINIMIZE,
        )

        self.objective_initialized = True

    def solve(self) -> Optional[Solution]:
        if not self.objective_initialized:
            raise ValueError("Objective must be initialized before optimization")
        self.model.optimize()
        try:
            self.objective_value = self.model.ObjVal
            obj_bound = self.model.ObjBound
            gap = abs(self.objective_value - obj_bound) / (abs(self.objective_value) + 1e-10) * 100
            charging_power = self.get_charging_power()
            energy_cost = np.sum(charging_power * self.opt_input.energy_price * self.opt_input.dt)
            power_cost = self.get_max_charging_power_used() * self.opt_input.grid_tariff
        except AttributeError:
            return None
        return Solution(
            self.opt_input,
            energy_cost + power_cost,
            energy_cost,
            power_cost,
            gap,
            self.get_max_charging_power_used(),
            self.get_charging_power(),
            self.get_charging_efficiency(),
            self.get_state_of_energy(),
        )

    def get_charging_power(self) -> np.ndarray[np.float64]:
        charging_power = np.zeros((self.opt_input.num_vehicles, self.opt_input.num))
        for vehicle in range(self.opt_input.num_vehicles):
            for i, cp in zip(self.charging_indices[vehicle], self.charging_power[vehicle]):
                charging_power[vehicle, i] = cp.X
        return charging_power * self.charge_unit / self.opt_input.dt

    def get_charging_efficiency(self) -> np.ndarray[np.float64]:
        charging_efficiency = np.zeros((self.opt_input.num_vehicles, self.opt_input.num))
        for vehicle in range(self.opt_input.num_vehicles):
            for i, cp in zip(self.charging_indices[vehicle], self.charging_efficiency[vehicle]):
                charging_efficiency[vehicle, i] = cp.X
        return charging_efficiency

    def get_state_of_energy(self) -> np.ndarray[np.float64]:
        soe = np.empty((self.opt_input.num_vehicles, self.opt_input.num + 1))
        for vehicle in range(self.opt_input.num_vehicles):
            soe[vehicle] = np.array(
                [soe.X * self.charge_unit + self.opt_input.soe_lb[vehicle] for soe in self.state_of_energy[vehicle]]
            )
        return soe

    def get_max_charging_power_used(self) -> float:
        return np.max(np.sum(self.get_charging_power(), axis=0))
