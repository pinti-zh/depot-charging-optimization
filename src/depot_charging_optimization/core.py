from dataclasses import asdict, dataclass
from itertools import product
from typing import Optional

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from pydantic import BaseModel, field_validator, model_validator

from depot_charging_optimization.utils import (
    group_vehicles_by_index,
    numpy_to_py,
    py_to_numpy,
)


class OptimizationInput(BaseModel):
    num_vehicles: int
    time: list[int]
    energy_price: list[float]
    grid_tariff: float
    energy_demand: list[list[float]]
    soe_lb: list[float]
    soe_ub: list[float]
    max_charging_power: float
    battery_capacity: list[float]
    depot_charge: list[list[bool]]

    @field_validator("num_vehicles", "max_charging_power")
    @classmethod
    def check_positive(cls, value, info):
        if value <= 0:
            raise ValueError(f"Field[{info.field_name}] must be positive")
        return value

    @field_validator("time", "battery_capacity")
    @classmethod
    def check_positive_list(cls, value, info):
        if not all(v > 0 for v in value):
            raise ValueError(f"Field[{info.field_name}] must be positive")
        return value

    @field_validator("time")
    @classmethod
    def check_ascending(cls, value, info):
        diff = [v2 - v1 for v1, v2 in zip(value[:-1], value[1:])]
        if not all(d > 0 for d in diff):
            raise ValueError(f"Field[{info.field_name}] must be strictly ascending")
        return value

    @field_validator("soe_lb", "soe_ub")
    @classmethod
    def check_between_0_and_1(cls, value, info):
        if not all(0 <= v <= 1 for v in value):
            raise ValueError(f"Field[{info.field_name}] must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def check_soe_bounds(self):
        if not all(lb < ub for lb, ub in zip(self.soe_lb, self.soe_ub)):
            raise ValueError("Field[soe_lb] must be smaller than Field[soe_ub]")
        return self

    @model_validator(mode="after")
    def check_list_lengths(self):
        n = len(self.time)
        if not len(self.energy_price) == n:
            raise ValueError(f"Field[energy_price] has length {len(self.energy_price)}, expected {n}")
        if not all(len(v) == n for v in self.energy_demand):
            raise ValueError(f"Entry of Field[energy_demand] does not have expected length {n}")
        if not all(len(v) == n for v in self.depot_charge):
            raise ValueError(f"Entry of Field[depot_charge] does not have expected length {n}")
        if not len(self.energy_demand) == self.num_vehicles:
            raise ValueError(
                f"Field[energy_demand] has length {len(self.energy_demand)}, expected {self.num_vehicles}"
            )
        if not len(self.depot_charge) == self.num_vehicles:
            raise ValueError(f"Field[depot_charge] has length {len(self.depot_charge)}, expected {self.num_vehicles}")
        if not len(self.battery_capacity) == self.num_vehicles:
            raise ValueError(
                f"Field[battery_capacity] has length {len(self.battery_capacity)}, expected {self.num_vehicles}"
            )
        if not len(self.soe_lb) == self.num_vehicles:
            raise ValueError(f"Field[soe_lb] has length {len(self.soe_lb)}, expected {self.num_vehicles}")
        if not len(self.soe_ub) == self.num_vehicles:
            raise ValueError(f"Field[soe_ub] has length {len(self.soe_ub)}, expected {self.num_vehicles}")
        return self

    @classmethod
    def from_dataframes(cls, data: list[pd.DataFrame], energy_price: pd.DataFrame, grid_tariff: float):
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
        battery_capacity = [df["battery_capacity"][0] for df in data]
        soe_lb = [bc * 0.2 for bc in battery_capacity]
        soe_ub = [bc * 0.8 for bc in battery_capacity]
        grid_tariff = grid_tariff

        energy_demand = [df["energy_demand"].to_list() for df in data]
        depot_charge = [df["depot_charge"].to_list() for df in data]
        energy_price = energy_price["energy_price"].to_list()
        return cls(
            np.int64(num),
            np.int64(num_vehicles),
            np.int64(dt),
            np.array(soe_lb, dtype=np.int64),
            np.array(soe_ub, dtype=np.int64),
            np.float32(grid_tariff),
            np.float32(max_charging_power),
            np.array(battery_capacity, dtype=np.float32),
            np.array(energy_demand, dtype=np.float32),
            np.array(energy_price, dtype=np.float32),
            np.array(depot_charge, dtype=np.bool_),
        )

    def to_dict(self) -> dict:
        data_dict = asdict(self)
        for key, value in data_dict.items():
            data_dict[key] = numpy_to_py(value)
        return data_dict

    @classmethod
    def from_dict(cls, data_dict: dict) -> dict:
        for key, value in data_dict.items():
            data_dict[key] = py_to_numpy(value)
        return cls(**data_dict)


@dataclass
class Solution:
    optimization_input: OptimizationInput
    total_cost: np.float32
    energy_cost: np.float32
    power_cost: np.float32
    gap: np.float32
    max_charging_power_used: np.float32
    charging_power: np.ndarray[np.float32]
    charging_efficiency: np.ndarray[np.float32]
    state_of_energy: np.ndarray[np.float32]

    def to_dict(self) -> dict:
        data_dict = asdict(self)
        for key, value in data_dict.items():
            data_dict[key] = numpy_to_py(value)
        data_dict["optimization_input"] = self.optimization_input.to_dict()
        return data_dict

    @classmethod
    def from_dict(cls, data_dict: dict) -> dict:
        for key, value in data_dict.items():
            data_dict[key] = py_to_numpy(value)
        data_dict["optimization_input"] = OptimizationInput.from_dict(data_dict["optimization_input"])
        return cls(**data_dict)


class OptimizationModel:
    def __init__(self, opt_input: OptimizationInput, name: str = "OptimizationModel", greedy: bool = False):
        self.opt_input: OptimizationInput = opt_input
        self.name: str = name
        self.model: gp.Model = gp.Model(self.name)
        self.greedy = greedy

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

        if self.greedy:
            self.model.setObjective(
                gp.quicksum(
                    gp.quicksum(self.state_of_energy[vehicle]) for vehicle in range(self.opt_input.num_vehicles)
                ),
                GRB.MAXIMIZE,
            )
        else:
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
            np.float32(energy_cost + power_cost),
            np.float32(energy_cost),
            np.float32(power_cost),
            np.float32(gap),
            np.float32(self.get_max_charging_power_used()),
            self.get_charging_power(),
            self.get_charging_efficiency(),
            self.get_state_of_energy(),
        )

    def get_charging_power(self) -> np.ndarray[np.float64]:
        charging_power = np.zeros((self.opt_input.num_vehicles, self.opt_input.num), dtype=np.float32)
        for vehicle in range(self.opt_input.num_vehicles):
            for i, cp in zip(self.charging_indices[vehicle], self.charging_power[vehicle]):
                charging_power[vehicle, i] = cp.X
        return charging_power * self.charge_unit / self.opt_input.dt

    def get_charging_efficiency(self) -> np.ndarray[np.float64]:
        charging_efficiency = np.zeros((self.opt_input.num_vehicles, self.opt_input.num), dtype=np.float32)
        for vehicle in range(self.opt_input.num_vehicles):
            for i, cp in zip(self.charging_indices[vehicle], self.charging_efficiency[vehicle]):
                charging_efficiency[vehicle, i] = cp.X
        return charging_efficiency

    def get_state_of_energy(self) -> np.ndarray[np.float64]:
        soe = np.empty((self.opt_input.num_vehicles, self.opt_input.num + 1), dtype=np.float32)
        for vehicle in range(self.opt_input.num_vehicles):
            soe[vehicle] = np.array(
                [soe.X * self.charge_unit + self.opt_input.soe_lb[vehicle] for soe in self.state_of_energy[vehicle]]
            )
        return soe

    def get_max_charging_power_used(self) -> float:
        return np.max(np.sum(self.get_charging_power(), axis=0))
