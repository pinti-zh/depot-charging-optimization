from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator, model_validator


class Input(BaseModel):
    num_timesteps: int = -1
    num_vehicles: int
    time: list[int]
    energy_demand: list[list[float]]
    soe_lb: list[float]
    soe_ub: list[float]
    max_charging_power: float
    battery_capacity: list[float]
    depot_charge: list[list[bool]]
    energy_price: Optional[list[float]] = None
    grid_tariff: Optional[float] = None

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
        if n <= 0:
            raise ValueError("Field[time] must not be empty")
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

    @model_validator(mode="after")
    def set_num_timesteps(self):
        self.num_timesteps = len(self.time)
        return self

    @classmethod
    def from_dataframes(cls, df: pd.DataFrame):
        required_dataframe_columns = [
            "time",
            "energy_demand",
            "depot_charge",
            "battery_capacity",
            "max_charging_power",
        ]

        # assert all dataframes are non-empty
        if len(df) <= 0:
            raise ValueError("Dataframe is empty")

        # assert that all dataframes have the required columns
        for col in required_dataframe_columns:
            if col not in df.columns:
                raise ValueError(f"Column [{col}] not found in dataframe")

        # assert consistent scalar values
        if not all(cap == df["battery_capacity"][0] for cap in df["battery_capacity"]):
            raise ValueError("Battery capacity columns do not match")

        max_charging_power = df[df["depot_charge"]]["max_charging_power"].max()
        if not all(mcp == max_charging_power or mcp == 0 for mcp in df[df["depot_charge"]]["max_charging_power"]):
            raise ValueError("Max charging power columns do not match")

        # create OptimizationInput
        return cls(
            num_vehicles=1,
            time=df["time"].to_list(),
            energy_demand=[df["energy_demand"].to_list()],
            soe_lb=[0.2],
            soe_ub=[0.8],
            max_charging_power=max_charging_power,
            battery_capacity=[df["battery_capacity"][0]],
            depot_charge=[df["depot_charge"].to_list()],
        )


class Solution(BaseModel):
    optimization_input: Input
    total_cost: float
    energy_cost: float
    power_cost: float
    gap: float
    max_charging_power_used: float
    charging_power: list[list[float]]
    charging_efficiency: list[list[float]]
    state_of_energy: list[list[float]]

    @field_validator("gap")
    @classmethod
    def check_between_0_and_1(cls, value, info):
        if not 0 <= value <= 1:
            raise ValueError(f"Field[{info.field_name}] must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def check_list_lengths(self):
        num_vehicles = self.optimization_input.num_vehicles
        num_timesteps = self.optimization_input.num_timesteps

        # assert all list contain [num_vehicles] lists
        if not len(self.charging_power) == num_vehicles:
            raise ValueError(f"Field[charging_power] has length {len(self.charging_power)}, expected {num_vehicles}")
        if not len(self.charging_efficiency) == num_vehicles:
            raise ValueError(
                f"Field[charging_efficiency] has length {len(self.charging_efficiency)}, expected {num_vehicles}"
            )
        if not len(self.state_of_energy) == num_vehicles:
            raise ValueError(f"Field[state_of_energy] has length {len(self.state_of_energy)}, expected {num_vehicles}")

        # assert all lists contain lists with [num_timesteps] values
        for values in self.charging_power:
            if not len(values) == num_timesteps:
                raise ValueError(
                    f"Field[charging_power] contains list of length {len(values)}, expected {num_timesteps}"
                )
        for values in self.charging_efficiency:
            if not len(values) == num_timesteps:
                raise ValueError(
                    f"Field[charging_efficiency] contains list of length {len(values)}, expected {num_timesteps}"
                )
        for values in self.state_of_energy:
            if not len(values) == num_timesteps + 1:
                raise ValueError(
                    f"Field[state_of_energy] contains list of length {len(values)}, expected {num_timesteps + 1}"
                )

        return self
