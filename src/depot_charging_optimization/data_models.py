from functools import reduce
from math import gcd

from pydantic import BaseModel, field_validator, model_validator


class Input(BaseModel):
    num_timesteps: int = -1
    num_vehicles: int
    time: list[int]
    energy_demand: list[list[float]]
    max_charging_power: float
    battery_capacity: list[float]
    depot_charge: list[list[bool]]
    energy_price: list[float] | None = None
    grid_tariff: float | None = None
    is_battery: list[bool] | None = None

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
        if not len(self.is_battery) == self.num_vehicles:
            raise ValueError(f"Field[is_battery] has length {len(self.is_battery)}, expected {self.num_vehicles}")
        return self

    @model_validator(mode="after")
    def check_no_energy_demand_in_depot(self):
        for vehicle_depot_charge, vehicle_energy_demand in zip(self.depot_charge, self.energy_demand):
            energy_demand_in_depot = any(
                dc and (demand > 0) for dc, demand in zip(vehicle_depot_charge, vehicle_energy_demand)
            )
            if energy_demand_in_depot:
                raise ValueError("Nonzero energy demand found while depot charging")
        return self

    @model_validator(mode="after")
    def set_num_timesteps(self):
        self.num_timesteps = len(self.time)
        return self

    @model_validator(mode="after")
    def set_is_battery(self):
        if self.is_battery is None:
            self.is_battery = [False] * self.num_vehicles
        return self

    def rotate(self) -> "Input":
        if self.energy_price is None:
            rotated_energy_price = None
        else:
            rotated_energy_price = self.energy_price[1:] + [self.energy_price[0]]  # type: ignore
        return Input(
            num_timesteps=self.num_timesteps,
            num_vehicles=self.num_vehicles,
            time=[t - self.time[0] for t in self.time[1:]] + [self.time[-1]],
            energy_demand=[item[1:] + [item[0]] for item in self.energy_demand],
            max_charging_power=self.max_charging_power,
            battery_capacity=self.battery_capacity,
            depot_charge=[item[1:] + [item[0]] for item in self.depot_charge],
            energy_price=rotated_energy_price,
            grid_tariff=self.grid_tariff,
            is_battery=self.is_battery,
        )

    def truncate(self, n: int) -> "Input":
        return Input(
            num_timesteps=min(self.num_timesteps, n),
            num_vehicles=self.num_vehicles,
            time=self.time[:n],
            energy_demand=[energy_demand[:n] for energy_demand in self.energy_demand],
            max_charging_power=self.max_charging_power,
            battery_capacity=self.battery_capacity,
            depot_charge=[depot_charge[:n] for depot_charge in self.depot_charge],
            energy_price=self.energy_price[:n] if self.energy_price is not None else None,  # type: ignore
            grid_tariff=self.grid_tariff,
            is_battery=self.is_battery,
        )

    def loop(self, loops: int) -> "Input":
        if self.energy_price is None:
            looped_energy_price = None
        else:
            looped_energy_price = self.energy_price * loops

        looped_time = []
        max_time = max(self.time)
        for i in range(loops):
            looped_time += [t + (i * max_time) for t in self.time]

        return Input(
            num_timesteps=self.num_timesteps * loops,
            num_vehicles=self.num_vehicles,
            time=looped_time,
            energy_demand=[energy_demand * loops for energy_demand in self.energy_demand],
            max_charging_power=self.max_charging_power,
            battery_capacity=self.battery_capacity,
            depot_charge=[depot_charge * loops for depot_charge in self.depot_charge],
            energy_price=looped_energy_price,
            grid_tariff=self.grid_tariff,
            is_battery=self.is_battery,
        )

    def maximum_possible_equal_timestep(self) -> int:
        delta_time = []
        for t1, t2 in zip([0] + self.time[:-1], self.time):
            delta_time.append(t2 - t1)
        return int(reduce(gcd, delta_time))

    def equalize_timesteps(self, dt: int | None = None) -> "Input":
        max_dt = self.maximum_possible_equal_timestep()
        dt = dt or max_dt

        if gcd(max_dt, dt) != dt:
            raise ValueError(f"Incompatible dt for equalizing timestep: dt = {dt}, must be divisor of {max_dt}")
        equalized_timesteps = [dt * (i + 1) for i in range(self.time[-1] // dt)]

        return self._extend(equalized_timesteps)

    @classmethod
    def combine(cls, inputs: list["Input"]) -> "Input":
        if not len(inputs) > 0:
            raise ValueError("Inputs must not be empty")

        if not all(item.time[-1] == inputs[0].time[-1] for item in inputs):
            raise ValueError("Inputs do not cover same time period")

        if not all(item.max_charging_power == inputs[0].max_charging_power for item in inputs):
            raise ValueError("Inputs do not have same max_charging_power")

        num_vehicles = sum(item.num_vehicles for item in inputs)

        battery_capacity = []
        for item in inputs:
            battery_capacity += item.battery_capacity

        time = []
        for item in inputs:
            time += item.time
        time = sorted(list(set(time)))

        extended_inputs = [item._extend(time) for item in inputs]

        energy_demand = []
        depot_charge = []
        for item in extended_inputs:
            energy_demand += item.energy_demand
            depot_charge += item.depot_charge

        is_battery: list[bool] = []
        for item in inputs:
            assert item.is_battery is not None
            is_battery += item.is_battery

        return Input(
            num_vehicles=num_vehicles,
            time=time,
            energy_demand=energy_demand,
            max_charging_power=inputs[0].max_charging_power,
            battery_capacity=battery_capacity,
            depot_charge=depot_charge,
            is_battery=is_battery,
        )

    def add_grid_tariff(self, grid_tariff: float) -> "Input":
        self.grid_tariff = grid_tariff
        return self

    def add_energy_price(self, energy_time: list[int], energy_price: list[float]) -> "Input":
        # check that energy price is not empty
        if not len(energy_price) > 0:
            raise ValueError("Energy time must not be empty")
        # check that energy time and price have equal length
        if not len(energy_price) == len(energy_time):
            raise ValueError("Energy time and price must have equal length")
        # check that energy time is ascending
        if not all(energy_time[i] <= energy_time[i + 1] for i in range(len(energy_time) - 1)):
            raise ValueError("Cannot add energy price, time is not ascending")
        # check that time is positive
        if not all(t_i > 0 for t_i in energy_time):
            raise ValueError("Cannot add energy price, time is not positive")

        time = sorted(list(set(self.time + energy_time)))
        current_energy_price_index = 0
        extended_energy_price = []
        for t_i in time:
            extended_energy_price.append(energy_price[current_energy_price_index])
            if t_i == energy_time[current_energy_price_index]:
                current_energy_price_index += 1

        extended_input = self._extend(time)
        extended_input.energy_price = extended_energy_price
        return extended_input

    def _extend(self, extended_time: list[int]) -> "Input":
        # check that extended_time is a superset
        if not all(t_i in extended_time for t_i in self.time):
            raise ValueError("Cannot extend Input to new time period, Input.time is not a subset")
        # check that extended_time is ascending
        if not all(extended_time[i] <= extended_time[i + 1] for i in range(len(extended_time) - 1)):
            raise ValueError("Cannot extend Input to new time period, Input.time is not ascending")
        # check positivity
        if not all(t_i > 0 for t_i in extended_time):
            raise ValueError("Cannot extend Input to new time period, Input.time is not positive")

        energy_demand: list[list[float]] = [[] for _ in range(self.num_vehicles)]
        depot_charge: list[list[bool]] = [[] for _ in range(self.num_vehicles)]
        for t1, t2 in zip([0] + extended_time[:-1], extended_time):
            index = self._index_of_time_interval(t1, t2)
            if index == 0:
                dt = self.time[index]
            else:
                dt = self.time[index] - self.time[index - 1]
            for vehicle in range(self.num_vehicles):
                ed = self.energy_demand[vehicle][index] * ((t2 - t1) / dt)
                dc = self.depot_charge[vehicle][index]
                energy_demand[vehicle].append(ed)
                depot_charge[vehicle].append(dc)

        return Input(
            num_vehicles=self.num_vehicles,
            time=extended_time,
            energy_demand=energy_demand,
            max_charging_power=self.max_charging_power,
            battery_capacity=self.battery_capacity,
            depot_charge=depot_charge,
            is_battery=self.is_battery,
        )

    def _index_of_time_interval(self, start: int, end: int) -> int:
        if start >= end or end <= 0:
            raise ValueError(f"Invalid time interval, ({start}, {end})")

        for i, t_i in enumerate(self.time):
            if t_i >= end:
                index = i
                break
        else:
            raise ValueError(f"Time interval ({start}, {end}) not contained in Input.time")
        if index > 0 and self.time[index - 1] > start:
            raise ValueError(f"Time interval ({start}, {end}) spans more than one interval")
        return index


class Solution(BaseModel):
    input_data: Input
    total_cost: float
    energy_cost: float
    power_cost: float
    max_charging_power_used: float
    charging_power: list[list[float]]
    effective_charging_power: list[list[float]]
    state_of_energy: list[list[float]]
    lower_soe_envelope: list[list[float]]

    @model_validator(mode="after")
    def check_list_lengths(self):
        num_vehicles = self.input_data.num_vehicles
        num_timesteps = self.input_data.num_timesteps

        # assert all list contain [num_vehicles] lists
        if not len(self.charging_power) == num_vehicles:
            raise ValueError(f"Field[charging_power] has length {len(self.charging_power)}, expected {num_vehicles}")
        if not len(self.effective_charging_power) == num_vehicles:
            raise ValueError(
                f"Field[effective_charging_power] has length {len(self.effective_charging_power)}, expected {num_vehicles}"
            )
        if not len(self.state_of_energy) == num_vehicles:
            raise ValueError(f"Field[state_of_energy] has length {len(self.state_of_energy)}, expected {num_vehicles}")

        if not len(self.lower_soe_envelope) == num_vehicles:
            raise ValueError(
                f"Field[lower_soe_envelope] has length {len(self.lower_soe_envelope)}, expected {num_vehicles}"
            )

        # assert all lists contain lists with [num_timesteps] values
        for values in self.charging_power:
            if not len(values) == num_timesteps:
                raise ValueError(
                    f"Field[charging_power] contains list of length {len(values)}, expected {num_timesteps}"
                )
        for values in self.effective_charging_power:
            if not len(values) == num_timesteps:
                raise ValueError(
                    f"Field[effective_charging_power] contains list of length {len(values)}, expected {num_timesteps}"
                )
        for values in self.state_of_energy:
            if not len(values) == num_timesteps + 1:
                raise ValueError(
                    f"Field[state_of_energy] contains list of length {len(values)}, expected {num_timesteps + 1}"
                )
        for values in self.lower_soe_envelope:
            if not len(values) == num_timesteps + 1:
                raise ValueError(
                    f"Field[lower_soe_envelope] contains list of length {len(values)}, expected {num_timesteps + 1}"
                )

        return self
