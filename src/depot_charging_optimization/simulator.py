import math
from abc import ABC, abstractmethod
from typing import Any

from depot_charging_optimization.data_models import Input, Solution


class Simulator(ABC):
    def __init__(self, input_data: Input):
        self.input_data: Input = input_data
        self._solution: Solution | None = None

        # decision variables
        self._state_of_energy: list[list[float]] = []
        self._charging_power: list[list[float]] = []
        self._effective_charging_power: list[list[float]] = []

        # aux variables
        self._delta_time: list[int] = [
            t2 - t1 for t1, t2 in zip([0] + self.input_data.time[:-1], self.input_data.time)
        ]

    def run(self, **kwargs: Any) -> Solution | None:
        self._setup(**kwargs)
        for t_i in range(self.input_data.num_timesteps):
            self._on_timestep(t_i, **kwargs)
        self._post_processing(**kwargs)
        return self._solution

    @property
    def state_of_energy(self) -> list[list[float]]:
        return self._state_of_energy

    @property
    def charging_power(self) -> list[list[float]]:
        return self._charging_power

    @property
    def effective_charging_power(self) -> list[list[float]]:
        return self._effective_charging_power

    @property
    def max_charging_power(self) -> float:
        return max(self.total_charging_power)

    @property
    def total_charging_power(self) -> list[list[float]]:
        tcp = []
        cp = self.charging_power
        for t_i in range(self.input_data.num_timesteps):
            tcp.append(sum(cp[vehicle][t_i] for vehicle in range(self.input_data.num_vehicles)))
        return tcp

    @property
    def energy_cost(self) -> float:
        tcp = self.total_charging_power
        cost = sum(
            tcp[t_i] * self._delta_time[t_i] * self.input_data.energy_price[t_i]
            for t_i in range(self.input_data.num_timesteps)
        )
        return cost

    @property
    def power_cost(self) -> float:
        return self.max_charging_power * self.input_data.grid_tariff

    @abstractmethod
    def _setup(self, **kwargs) -> None:
        pass

    @abstractmethod
    def _on_timestep(self, t_i: int, **kwargs) -> None:
        pass

    @abstractmethod
    def _post_processing(self, **kwargs) -> None:
        pass


class GreedySimulator(Simulator):
    def __init__(self, input_data: Input):
        super().__init__(input_data)
        self._timestamp_map: list[list[int]] = []

    def _setup(self, **kwargs) -> None:
        for vehicle in range(self.input_data.num_vehicles):
            self._state_of_energy.append([0.0 for _ in range(self.input_data.num_timesteps + 1)])
            self._charging_power.append([0.0 for _ in range(self.input_data.num_timesteps)])
            self._effective_charging_power.append([0.0 for _ in range(self.input_data.num_timesteps)])
            t_start = first_depot_departure(self.input_data.depot_charge[vehicle])
            self._timestamp_map.append(
                [(t_i + t_start) % self.input_data.num_timesteps for t_i in range(self.input_data.num_timesteps)]
            )
            self._state_of_energy[vehicle][t_start] = 0.8 * self.input_data.battery_capacity[vehicle]

    def _on_timestep(self, t_i, ce_function_type="one", alpha=1.0, **kwargs) -> None:
        for vehicle in range(self.input_data.num_vehicles):
            t_m = self._timestamp_map[vehicle][t_i]
            next_soe = self._state_of_energy[vehicle][t_m] - self.input_data.energy_demand[vehicle][t_m]
            max_soe = 0.8 * self.input_data.battery_capacity[vehicle]
            if self.input_data.depot_charge[vehicle][t_m] and (next_soe < max_soe):
                desired_power = (max_soe - next_soe) / self._delta_time[t_m]
                cp, ecp = power_and_effective_power(
                    desired_power, self.input_data.max_charging_power, ce_function_type, alpha
                )
                next_soe += ecp * self._delta_time[t_m]
                self._charging_power[vehicle][t_m] = cp
                self._effective_charging_power[vehicle][t_m] = ecp
            t_next = self._timestamp_map[vehicle][(t_i + 1) % self.input_data.num_timesteps]
            self._state_of_energy[vehicle][t_next] = next_soe

    def _post_processing(self, **kwargs) -> None:
        for vehicle in range(self.input_data.num_vehicles):
            self._state_of_energy[vehicle][-1] = self._state_of_energy[vehicle][0]
        self._solution = Solution(
            input_data=self.input_data,
            total_cost=self.energy_cost + self.power_cost,
            energy_cost=self.energy_cost,
            power_cost=self.power_cost,
            gap=0.0,
            max_charging_power_used=self.max_charging_power,
            charging_power=self.charging_power,
            effective_charging_power=self.effective_charging_power,
            state_of_energy=self.state_of_energy,
        )


class PeakShavingSimulator(Simulator):
    def __init__(self, input_data: Input, max_power: float):
        super().__init__(input_data)
        self._timestamp_map: list[list[int]] = []
        self._max_power = max_power

    def _setup(self, **kwargs) -> None:
        for vehicle in range(self.input_data.num_vehicles):
            self._state_of_energy.append([0.0 for _ in range(self.input_data.num_timesteps + 1)])
            self._charging_power.append([0.0 for _ in range(self.input_data.num_timesteps)])
            self._effective_charging_power.append([0.0 for _ in range(self.input_data.num_timesteps)])
            t_start = first_depot_departure(self.input_data.depot_charge[vehicle])
            self._timestamp_map.append(
                [(t_i + t_start) % self.input_data.num_timesteps for t_i in range(self.input_data.num_timesteps)]
            )
            self._state_of_energy[vehicle][t_start] = 0.8 * self.input_data.battery_capacity[vehicle]

    def _on_timestep(self, t_i, ce_function_type="one", alpha=1.0, **kwargs) -> None:
        for vehicle in range(self.input_data.num_vehicles):
            t_m = self._timestamp_map[vehicle][t_i]
            next_soe = self._state_of_energy[vehicle][t_m] - self.input_data.energy_demand[vehicle][t_m]
            max_soe = 0.8 * self.input_data.battery_capacity[vehicle]
            if self.input_data.depot_charge[vehicle][t_m] and (next_soe < max_soe):
                tund = time_until_next_departure(self._delta_time, self.input_data.depot_charge[vehicle], t_m)
                desired_power = (max_soe - next_soe) / tund
                cp, ecp = power_and_effective_power(desired_power, self._max_power, ce_function_type, alpha)
                next_soe += ecp * self._delta_time[t_m]
                self._charging_power[vehicle][t_m] = cp
                self._effective_charging_power[vehicle][t_m] = ecp
            t_next = self._timestamp_map[vehicle][(t_i + 1) % self.input_data.num_timesteps]
            self._state_of_energy[vehicle][t_next] = next_soe

    def _post_processing(self, **kwargs) -> None:
        for vehicle in range(self.input_data.num_vehicles):
            self._state_of_energy[vehicle][-1] = self._state_of_energy[vehicle][0]
        self._solution = Solution(
            input_data=self.input_data,
            total_cost=self.energy_cost + self.power_cost,
            energy_cost=self.energy_cost,
            power_cost=self.power_cost,
            gap=0.0,
            max_charging_power_used=self.max_charging_power,
            charging_power=self.charging_power,
            effective_charging_power=self.effective_charging_power,
            state_of_energy=self.state_of_energy,
        )


def first_depot_departure(depot_charge) -> int:
    state = False
    for t_i, dc in enumerate(depot_charge):
        if state and not dc:
            return t_i
        state = dc
    return 0


def power_and_effective_power(
    desired_power: float, max_power: float, ce_function_type: str, alpha: float
) -> (float, float):
    if ce_function_type == "one":
        cp = min(desired_power, max_power)
        return cp, cp
    elif ce_function_type == "constant":
        cp = desired_power / alpha
        cp = min(cp, max_power)
        return cp, alpha * cp
    elif ce_function_type == "quadratic":
        if desired_power >= max_power * (1 - (1 - alpha) / 2):
            return max_power, max_power * (1 - (1 - alpha) / 2)
        else:
            k = (1 - alpha) / (2 * max_power)
            root_term = math.sqrt(1 - 4 * k * desired_power)
            cp = -(root_term - 1) / (2 * k)
            return cp, desired_power
    else:
        raise ValueError(f"Unknown ce_function_type: {ce_function_type}")


def time_until_next_departure(delta_time: list[float], depot_charge: list[bool], start_time_index: int) -> int | float:
    dt = 0
    t_i = start_time_index
    for _ in range(len(delta_time)):
        if not depot_charge[t_i % len(depot_charge)]:
            return dt
        dt += delta_time[t_i % len(delta_time)]
        t_i += 1
    return float("inf")
