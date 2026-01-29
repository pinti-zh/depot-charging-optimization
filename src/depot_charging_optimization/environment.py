from math import sqrt
from random import gauss

from pydantic import BaseModel, model_validator

from depot_charging_optimization.config import EnvironmentConfig
from depot_charging_optimization.data_models import Input, Solution


class State(BaseModel):
    num_vehicles: int
    state_of_energy: list[float]
    in_depot: list[bool]
    battery_capacity: list[float]

    @model_validator(mode="after")
    def check_list_lengths(self):
        if not len(self.state_of_energy) == self.num_vehicles:
            raise ValueError(
                f"Field[state_of_energy] has length {len(self.state_of_energy)}, {self.num_vehicles} expected"
            )
        if not len(self.in_depot) == self.num_vehicles:
            raise ValueError(f"Field[in_depot] has length {len(self.in_depot)}, {self.num_vehicles} expected")
        if not len(self.battery_capacity) == self.num_vehicles:
            raise ValueError(
                f"Field[battery_capacity] has length {len(self.battery_capacity)}, {self.num_vehicles} expected"
            )
        return self

    def update_soe(self, energy: list[float]):
        assert len(energy) == self.num_vehicles
        for i in range(self.num_vehicles):
            self.state_of_energy[i] = min(self.state_of_energy[i] + energy[i], self.battery_capacity[i])

    def is_valid(self):
        return all((soe >= -1.0e-6 * cap) for soe, cap in zip(self.state_of_energy, self.battery_capacity))


class Charger:
    def __init__(self, max_charging_power: float, max_efficiency: float, loss_coefficient: float):
        self.max_charging_power: float = max_charging_power
        self.max_efficiency: float = max_efficiency
        self.loss_coefficient: float = loss_coefficient

    def max_possible_effective_charging_power(self) -> float:
        return self.effective_charging_power(self.max_charging_power)

    def effective_charging_power(self, charging_power: float) -> float:
        if self.max_charging_power == 0:
            return 0.0
        else:
            charging_power = min(charging_power, self.max_charging_power)
            return self.max_efficiency * (
                charging_power - (self.loss_coefficient / 2) * charging_power**2 / self.max_charging_power
            )

    def inverse_effective_charging_power(self, effective_charging_power: float) -> float:
        if effective_charging_power > self.max_possible_effective_charging_power():
            raise ValueError(f"Effective charging power {effective_charging_power} is not achievable")
        if self.loss_coefficient == 0:
            return effective_charging_power / self.max_efficiency
        kappa = self.loss_coefficient / self.max_charging_power
        root_term = sqrt(1 - (2 * kappa / self.max_efficiency) * effective_charging_power)
        return (1 - root_term) / kappa


class Environment:
    def __init__(self, plan: Input, config: EnvironmentConfig = EnvironmentConfig()):
        self.plan: Input = plan.model_copy(deep=True).loop(config.num_days)
        self.config: EnvironmentConfig = config
        self.charger: Charger = Charger(
            config.charger_max_charging_power, config.charger_max_efficiency, config.charger_loss_coefficient
        )
        self.state: State | None = None
        self.timestep: int = 0
        self.energy_std_dev: float = config.env_energy_std_dev
        self.state_history: list[State] = []
        self.policy_history: list[list[float]] = []
        self.time_delta: list[int] = [t2 - t1 for t1, t2 in zip([0] + self.plan.time[:-1], self.plan.time)]

    def reset(self, initial_soe: list[float]):
        assert len(initial_soe) == self.plan.num_vehicles
        self.state = State(
            num_vehicles=self.plan.num_vehicles,
            state_of_energy=initial_soe,
            battery_capacity=self.plan.battery_capacity,
            in_depot=[dc[0] for dc in self.plan.depot_charge],
        )
        self.timestep = 0
        self.state_history = [self.state.model_copy(deep=True)]
        self.policy_history = []

    def step(self, policy: list[float]) -> State:
        if self.state is None:
            raise RuntimeError("Environment has not been reset, call Environment.reset() first")
        if self.timestep >= self.plan.num_timesteps:
            return self.state
        assert len(policy) == self.plan.num_vehicles
        policy = [p * dc for dc, p in zip(self.state.in_depot, policy)]
        energy_delta = []
        for i in range(self.state.num_vehicles):
            if self.state.in_depot[i]:
                energy_delta.append(self.time_delta[self.timestep] * self.charger.effective_charging_power(policy[i]))
            else:
                energy_delta.append(0.0)
            energy_delta[i] -= (self.plan.energy_demand[i][self.timestep]) * gauss(1, self.energy_std_dev)

        self.timestep += 1
        self.state.update_soe(energy_delta)
        if self.timestep < self.plan.num_timesteps:
            self.state.in_depot = [self.plan.depot_charge[i][self.timestep] for i in range(self.state.num_vehicles)]
        self.state_history.append(self.state.model_copy(deep=True))
        self.policy_history.append(policy)
        return self.state

    def get_solution(self) -> Solution:
        charging_power = [list(column) for column in zip(*self.policy_history)]
        effective_charging_power = [list(map(self.charger.effective_charging_power, cp)) for cp in charging_power]
        total_charging_power = [sum(policy) for policy in self.policy_history]
        energy_used = []
        for cp, t1, t2 in zip(total_charging_power, [0] + self.plan.time[:-1], self.plan.time):
            energy_used.append(cp * (t2 - t1))
        assert self.plan.energy_price is not None
        assert self.plan.grid_tariff is not None
        energy_cost = sum(p * e for p, e in zip(self.plan.energy_price, energy_used))
        power_cost = max(total_charging_power) * self.plan.grid_tariff * self.config.num_days
        state_of_energy = [list(column) for column in zip(*(state.state_of_energy for state in self.state_history))]
        return Solution(
            input_data=self.plan.model_copy(deep=True).truncate(len(total_charging_power)),
            total_cost=energy_cost + power_cost,
            energy_cost=energy_cost,
            power_cost=power_cost,
            max_charging_power_used=max(total_charging_power),
            charging_power=charging_power,
            effective_charging_power=effective_charging_power,
            state_of_energy=state_of_energy,
            lower_soe_envelope=state_of_energy,
        )
