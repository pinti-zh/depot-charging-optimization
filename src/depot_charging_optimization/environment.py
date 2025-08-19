from typing import Callable

from depot_charging_optimization.data_models import Input


class Environment:
    def __init__(self, plan: Input, initial_soe: list[float], charging_efficiency: Callable[[float], float]):
        self._plan: Input = plan.model_copy()
        self.soe: list[float] = initial_soe
        self.in_depot: list[bool] = []
        self.charging_efficiency = charging_efficiency

    def step(self, policy: list[float]) -> list[float | None]:
        self.in_depot = [dc[0] for dc in self._plan.depot_charge]
        for i in range(len(self.soe)):
            demand = self._plan.energy_demand[i][0]
            if self.in_depot[i]:
                self.soe[i] += self.charging_efficiency(policy[i]) * self._plan.time[0]
            self.soe[i] -= demand
        self._plan = self._plan.rotate()
        return [state if dc else None for dc, state in zip(self.in_depot, self.soe)]

    @property
    def plan(self) -> Input:
        return self._plan.model_copy()
