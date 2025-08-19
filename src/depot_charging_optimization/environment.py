from depot_charging_optimization.data_models import Input


class Environment:
    def __init__(self, initial_soe: list[float]):
        self.soe: list[float] = initial_soe
        self.in_depot: list[bool] = []

    def step(self, plan: Input, policy: list[float]) -> list[float | None]:
        self.in_depot = [dc[0] for dc in plan.depot_charge]
        for i in range(len(self.soe)):
            demand = plan.energy_demand[i][0]
            if self.in_depot[i]:
                self.soe[i] += policy[i] * plan.time[0]
            self.soe[i] -= demand
        return [state if dc else None for dc, state in zip(self.in_depot, self.soe)]
