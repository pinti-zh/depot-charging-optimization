from depot_charging_optimization.core import Optimizer
from depot_charging_optimization.data_models import Input, Solution
from depot_charging_optimization.logging import suppress_stdout_stderr


class Controller:
    def __init__(self, plan: Input, optimizer: Optimizer):
        self.plan = plan
        self.optimizer = optimizer

    def step(self) -> None:
        self.plan = self.plan.rotate()

    def get_policy(self) -> list[float | None]:
        with suppress_stdout_stderr():
            pass
        return [None]


def policy_from_solution(solution: Solution) -> list[float | None]:
    return [cp[0] for cp in solution.charging_power]
