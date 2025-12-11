from typing import Protocol

from depot_charging_optimization.config import OptimizerConfig
from depot_charging_optimization.data_models import Input, Solution


class Optimizer(Protocol):

    def build(self) -> None:
        ...

    def solve(self) -> Solution | None:
        ...