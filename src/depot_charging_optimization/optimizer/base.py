from typing import Protocol

from depot_charging_optimization.data_models import Solution


class Optimizer(Protocol):
    def build(self) -> None: ...

    def solve(self) -> Solution | None: ...
