from typing import Literal


class VariableSpec:
    def __init__(
            self,
            name: str,
            lb: float | None = None,
            ub: float | None = None,
            vtype: str = "continuous",
    ) -> None:
        self.name: str = name
        self.lb: float | None = lb
        self.ub: float | None = ub
        self.vtype: str = vtype


class QuadraticExpression:
    def __init__(
            self,
            constant: float = 0.0,
            linear: dict[VariableSpec, float] | None = None,
            quadratic: dict[tuple[VariableSpec, VariableSpec], float] | None = None,
    ) -> None:
        self.constant: float = constant
        self.linear: dict[VariableSpec, float] | None = linear
        self.quadratic: dict[tuple[VariableSpec, VariableSpec], float] | None = quadratic


class ConstraintSpec:
    def __init__(
            self,
            lhs: QuadraticExpression,
            sense: Literal["==", "<=", ">="],
    ) -> None:
        self.lhs: QuadraticExpression = lhs
        self.sense: Literal["==", "<=", ">="] = sense
