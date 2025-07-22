import contextlib
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import casadi as ca
import gurobipy as gp
from gurobipy import GRB

from depot_charging_optimization.data_models import Input, Solution

OptVariable = TypeVar("OptVariable", gp.Var, ca.MX.sym)


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class Optimizer(ABC, Generic[OptVariable]):
    def __init__(self, input_data: Input, name: str | None = None, greedy: bool = False, **kwargs):
        self.input_data: Input = input_data
        self.name: str = name or self.__class__.__name__
        self.greedy: bool = greedy

        # State
        self._built: bool = False

        # Auxiliary variables
        self._num_vehicles: int = input_data.num_vehicles
        self._num_timesteps: int = input_data.num_timesteps
        self._delta_time: list[int] = [
            t2 - t1 for t1, t2 in zip([0] + self.input_data.time[:-1], self.input_data.time)
        ]
        self.gap: float = 0.0

        # Optimization variables
        self._charging_power: list[list[OptVariable]] = []
        self._state_of_energy: list[list[OptVariable]] = []
        self._effective_charging_power: list[list[OptVariable]] = []
        self._mcp: OptVariable | None = None

        # Scaling factors
        self._factor_cp: float = 1.0 / self.input_data.max_charging_power
        self._factor_soe: float = 1.0 / max(self.input_data.battery_capacity)
        self._factor_ep: float = 1.0 / max(self.input_data.energy_price)

        # Variable bounds
        self._lb_cp: list[list[float]] = [[0.0 for _ in range(self._num_timesteps)] for _ in range(self._num_vehicles)]
        self._ub_cp: list[list[float]] = [[1.0 for _ in range(self._num_timesteps)] for _ in range(self._num_vehicles)]

        self._lb_ecp: list[list[float]] = [
            [0.0 for _ in range(self._num_timesteps)] for _ in range(self._num_vehicles)
        ]
        self._ub_ecp: list[list[float]] = [
            [1.0 for _ in range(self._num_timesteps)] for _ in range(self._num_vehicles)
        ]

        self._lb_soe: list[list[float]] = [
            [
                0.2 * self.input_data.battery_capacity[vehicle] * self._factor_soe
                for _ in range(self._num_timesteps + 1)
            ]
            for vehicle in range(self._num_vehicles)
        ]
        self._ub_soe: list[list[float]] = [
            [
                0.8 * self.input_data.battery_capacity[vehicle] * self._factor_soe
                for _ in range(self._num_timesteps + 1)
            ]
            for vehicle in range(self._num_vehicles)
        ]

        self._lb_mcp: float = 0.0
        self._ub_mcp: float = float(self._num_vehicles)

    def build(self, **kwargs: Any) -> None:
        self._set_all_variables(**kwargs)
        self._set_all_constraints(**kwargs)
        self._set_objective(**kwargs)
        self._built = True

    @abstractmethod
    def _set_variable(self, name: str, lb: float, ub: float) -> OptVariable:
        pass

    @property
    @abstractmethod
    def charging_power(self) -> list[list[float]]:
        pass

    @property
    @abstractmethod
    def effective_charging_power(self) -> list[list[float]]:
        pass

    @property
    @abstractmethod
    def state_of_energy(self) -> list[list[float]]:
        pass

    @property
    def max_charging_power(self) -> float:
        mcp: float = 0.0
        charging_power: list[list[float]] = self.charging_power
        for t_i in range(self._num_timesteps):
            column = [charging_power[vehicle][t_i] for vehicle in range(self._num_vehicles)]
            mcp = max(mcp, sum(column))
        return mcp

    @property
    def energy_cost(self) -> float:
        energy_cost_vector = []
        for _ in range(self._num_vehicles):
            energy_cost_vector += [
                self.input_data.energy_price[t_i] * self._delta_time[t_i] for t_i in range(self._num_timesteps)
            ]
        energy_cost_vector = [cp * ec for cp, ec in zip(flatten_lol(self.charging_power), energy_cost_vector)]
        return sum(energy_cost_vector)

    @property
    def power_cost(self) -> float:
        return self.max_charging_power * self.input_data.grid_tariff

    @property
    def solution(self) -> Solution | None:
        return Solution(
            input_data=self.input_data,
            total_cost=self.energy_cost + self.power_cost,
            energy_cost=self.energy_cost,
            power_cost=self.power_cost,
            gap=self.gap,
            max_charging_power_used=self.max_charging_power,
            charging_power=self.charging_power,
            effective_charging_power=self.effective_charging_power,
            state_of_energy=self.state_of_energy,
        )

    def _set_all_variables(self, **kwargs: Any) -> None:
        for vehicle in range(self._num_vehicles):
            self._charging_power.append(
                [
                    self._set_variable(
                        f"chargingPower_v{vehicle}_{t_i}", self._lb_cp[vehicle][t_i], self._ub_cp[vehicle][t_i]
                    )
                    for t_i in range(self._num_timesteps)
                ]
            )
            self._effective_charging_power.append(
                [
                    self._set_variable(
                        f"effectiveChargingPower_v{vehicle}_{t_i}",
                        self._lb_ecp[vehicle][t_i],
                        self._ub_ecp[vehicle][t_i],
                    )
                    for t_i in range(self._num_timesteps)
                ]
            )
            self._state_of_energy.append(
                [
                    self._set_variable(
                        f"stateOfEnergy_v{vehicle}_{t_i}", self._lb_soe[vehicle][t_i], self._ub_soe[vehicle][t_i]
                    )
                    for t_i in range(self._num_timesteps + 1)
                ]
            )
            self._mcp = self._set_variable("maxChargingPower", self._lb_mcp, self._ub_mcp)

    @abstractmethod
    def _set_all_constraints(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def _set_objective(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def solve(self) -> None | Solution:
        pass


class GurobiOptimizer(Optimizer[gp.Var]):
    def __init__(self, input_data: Input, name: str = "GurobiOptimizer", greedy: bool = False, time_limit: int = 5):
        super().__init__(input_data, name=name, greedy=greedy)
        with suppress_stdout_stderr():
            self._model: gp.Model = gp.Model(self.name)
            self._model.setParam("LogToConsole", 0)
            self._model.setParam("OutputFlag", 1)
            self._model.setParam("TimeLimit", time_limit)

    @property
    def charging_power(self) -> list[list[float]]:
        return [list(map(lambda v: v.X / self._factor_cp, sublist)) for sublist in self._charging_power]

    @property
    def effective_charging_power(self) -> list[list[float]]:
        return [list(map(lambda v: v.X, sublist)) for sublist in self._effective_charging_power]

    @property
    def state_of_energy(self) -> list[list[float]]:
        return [list(map(lambda v: v.X / self._factor_soe, sublist)) for sublist in self._state_of_energy]

    def _set_variable(self, name: str, lb: float, ub: float) -> gp.Var:
        return self._model.addVar(name=name, vtype=GRB.CONTINUOUS, lb=lb, ub=ub)

    def _set_all_constraints(
        self, ce_function_type: str = "one", alpha: float = 1.0, cp_throttle: float = 1.0, **kwargs
    ) -> None:
        # energy flow
        for vehicle in range(self._num_vehicles):
            for t_i in range(self._num_timesteps):
                if ce_function_type == "one":
                    self._model.addConstr(
                        self._effective_charging_power[vehicle][t_i] == self._charging_power[vehicle][t_i],
                        f"effectiveChargingPower_v{vehicle}_{t_i}",
                    )
                elif ce_function_type == "constant":
                    self._model.addConstr(
                        self._effective_charging_power[vehicle][t_i] == alpha * self._charging_power[vehicle][t_i],
                        f"effectiveChargingPower_v{vehicle}_{t_i}",
                    )
                elif ce_function_type == "quadratic":
                    self._model.addConstr(
                        self._effective_charging_power[vehicle][t_i]
                        == self._charging_power[vehicle][t_i]
                        - (1 - alpha) * self._charging_power[vehicle][t_i] ** 2 / 2,
                        f"effectiveChargingPower_v{vehicle}_{t_i}",
                    )
                else:
                    raise ValueError(f"Unknown ce_function_type: {ce_function_type}")

                self._model.addConstr(
                    self._state_of_energy[vehicle][t_i + 1]
                    == self._state_of_energy[vehicle][t_i]
                    + self._effective_charging_power[vehicle][t_i]
                    * self._delta_time[t_i]
                    * (self._factor_soe / self._factor_cp)
                    - self.input_data.energy_demand[vehicle][t_i] * self._factor_soe,
                    f"energyFlow_v{vehicle}_{t_i}",
                )
                if not self.input_data.depot_charge[vehicle][t_i]:
                    self._model.addConstr(
                        self._charging_power[vehicle][t_i] == 0,
                        f"noChargingPower_v{vehicle}_{t_i}",
                    )

        # energy loop
        for vehicle in range(self._num_vehicles):
            self._model.addConstr(
                self._state_of_energy[vehicle][0] <= self._state_of_energy[vehicle][self._num_timesteps],
                f"energyLoop_v{vehicle}",
            )

        # max power used
        for index in range(self._num_timesteps):
            self._model.addConstr(
                self._mcp
                >= gp.quicksum(self._charging_power[vehicle][index] for vehicle in range(self._num_vehicles)),
                f"maxChargingPower_{index}",
            )

        # charging power throttle
        self._model.addConstr(
            self._mcp <= cp_throttle * self._num_vehicles,
            "maxChargingPowerThrottle",
        )

    def _set_objective(self, **kwargs) -> None:
        assert self.input_data.energy_price is not None, "Energy price not provided"
        assert self.input_data.grid_tariff is not None, "Grid tariff not provided"

        if self.greedy:
            self._model.setObjective(
                gp.quicksum(gp.quicksum(self._state_of_energy[vehicle]) for vehicle in range(self._num_vehicles)),
                GRB.MAXIMIZE,
            )
        else:
            self._model.setObjective(
                gp.quicksum(
                    gp.quicksum(
                        self.input_data.energy_price[i] * cp * self._factor_ep * self._delta_time[i]
                        for i, cp in enumerate(self._charging_power[vehicle])
                    )
                    for vehicle in range(self._num_vehicles)
                )
                + self._mcp * self.input_data.grid_tariff * self._factor_ep,
                GRB.MINIMIZE,
            )

    def solve(self) -> Solution | None:
        if not self._built:
            raise ValueError("Optimizer not built")
        self._model.optimize()

        try:
            objective_value = self._model.ObjVal
            if self._model.Status != GRB.OPTIMAL:
                obj_bound = self._model.ObjBound
                self.gap = abs(objective_value - obj_bound) / (abs(objective_value) + 1e-10)

        except AttributeError:
            return None

        return self.solution


class CasadiOptimizer(Optimizer[ca.MX.sym]):
    def __init__(self, input_data: Input, name: str = "CasadiOptimizer", greedy: bool = False):
        super().__init__(input_data, name=name, greedy=greedy)

        self._constraints: list[ca.casadi.MX] = []
        self._constraints_lb: list[float] = []
        self._constraints_ub: list[float] = []
        self._objective: ca.MX | None = None
        self.solution_dict: dict | None = None

    def _set_variable(self, name: str, lb: float, ub: float) -> ca.MX.sym:
        return ca.MX.sym(name)

    @property
    def charging_power(self) -> list[list[float]]:
        if self.solution_dict is None:
            raise ValueError("Solution is not computed")
        return [
            [
                float(self.solution_dict["x"][vehicle * self._num_timesteps + t_i]) / self._factor_cp
                for t_i in range(self._num_timesteps)
            ]
            for vehicle in range(self._num_vehicles)
        ]

    @property
    def effective_charging_power(self) -> list[list[float]]:
        if self.solution_dict is None:
            raise ValueError("Solution is not computed")
        offset = self._num_vehicles * self._num_timesteps
        return [
            [
                float(self.solution_dict["x"][offset + vehicle * self._num_timesteps + t_i])
                for t_i in range(self._num_timesteps)
            ]
            for vehicle in range(self._num_vehicles)
        ]

    @property
    def state_of_energy(self) -> list[list[float]]:
        if self.solution_dict is None:
            raise ValueError("Solution is not computed")
        offset = 2 * self._num_vehicles * self._num_timesteps
        return [
            [
                float(self.solution_dict["x"][offset + vehicle * (self._num_timesteps + 1) + t_i]) / self._factor_soe
                for t_i in range(self._num_timesteps + 1)
            ]
            for vehicle in range(self._num_vehicles)
        ]

    def _set_all_constraints(
        self, ce_function_type: str = "one", alpha: float = 1.0, cp_throttle: float = 1.0, **kwargs
    ) -> None:
        # energy flow
        for vehicle in range(self._num_vehicles):
            for t_i in range(self._num_timesteps):
                if ce_function_type == "one":
                    self._constraints.append(
                        self._effective_charging_power[vehicle][t_i] - self._charging_power[vehicle][t_i]
                    )
                elif ce_function_type == "constant":
                    self._constraints.append(
                        self._effective_charging_power[vehicle][t_i] - alpha * self._charging_power[vehicle][t_i]
                    )
                elif ce_function_type == "quadratic":
                    self._constraints.append(
                        self._effective_charging_power[vehicle][t_i]
                        - (
                            self._charging_power[vehicle][t_i]
                            - ((1 - alpha) / 2) * self._charging_power[vehicle][t_i] ** 2
                        )
                    )
                else:
                    raise ValueError(f"Unknown ce_function_type: {ce_function_type}")

                self._constraints_lb.append(0)
                self._constraints_ub.append(0)

                self._constraints.append(
                    self._state_of_energy[vehicle][t_i + 1]
                    - (
                        self._state_of_energy[vehicle][t_i]
                        + self._effective_charging_power[vehicle][t_i]
                        * self._delta_time[t_i]
                        * (self._factor_soe / self._factor_cp)
                        - self.input_data.energy_demand[vehicle][t_i] * self._factor_soe
                    )
                )
                self._constraints_lb.append(0)
                self._constraints_ub.append(0)

                if not self.input_data.depot_charge[vehicle][t_i]:
                    self._constraints.append(self._charging_power[vehicle][t_i])
                    self._constraints_lb.append(0)
                    self._constraints_ub.append(0)

        # energy loop
        for vehicle in range(self._num_vehicles):
            self._constraints.append(
                self._state_of_energy[vehicle][self._num_timesteps] - self._state_of_energy[vehicle][0]
            )
            self._constraints_lb.append(0)
            self._constraints_ub.append(float("inf"))

        # max power used
        for t_i in range(self._num_timesteps):
            column = [self._charging_power[vehicle][t_i] for vehicle in range(self._num_vehicles)]
            self._constraints.append(self._mcp - sum(column, ca.MX(0)))
            self._constraints_lb.append(0)
            self._constraints_ub.append(float("inf"))

        # charging power throttle
        self._constraints.append(cp_throttle * self._num_vehicles - self._mcp)
        self._constraints_lb.append(0)
        self._constraints_ub.append(float("inf"))

    def _set_objective(self, **kwargs) -> None:
        assert self.input_data.energy_price is not None, "Energy price not provided"
        assert self.input_data.grid_tariff is not None, "Grid tariff not provided"

        if self.greedy:
            self._objective = -sum(
                (
                    sum((self._state_of_energy[vehicle][t_i] for t_i in range(self._num_timesteps)), ca.MX(0))
                    for vehicle in range(self._num_vehicles)
                ),
                ca.MX(0),
            )
        else:
            energy_cost_vector = []
            for _ in range(self._num_vehicles):
                energy_cost_vector += [
                    self.input_data.energy_price[t_i] * self._delta_time[t_i] * self._factor_ep
                    for t_i in range(self._num_timesteps)
                ]
            energy_cost_vector = [cp * ec for cp, ec in zip(flatten_lol(self._charging_power), energy_cost_vector)]
            self._objective = (
                sum(energy_cost_vector, ca.MX(0)) + self.input_data.grid_tariff * self._mcp * self._factor_ep
            )

    def solve(self) -> Solution | None:
        if not self._built:
            raise ValueError("Optimizer not built")
        nlp = {
            "x": ca.vertcat(
                *flatten_lol(self._charging_power),
                *flatten_lol(self._effective_charging_power),
                *flatten_lol(self._state_of_energy),
                self._mcp,
            ),
            "f": self._objective,
            "g": ca.vertcat(*self._constraints),
        }
        solver = ca.nlpsol("solver", "ipopt", nlp)

        self.solution_dict = solver(
            lbg=ca.vertcat(*self._constraints_lb),
            ubg=ca.vertcat(*self._constraints_ub),
            lbx=ca.vertcat(*self._lb_cp, *self._lb_ecp, *self._lb_soe, 0),
            ubx=ca.vertcat(*self._ub_cp, *self._ub_ecp, *self._ub_soe, float("inf")),
        )

        if not solver.stats()["success"]:
            return None

        return self.solution


def flatten_lol(list_of_lists: list[Any]):
    return [item for sublist in list_of_lists for item in sublist]
