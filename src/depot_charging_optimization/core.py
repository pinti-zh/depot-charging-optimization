from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import casadi as ca
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm

from depot_charging_optimization.data_models import Input, Solution
from depot_charging_optimization.logging import suppress_stdout_stderr

OptVariable = TypeVar("OptVariable", gp.Var, ca.MX.sym)


class Optimizer(ABC, Generic[OptVariable]):
    def __init__(
        self,
        input_data: Input,
        name: str | None = None,
        bidirectional_charging: bool = False,
        initial_soe: list[float | None] | None = None,
        confidence_level: float = 0.0,
        energy_std_dev: float = 0.0,
        **kwargs,
    ):
        self.input_data: Input = input_data
        self.name: str = name or self.__class__.__name__
        self._initial_soe: list[float | None] | None = initial_soe
        self._confidence_level: float = confidence_level
        self._energy_std_dev: float = energy_std_dev

        # State
        self._built: bool = False

        # Auxiliary variables
        self._num_vehicles: int = input_data.num_vehicles
        self._num_timesteps: int = input_data.num_timesteps
        self._delta_time: list[int] = [
            t2 - t1 for t1, t2 in zip([0] + self.input_data.time[:-1], self.input_data.time)
        ]
        self.gap: float = 0.0
        self._alpha: float | None = None
        self._ce_mode: str | None = None

        # Optimization variables
        self._charging_power: list[list[OptVariable]] = []
        self._state_of_energy: list[list[OptVariable]] = []
        self._effective_charging_power: list[list[OptVariable]] = []
        self._total_charging_power: list[OptVariable] = []
        self._mcp: OptVariable | None = None

        # envelope for stochastic robustness
        self._lower_soe_envelope: list[list[OptVariable]] = []

        # Scaling factors
        self._factor_cp: float = 1.0 / self.input_data.max_charging_power
        self._factor_soe: float = 1.0 / max(self.input_data.battery_capacity)
        assert self.input_data.energy_price is not None
        self._factor_ep: float = 1.0 / max(self.input_data.energy_price)

        # Variable bounds
        assert self.input_data.is_battery is not None
        vehicle_lower_bounds = [
            -1.0 if (is_battery or bidirectional_charging) else 0.0 for is_battery in self.input_data.is_battery
        ]
        self._lb_cp: list[list[float]] = [[lb for _ in range(self._num_timesteps)] for lb in vehicle_lower_bounds]
        self._ub_cp: list[list[float]] = [[1.0 for _ in range(self._num_timesteps)] for _ in range(self._num_vehicles)]

        self._lb_ecp: list[list[float]] = [
            [-2.0 for _ in range(self._num_timesteps)] for _ in range(self._num_vehicles)
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

        self._lb_tcp: list[float] = [0.0 for _ in range(self._num_timesteps)]
        self._ub_tcp: list[float] = [float(self._num_vehicles) for _ in range(self._num_timesteps)]

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
    @abstractmethod
    def lower_soe_envelope(self) -> list[list[float]]:
        pass

    @property
    def slack(self) -> dict[str, list[list[float]] | float]:
        state_of_energy = self.state_of_energy
        charging_power = self.charging_power
        effective_charging_power = self.effective_charging_power
        max_charging_power = self.max_charging_power
        total_charging_power = self.total_charging_power
        soe_slack = []
        cp_slack = []
        mcp_slack = abs(max_charging_power - max(total_charging_power)) * self._factor_cp
        for vehicle in range(self._num_vehicles):
            soe_slack_v = []
            cp_slack_v = []
            for t_i in range(self._num_timesteps):
                # SOE slack
                soe_1 = state_of_energy[vehicle][t_i + 1]
                soe_2 = state_of_energy[vehicle][t_i] + effective_charging_power[vehicle][t_i] * self._delta_time[t_i]
                soe_2 -= self.input_data.energy_demand[vehicle][t_i]
                soe_slack_v.append(abs((soe_1 - soe_2) / soe_1))

                # CP slack
                assert self._alpha is not None
                cp_1 = effective_charging_power[vehicle][t_i]
                if self._ce_mode == "one":
                    cp_2 = charging_power[vehicle][t_i]
                elif self._ce_mode == "constant":
                    cp_2 = charging_power[vehicle][t_i] * self._alpha
                elif self._ce_mode == "quadratic":
                    cp_2 = charging_power[vehicle][t_i] - (1 - self._alpha) * charging_power[vehicle][t_i] ** 2 / (
                        2 * self.input_data.max_charging_power
                    )
                else:
                    cp_2 = cp_1
                cp_slack_v.append(abs(cp_1 - cp_2) * self._factor_cp)

            # SOE loop slack
            soe_slack_v.append(
                abs(state_of_energy[vehicle][0] - state_of_energy[vehicle][self._num_timesteps]) * self._factor_soe
            )
            soe_slack.append(soe_slack_v)
            cp_slack.append(cp_slack_v)

        return {
            "state_of_energy": soe_slack,
            "charging_power": cp_slack,
            "max_charging_power": mcp_slack,
        }

    @property
    def max_charging_power(self) -> float:
        mcp: float = 0.0
        charging_power: list[list[float]] = self.charging_power
        for t_i in range(self._num_timesteps):
            column = [charging_power[vehicle][t_i] for vehicle in range(self._num_vehicles)]
            mcp = max(mcp, sum(column))
        return mcp

    @property
    def total_charging_power(self) -> list[float]:
        charging_power = self.charging_power
        total_charging_power = []
        for t_i in range(self._num_timesteps):
            total_charging_power.append(sum(charging_power[vehicle][t_i] for vehicle in range(self._num_vehicles)))
        return total_charging_power

    @property
    def energy_cost(self) -> float:
        assert self.input_data.energy_price is not None
        energy_cost_vector = []
        for _ in range(self._num_vehicles):
            energy_cost_vector += [
                self.input_data.energy_price[t_i] * self._delta_time[t_i] for t_i in range(self._num_timesteps)
            ]
        energy_cost_vector = [cp * ec for cp, ec in zip(flatten_lol(self.charging_power), energy_cost_vector)]
        return sum(energy_cost_vector)

    @property
    def power_cost(self) -> float:
        assert self.input_data.grid_tariff is not None
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
            lower_soe_envelope=self.lower_soe_envelope,
        )

    def _set_all_variables(self, **kwargs: Any) -> None:
        for vehicle in range(self._num_vehicles):
            self._charging_power.append(
                [
                    self._set_variable(
                        f"chargingPower_v{vehicle}_{t_i}",
                        self._lb_cp[vehicle][t_i],
                        self._ub_cp[vehicle][t_i],
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
                        f"stateOfEnergy_v{vehicle}_{t_i}",
                        self._lb_soe[vehicle][t_i],
                        self._ub_soe[vehicle][t_i],
                    )
                    for t_i in range(self._num_timesteps + 1)
                ]
            )
            self._total_charging_power = [
                self._set_variable(f"totalChargingPower_{t_i}", self._lb_tcp[t_i], self._ub_tcp[t_i])
                for t_i in range(self._num_timesteps)
            ]
            self._mcp = self._set_variable("maxChargingPower", self._lb_mcp, self._ub_mcp)
            self._lower_soe_envelope.append(
                [
                    self._set_variable(
                        f"lowerSoEEnvelope_v{vehicle}_{t_i}",
                        self._lb_soe[vehicle][t_i],
                        self._ub_soe[vehicle][t_i],
                    )
                    for t_i in range(self._num_timesteps + 1)
                ]
            )

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
    def __init__(
        self,
        input_data: Input,
        name: str = "GurobiOptimizer",
        bidirectional_charging: bool = True,
        initial_soe: list[float | None] | None = None,
        time_limit: int = 5,
        confidence_level: float = 0.0,
        energy_std_dev: float = 0.0,
    ):
        super().__init__(
            input_data,
            name=name,
            bidirectional_charging=bidirectional_charging,
            initial_soe=initial_soe,
            confidence_level=confidence_level,
            energy_std_dev=energy_std_dev,
        )
        with suppress_stdout_stderr():
            self._model: gp.Model = gp.Model(self.name)
            self._model.setParam("LogToConsole", 1)
            self._model.setParam("OutputFlag", 1)
            self._model.setParam("TimeLimit", time_limit)

    @property
    def charging_power(self) -> list[list[float]]:
        return [list(map(lambda v: v.X / self._factor_cp, sublist)) for sublist in self._charging_power]

    @property
    def effective_charging_power(self) -> list[list[float]]:
        return [list(map(lambda v: v.X / self._factor_cp, sublist)) for sublist in self._effective_charging_power]

    @property
    def state_of_energy(self) -> list[list[float]]:
        return [list(map(lambda v: v.X / self._factor_soe, sublist)) for sublist in self._state_of_energy]

    @property
    def lower_soe_envelope(self) -> list[list[float]]:
        return [list(map(lambda v: v.X / self._factor_soe, sublist)) for sublist in self._lower_soe_envelope]

    def _set_variable(self, name: str, lb: float, ub: float) -> gp.Var:
        return self._model.addVar(name=name, vtype=GRB.CONTINUOUS, lb=lb, ub=ub)

    def _set_all_constraints(self, ce_function_type: str = "one", alpha: float = 1.0, **kwargs) -> None:
        self._alpha = alpha
        self._ce_mode = ce_function_type
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
                        <= self._charging_power[vehicle][t_i]
                        - (1 - alpha) * self._charging_power[vehicle][t_i] ** 2 / 2,
                        f"effectiveChargingPower_v{vehicle}_{t_i}",
                    )
                else:
                    raise ValueError(f"Unknown ce_function_type: {ce_function_type}")

                self._model.addConstr(
                    self._state_of_energy[vehicle][t_i + 1]
                    <= self._state_of_energy[vehicle][t_i]
                    + self._effective_charging_power[vehicle][t_i]
                    * self._delta_time[t_i]
                    * (self._factor_soe / self._factor_cp)
                    - self.input_data.energy_demand[vehicle][t_i] * self._factor_soe,
                    f"energyFlow_v{vehicle}_{t_i}",
                )
                energy_demand_high = upper_energy_confidence_bound(
                    self.input_data.energy_demand[vehicle][t_i], self._confidence_level, self._energy_std_dev
                )
                self._model.addConstr(
                    self._lower_soe_envelope[vehicle][t_i + 1]
                    <= self._lower_soe_envelope[vehicle][t_i]
                    + self._effective_charging_power[vehicle][t_i]
                    * self._delta_time[t_i]
                    * (self._factor_soe / self._factor_cp)
                    - energy_demand_high * self._factor_soe,
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

        # total charging power
        for t_i in range(self._num_timesteps):
            self._model.addConstr(
                self._total_charging_power[t_i]
                == gp.quicksum(self._charging_power[vehicle][t_i] for vehicle in range(self._num_vehicles)),
                f"totalChargingPower_{t_i}",
            )

        # max power used
        for t_i in range(self._num_timesteps):
            assert self._mcp is not None
            self._model.addConstr(
                self._mcp >= self._total_charging_power[t_i],
                f"maxChargingPower_{t_i}",
            )

        # initial state of energy
        if self._initial_soe is not None:
            for vehicle, soe in enumerate(self._initial_soe):
                if soe is not None:
                    self._model.addConstr(self._state_of_energy[vehicle][0] == soe * self._factor_soe)

    def _set_objective(self, **kwargs) -> None:
        assert self.input_data.energy_price is not None, "Energy price not provided"
        assert self.input_data.grid_tariff is not None, "Grid tariff not provided"
        assert self._mcp is not None

        self._model.setObjective(
            gp.quicksum(
                self.input_data.energy_price[t_i] * cp * self._factor_ep * self._delta_time[t_i]
                for t_i, cp in enumerate(self._total_charging_power)
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
    def __init__(
        self,
        input_data: Input,
        name: str = "CasadiOptimizer",
        initial_soe: list[float | None] | None = None,
        bidirectional_charging: bool = True,
        confidence_level: float = 0.0,
        energy_std_dev: float = 0.0,
    ):
        super().__init__(
            input_data,
            name=name,
            bidirectional_charging=bidirectional_charging,
            initial_soe=initial_soe,
            confidence_level=confidence_level,
            energy_std_dev=energy_std_dev,
        )

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
                float(self.solution_dict["x"][offset + vehicle * self._num_timesteps + t_i]) / self._factor_cp
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

    @property
    def lower_soe_envelope(self) -> list[list[float]]:
        if self.solution_dict is None:
            raise ValueError("Solution is not computed")
        offset = 3 * self._num_vehicles * self._num_timesteps + self._num_vehicles + self._num_timesteps + 1
        return [
            [
                float(self.solution_dict["x"][offset + vehicle * (self._num_timesteps + 1) + t_i]) / self._factor_soe
                for t_i in range(self._num_timesteps + 1)
            ]
            for vehicle in range(self._num_vehicles)
        ]

    def _set_all_constraints(self, ce_function_type: str = "one", alpha: float = 1.0, **kwargs) -> None:
        self._alpha = alpha
        self._ce_mode = ce_function_type
        # energy flow
        for vehicle in range(self._num_vehicles):
            for t_i in range(self._num_timesteps):
                if ce_function_type == "one":
                    self._constraints.append(
                        self._effective_charging_power[vehicle][t_i] - self._charging_power[vehicle][t_i]
                    )
                    self._constraints_lb.append(0)
                    self._constraints_ub.append(0)
                elif ce_function_type == "constant":
                    self._constraints.append(
                        self._effective_charging_power[vehicle][t_i] - alpha * self._charging_power[vehicle][t_i]
                    )
                    self._constraints_lb.append(0)
                    self._constraints_ub.append(0)
                elif ce_function_type == "quadratic":
                    self._constraints.append(
                        self._effective_charging_power[vehicle][t_i]
                        - (
                            self._charging_power[vehicle][t_i]
                            - ((1 - alpha) / 2) * self._charging_power[vehicle][t_i] ** 2
                        )
                    )
                    self._constraints_lb.append(float("-inf"))
                    self._constraints_ub.append(0)
                else:
                    raise ValueError(f"Unknown ce_function_type: {ce_function_type}")

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
                self._constraints_lb.append(float("-inf"))
                self._constraints_ub.append(0)

                energy_demand_high = upper_energy_confidence_bound(
                    self.input_data.energy_demand[vehicle][t_i], self._confidence_level, self._energy_std_dev
                )
                self._constraints.append(
                    self._lower_soe_envelope[vehicle][t_i + 1]
                    - (
                        self._lower_soe_envelope[vehicle][t_i]
                        + self._effective_charging_power[vehicle][t_i]
                        * self._delta_time[t_i]
                        * (self._factor_soe / self._factor_cp)
                        - energy_demand_high * self._factor_soe
                    )
                )
                self._constraints_lb.append(float("-inf"))
                self._constraints_ub.append(0)

                if not self.input_data.depot_charge[vehicle][t_i]:
                    self._constraints.append(self._charging_power[vehicle][t_i])
                    self._constraints_lb.append(0)
                    self._constraints_ub.append(0)

        # initial state of energy
        if self._initial_soe is not None:
            for vehicle, soe in enumerate(self._initial_soe):
                if soe is not None:
                    self._constraints.append(self._state_of_energy[vehicle][0] - soe * self._factor_soe)
                    self._constraints_lb.append(0)
                    self._constraints_ub.append(0)

        # energy loop
        for vehicle in range(self._num_vehicles):
            self._constraints.append(
                self._state_of_energy[vehicle][self._num_timesteps] - self._state_of_energy[vehicle][0]
            )
            self._constraints_lb.append(0)
            self._constraints_ub.append(float("inf"))

        # total charging power
        for t_i in range(self._num_timesteps):
            column = [self._charging_power[vehicle][t_i] for vehicle in range(self._num_vehicles)]
            self._constraints.append(self._total_charging_power[t_i] - sum(column, ca.MX(0)))
            self._constraints_lb.append(0)
            self._constraints_ub.append(0)

        # max power used
        for t_i in range(self._num_timesteps):
            self._constraints.append(self._mcp - self._total_charging_power[t_i])
            self._constraints_lb.append(0)
            self._constraints_ub.append(float("inf"))

    def _set_objective(self, **kwargs) -> None:
        assert self.input_data.energy_price is not None, "Energy price not provided"
        assert self.input_data.grid_tariff is not None, "Grid tariff not provided"
        assert self._mcp is not None

        energy_cost_vector = [
            self.input_data.energy_price[t_i] * self._delta_time[t_i] * self._factor_ep
            for t_i in range(self._num_timesteps)
        ]
        energy_cost_vector = [cp * ec for cp, ec in zip(self._total_charging_power, energy_cost_vector)]
        self._objective = sum(energy_cost_vector, ca.MX(0)) + self.input_data.grid_tariff * self._mcp * self._factor_ep

    def solve(self) -> Solution | None:
        if not self._built:
            raise ValueError("Optimizer not built")
        nlp = {
            "x": ca.vertcat(
                *flatten_lol(self._charging_power),
                *flatten_lol(self._effective_charging_power),
                *flatten_lol(self._state_of_energy),
                *self._total_charging_power,
                self._mcp,
                *flatten_lol(self._lower_soe_envelope),
            ),
            "f": self._objective,
            "g": ca.vertcat(*self._constraints),
        }
        solver = ca.nlpsol("solver", "ipopt", nlp)

        self.solution_dict = solver(
            lbg=ca.vertcat(*self._constraints_lb),
            ubg=ca.vertcat(*self._constraints_ub),
            lbx=ca.vertcat(*self._lb_cp, *self._lb_ecp, *self._lb_soe, *self._lb_tcp, self._lb_mcp, *self._lb_soe),
            ubx=ca.vertcat(*self._ub_cp, *self._ub_ecp, *self._ub_soe, *self._ub_tcp, self._ub_mcp, *self._ub_soe),
        )

        if not solver.stats()["success"]:
            return None

        return self.solution


def flatten_lol(list_of_lists: list[Any]):
    return [item for sublist in list_of_lists for item in sublist]


def upper_energy_confidence_bound(value: float, confidence_level: float, standard_deviation: float) -> float:
    z = float(norm.ppf((1 - confidence_level) / 2))
    dv = abs(value * standard_deviation * z)
    return value + dv
