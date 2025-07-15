from abc import ABC, abstractmethod
from typing import Any, Optional

import casadi as ca
import gurobipy as gp
from gurobipy import GRB

from depot_charging_optimization.data_models import Input, Solution


class Optimizer(ABC):
    @abstractmethod
    def set_variables(self, *args: Any) -> None:
        pass

    @abstractmethod
    def set_constraints(self, *args: Any) -> None:
        pass

    @abstractmethod
    def set_objective(self, *args: Any) -> None:
        pass

    def solve(self, *args: Any) -> Optional[Solution]:
        pass


class GurobiOptimizer(Optimizer):
    def __init__(self, input_data: Input, name: str = "GurobiOptimizer", greedy: bool = False):
        self.input_data: Input = input_data
        self.name: str = name
        self.model: gp.Model = gp.Model(self.name)
        self.greedy = greedy

        self.charging_power: list[list[gp.Var]] = []
        self.state_of_energy: list[list[gp.Var]] = []
        self.charging_efficiency: list[list[gp.Var]] = []

        self.mcp: Optional[gp.Var] = None

        self.objective_value: Optional[float] = None
        self.vars_initialized: bool = False
        self.constraints_initialized: bool = False
        self.objective_initialized: bool = False

        self.delta_time: list[int] = [t2 - t1 for t1, t2 in zip([0] + self.input_data.time[:-1], self.input_data.time)]
        self.max_soe_increase: list[list[float]] = []
        self.soe_decrease: list[list[float]] = []
        self.storable_energy: list[float] = []
        for vehicle in range(self.input_data.num_vehicles):
            storable_energy_v = (
                self.input_data.soe_ub[vehicle] - self.input_data.soe_lb[vehicle]
            ) * self.input_data.battery_capacity[vehicle]
            self.storable_energy.append(storable_energy_v)

            self.max_soe_increase.append(
                [self.input_data.max_charging_power * dt / storable_energy_v for dt in self.delta_time]
            )

            self.soe_decrease.append(
                [
                    self.input_data.energy_demand[vehicle][i] / storable_energy_v
                    for i in range(self.input_data.num_timesteps)
                ]
            )

        self.opt_input: Input = input_data

    def set_variables(self) -> None:

        # decision variables
        for vehicle in range(self.input_data.num_vehicles):
            self.charging_power.append(
                [
                    self.model.addVar(name=f"chargingPower_v{vehicle}_{t_i+1}", vtype=GRB.CONTINUOUS, lb=0, ub=1)
                    for t_i in range(self.input_data.num_timesteps)
                ]
            )
            self.charging_efficiency.append(
                [
                    self.model.addVar(name=f"chargingEfficiency_v{vehicle}_{t_i+1}", vtype=GRB.CONTINUOUS, lb=0, ub=1)
                    for t_i in range(self.input_data.num_timesteps)
                ]
            )
            self.state_of_energy.append(
                [
                    self.model.addVar(
                        name=f"stateOfEnergy_v{vehicle}_{t_i}",
                        vtype=GRB.CONTINUOUS,
                        lb=0,
                        ub=1,
                    )
                    for t_i in range(self.input_data.num_timesteps + 1)
                ]
            )

        # aux variables
        self.mcp = self.model.addVar(name="maxChargingPower", vtype=GRB.CONTINUOUS, lb=0)

        self.vars_initialized = True

    def set_constraints(self, ce_function_type: str = "one", alpha: float = 1.0, cp_throttle: float = 1.0) -> None:
        if not self.vars_initialized:
            raise ValueError("Variables must be initialized before constraints")

        # energy flow
        for vehicle in range(self.input_data.num_vehicles):
            for i, values in enumerate(
                zip(
                    self.input_data.depot_charge[vehicle],
                    self.soe_decrease[vehicle],
                    self.max_soe_increase[vehicle],
                    self.charging_power[vehicle],
                    self.charging_efficiency[vehicle],
                )
            ):
                dc, soe_decr, max_soe_incr, cp, ce = values
                if dc:
                    if ce_function_type == "one":
                        self.model.addConstr(ce == 1.0, f"chargingEfficiency_v{vehicle}_{i}")
                    elif ce_function_type == "constant":
                        self.model.addConstr(ce == alpha, f"chargingEfficiency_v{vehicle}_{i}")
                    elif ce_function_type == "quadratic":
                        self.model.addConstr(
                            ce == 1 - (1 - alpha) * cp / 2,
                            f"chargingEfficiency_v{vehicle}_{i}",
                        )
                    else:
                        raise ValueError(f"Unknown ce_function_type: {ce_function_type}")

                    self.model.addConstr(
                        self.state_of_energy[vehicle][i + 1]
                        == self.state_of_energy[vehicle][i] + cp * ce * max_soe_incr,
                        f"charging_v{vehicle}_{i}",
                    )

                else:
                    self.model.addConstr(
                        self.state_of_energy[vehicle][i + 1] == self.state_of_energy[vehicle][i] - soe_decr,
                        f"energyDemand_v{vehicle}_{i}",
                    )
                    self.model.addConstr(
                        cp == 0,
                        f"noChargingPower_v{vehicle}_{i}",
                    )
                    self.model.addConstr(
                        ce == 0,
                        f"noChargingEfficiency_v{vehicle}_{i}",
                    )

        # energy loop
        for vehicle in range(self.opt_input.num_vehicles):
            self.model.addConstr(
                self.state_of_energy[vehicle][0] <= self.state_of_energy[vehicle][self.input_data.num_timesteps],
                f"energyLoop_v{vehicle}",
            )

        # max power used
        for index in range(self.input_data.num_timesteps):
            self.model.addConstr(
                self.mcp
                >= gp.quicksum(self.charging_power[vehicle][index] for vehicle in range(self.input_data.num_vehicles)),
                f"maxChargingPower_{index}",
            )

        self.model.addConstr(
            self.mcp <= cp_throttle * self.input_data.num_vehicles,
            "maxChargingPowerThrottle",
        )

        self.constraints_initialized = True

    def set_objective(self) -> None:
        if not self.constraints_initialized:
            raise ValueError("Constraints must be initialized before objective")

        if self.greedy:
            self.model.setObjective(
                gp.quicksum(
                    gp.quicksum(self.state_of_energy[vehicle]) for vehicle in range(self.input_data.num_vehicles)
                ),
                GRB.MAXIMIZE,
            )
        else:
            self.model.setObjective(
                gp.quicksum(
                    gp.quicksum(
                        self.input_data.energy_price[i] * cp * self.input_data.max_charging_power * self.delta_time[i]
                        for i, cp in enumerate(self.charging_power[vehicle])
                    )
                    for vehicle in range(self.opt_input.num_vehicles)
                )
                + self.mcp * self.opt_input.grid_tariff * self.input_data.max_charging_power,
                GRB.MINIMIZE,
            )

        self.objective_initialized = True

    def solve(self) -> Optional[Solution]:
        if not self.objective_initialized:
            raise ValueError("Objective must be initialized before optimization")
        self.model.optimize()
        try:
            self.objective_value = self.model.ObjVal
            if self.model.Status != GRB.OPTIMAL:
                gap = 0
            else:
                obj_bound = self.model.ObjBound
                gap = abs(self.objective_value - obj_bound) / (abs(self.objective_value) + 1e-10)
            charging_power = self.get_charging_power()
            energy_cost = sum(
                sum(
                    cp * self.delta_time[i] * self.input_data.energy_price[i]
                    for i, cp in enumerate(charging_power[vehicle])
                )
                for vehicle in range(self.input_data.num_vehicles)
            )
            power_cost = self.get_max_charging_power_used() * self.opt_input.grid_tariff
        except AttributeError:
            return None
        return Solution(
            input_data=self.input_data,
            total_cost=energy_cost + power_cost,
            energy_cost=energy_cost,
            power_cost=power_cost,
            gap=gap,
            max_charging_power_used=self.get_max_charging_power_used(),
            charging_power=self.get_charging_power(),
            charging_efficiency=self.get_charging_efficiency(),
            state_of_energy=self.get_state_of_energy(),
        )

    def get_charging_power(self) -> list[list[float]]:
        return [
            list(map(lambda cp: cp.X * self.input_data.max_charging_power, vehicle_cp))
            for vehicle_cp in self.charging_power
        ]

    def get_charging_efficiency(self) -> list[list[float]]:
        return [list(map(lambda ce: ce.X, vehicle_ce)) for vehicle_ce in self.charging_efficiency]

    def get_state_of_energy(self) -> list[list[float]]:
        return [
            list(
                map(
                    lambda soe: soe.X * self.storable_energy[vehicle]
                    + self.input_data.soe_lb[vehicle] * self.input_data.battery_capacity[vehicle],
                    self.state_of_energy[vehicle],
                )
            )
            for vehicle in range(self.input_data.num_vehicles)
        ]

    def get_max_charging_power_used(self) -> float:
        charging_power = self.get_charging_power()
        return max(
            sum(charging_power[vehicle][index] for vehicle in range(self.input_data.num_vehicles))
            for index in range(self.input_data.num_timesteps)
        )


class CasadiOptimizer(Optimizer):
    def __init__(self, input_data: Input, name: str = "CasadiOptimizer", greedy: bool = False):
        self.input_data: Input = input_data
        self.name: str = name
        self.greedy: bool = greedy

        self.delta_time: list[int] = [t2 - t1 for t1, t2 in zip([0] + self.input_data.time[:-1], self.input_data.time)]

        self.charging_power: list[ca.MX.sym] = []
        for vehicle in range(self.input_data.num_vehicles):
            for t_i in range(self.input_data.num_timesteps):
                self.charging_power.append(ca.MX.sym(f"chargingPower_v{vehicle}_{t_i+1}"))

        self.charging_efficiency: list[ca.MX.sym] = []
        for vehicle in range(self.input_data.num_vehicles):
            for t_i in range(self.input_data.num_timesteps):
                self.charging_efficiency.append(ca.MX.sym(f"chargingEfficiency_v{vehicle}_{t_i+1}"))

        self.state_of_energy: list[ca.MX.sym] = []
        for vehicle in range(self.input_data.num_vehicles):
            for t_i in range(self.input_data.num_timesteps + 1):
                self.state_of_energy.append(ca.MX.sym(f"stateOfEnergy_v{vehicle}_{t_i}"))

        self.max_charging_power: ca.MX.sym = ca.MX.sym("maxChargingPower")

        self.lb_cp: list[float] = [0] * self.input_data.num_vehicles * self.input_data.num_timesteps
        self.ub_cp: list[float] = (
            [self.input_data.max_charging_power] * self.input_data.num_vehicles * self.input_data.num_timesteps
        )

        self.lb_ce: list[float] = [0] * self.input_data.num_vehicles * self.input_data.num_timesteps
        self.ub_ce: list[float] = [1] * self.input_data.num_vehicles * self.input_data.num_timesteps

        self.soe_lb: list[float] = []
        self.soe_ub: list[float] = []
        for vehicle in range(self.input_data.num_vehicles):
            self.soe_lb += [0.2 * self.input_data.battery_capacity[vehicle]] * (self.input_data.num_timesteps + 1)
            self.soe_ub += [0.8 * self.input_data.battery_capacity[vehicle]] * (self.input_data.num_timesteps + 1)

        self.constraints: list[ca.casadi.MX] = []
        self.constraints_lb: list[float] = []
        self.constraints_ub: list[float] = []
        self.objective: Optional[ca.casadi.MX] = None

    def set_variables(self) -> None:
        pass

    def set_constraints(self, ce_function_type: str = "one", alpha: float = 1.0, cp_throttle: float = 1.0) -> None:
        # energy flow
        for vehicle in range(self.input_data.num_vehicles):
            for t_i in range(self.input_data.num_timesteps):
                if self.input_data.depot_charge[vehicle][t_i]:
                    if ce_function_type == "one":
                        self.constraints.append(
                            self.charging_efficiency[vehicle * self.input_data.num_timesteps + t_i] - 1.0
                        )
                    elif ce_function_type == "constant":
                        self.constraints.append(
                            self.charging_efficiency[vehicle * self.input_data.num_timesteps + t_i] - alpha
                        )
                    elif ce_function_type == "quadratic":
                        self.constraints.append(
                            self.charging_efficiency[vehicle * self.input_data.num_timesteps + t_i]
                            - (
                                1
                                - (1 - alpha)
                                * self.charging_power[vehicle * self.input_data.num_timesteps + t_i]
                                / (2 * self.input_data.max_charging_power**2)
                            )
                        )
                    else:
                        raise ValueError(f"Unknown ce_function_type: {ce_function_type}")

                    self.constraints_lb.append(0)
                    self.constraints_ub.append(0)

                    self.constraints.append(
                        self.state_of_energy[vehicle * (self.input_data.num_timesteps + 1) + t_i + 1]
                        - (
                            self.state_of_energy[vehicle * (self.input_data.num_timesteps + 1) + t_i]
                            + self.charging_power[vehicle * self.input_data.num_timesteps + t_i]
                            * self.charging_efficiency[vehicle * self.input_data.num_timesteps + t_i]
                            * self.delta_time[t_i]
                        )
                    )

                    self.constraints_lb.append(0)
                    self.constraints_ub.append(0)
                else:
                    self.constraints.append(
                        self.state_of_energy[vehicle * (self.input_data.num_timesteps + 1) + t_i + 1]
                        - (
                            self.state_of_energy[vehicle * (self.input_data.num_timesteps + 1) + t_i]
                            - self.input_data.energy_demand[vehicle][t_i]
                        )
                    )
                    self.constraints.append(self.charging_power[vehicle * self.input_data.num_timesteps + t_i])
                    self.constraints.append(self.charging_efficiency[vehicle * self.input_data.num_timesteps + t_i])

                    self.constraints_lb.append(0)
                    self.constraints_ub.append(0)
                    self.constraints_lb.append(0)
                    self.constraints_ub.append(0)
                    self.constraints_lb.append(0)
                    self.constraints_ub.append(0)

        # energy loop
        for vehicle in range(self.input_data.num_vehicles):
            self.constraints.append(
                self.state_of_energy[vehicle * (self.input_data.num_timesteps + 1) + self.input_data.num_timesteps]
                - self.state_of_energy[vehicle * (self.input_data.num_timesteps + 1)]
            )
            self.constraints_lb.append(0)
            self.constraints_ub.append(float("inf"))

        # max power used
        for index in range(self.input_data.num_timesteps):
            column = [
                self.charging_power[vehicle * self.input_data.num_timesteps + index]
                for vehicle in range(self.input_data.num_vehicles)
            ]
            self.constraints.append(self.max_charging_power - sum(column, ca.MX(0)))
            self.constraints_lb.append(0)
            self.constraints_ub.append(float("inf"))

    def set_objective(self) -> None:
        energy_cost_vector = []
        for _ in range(self.input_data.num_vehicles):
            energy_cost_vector += [
                self.input_data.energy_price[t_i] * self.delta_time[t_i]
                for t_i in range(self.input_data.num_timesteps)
            ]
        energy_cost_vector = [cp * ec for cp, ec in zip(self.charging_power, energy_cost_vector)]
        self.objective = sum(energy_cost_vector, ca.MX(0)) + self.input_data.grid_tariff * self.max_charging_power

    def solve(self) -> None:
        nlp = {
            "x": ca.vertcat(
                *self.charging_power,
                *self.charging_efficiency,
                *self.state_of_energy,
                self.max_charging_power,
            ),
            "f": self.objective,
            "g": ca.vertcat(*self.constraints),
        }
        solver = ca.nlpsol("solver", "ipopt", nlp)

        solution = solver(
            lbg=ca.vertcat(*self.constraints_lb),
            ubg=ca.vertcat(*self.constraints_ub),
            lbx=ca.vertcat(*self.lb_cp, *self.lb_ce, *self.soe_lb, 0),
            ubx=ca.vertcat(*self.ub_cp, *self.ub_ce, *self.soe_ub, float("inf")),
        )

        charging_power = [
            [
                float(solution["x"][vehicle * self.input_data.num_timesteps + t_i])
                for t_i in range(self.input_data.num_timesteps)
            ]
            for vehicle in range(self.input_data.num_vehicles)
        ]
        offset = sum(map(len, charging_power))
        charging_efficiency = [
            [
                float(solution["x"][offset + vehicle * self.input_data.num_timesteps + t_i])
                for t_i in range(self.input_data.num_timesteps)
            ]
            for vehicle in range(self.input_data.num_vehicles)
        ]

        offset += sum(map(len, charging_efficiency))

        state_of_energy = [
            [
                float(solution["x"][offset + vehicle * (self.input_data.num_timesteps + 1) + t_i])
                for t_i in range(self.input_data.num_timesteps + 1)
            ]
            for vehicle in range(self.input_data.num_vehicles)
        ]

        energy_cost_vector = []
        for _ in range(self.input_data.num_vehicles):
            energy_cost_vector += [
                self.input_data.energy_price[t_i] * self.delta_time[t_i]
                for t_i in range(self.input_data.num_timesteps)
            ]
        energy_cost_vector = [
            cp * ec for cp, ec in zip([item for sublist in charging_power for item in sublist], energy_cost_vector)
        ]
        energy_cost = sum(energy_cost_vector, ca.MX(0))
        max_charging_power = float(solution["x"][-1])
        power_cost = max_charging_power * self.input_data.grid_tariff

        return Solution(
            input_data=self.input_data,
            total_cost=energy_cost + power_cost,
            energy_cost=energy_cost,
            power_cost=power_cost,
            gap=0,
            max_charging_power_used=max_charging_power,
            charging_power=charging_power,
            charging_efficiency=charging_efficiency,
            state_of_energy=state_of_energy,
        )
