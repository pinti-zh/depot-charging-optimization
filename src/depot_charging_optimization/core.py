from typing import Optional

import gurobipy as gp
from gurobipy import GRB

from depot_charging_optimization.data_models import Input, Solution


class OptimizationModel:
    def __init__(self, input_data: Input, name: str = "OptimizationModel", greedy: bool = False):
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

    def set_variables(self):

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

    def set_constraints(self, ce_function_type: str = "one", alpha: float = 1.0, cp_throttle: float = 1.0):
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

    def set_objective(self):
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
            obj_bound = self.model.ObjBound
            gap = abs(self.objective_value - obj_bound) / (abs(self.objective_value) + 1e-10) * 100
            charging_power = self.get_charging_power()
            energy_cost = sum(
                sum(
                    cp * self.delta_time[i] * self.input_data.energy_price[i]
                    for i, cp in enumerate(charging_power[vehicle])
                )
                for vehicle in range(self.input_data.num_vehicles)
            )
            power_cost = self.get_max_charging_power_used() * self.opt_input.grid_tariff
        except ValueError:
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
