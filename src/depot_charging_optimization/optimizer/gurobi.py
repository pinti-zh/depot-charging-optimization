import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.stats import norm

from depot_charging_optimization.config import OptimizerConfig
from depot_charging_optimization.data_models import Input, Solution
from depot_charging_optimization.logging import suppress_stdout_stderr


class GurobiOptimizer:
    def __init__(self, input_data: Input, config: OptimizerConfig=OptimizerConfig()):
        self._input_data: Input = input_data
        self._config: OptimizerConfig = config

        self._built: bool = False

        with suppress_stdout_stderr():
            self._model: gp.Model = gp.Model("gurobi_optimizer")
            self._model.setParam("LogToConsole", 1)
            self._model.setParam("OutputFlag", 1)

        # optimization variables
        self._charging_power: gp.MVar | None = None
        self._effective_charging_power: gp.MVar | None = None
        self._total_charging_power: gp.MVar | None = None
        self._max_charging_power: gp.MVar | None = None
        self._state_of_energy: gp.MVar | None = None
        self._lower_soe_envelope: gp.MVar | None = None

        self._initial_soe: np.ndarray | None = None            # constant once initialized
        self._energy_demand: np.ndarray | None = None          # constant once initialized
        self._realistic_worst_case: np.ndarray | None = None   # constant once initialized
        self._time_delta: np.ndarray | None = None             # constant once initialized
        self._energy_price: np.ndarray | None = None           # constant once initialized
        self._grid_tariff: float | None = None                 # constant once initialized

        # auxiliary constants: scaling factors
        self._factor_cp: float = 1.0 / self._input_data.max_charging_power
        self._factor_soe: float = 1.0 / max(self._input_data.battery_capacity)
        assert self._input_data.energy_price is not None
        self._factor_ep: float = 1.0 / max(self._input_data.energy_price)

    def build(self) -> None:
        self._set_variables()
        self._set_constraints()
        self._set_objective()
        self._built = True

    def solve(self) -> Solution | None:
        assert self._built, "optimizer not built"

        # assert all optimization variables are initialized
        assert isinstance(self._charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._effective_charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._total_charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._max_charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._state_of_energy, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._lower_soe_envelope, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._time_delta, np.ndarray), "uninitialized optimization variable"
        assert isinstance(self._energy_price, np.ndarray), "uninitialized optimization variable"
        assert isinstance(self._grid_tariff, float), "uninitialized optimization variable"

        self._model.optimize()
        if self._model.Status == GRB.OPTIMAL:
            factor = self._factor_cp * self._factor_ep

            energy_cost = (
                    self._total_charging_power.getAttr("X")
                    * self._time_delta[0]
                    * self._energy_price
                    / factor
            ).sum()

            power_cost = self._max_charging_power.getAttr("X").item() * self._grid_tariff / factor

            return Solution(
                input_data=self._input_data,
                total_cost=energy_cost + power_cost,
                energy_cost=energy_cost,
                power_cost=power_cost,
                max_charging_power_used=self._max_charging_power.getAttr("X").item() / self._factor_cp,
                charging_power=[cp / self._factor_cp for cp in self._charging_power.getAttr("X")],
                effective_charging_power=[cp / self._factor_cp for cp in self._effective_charging_power.getAttr("X")],
                state_of_energy=[soe / self._factor_soe for soe in self._state_of_energy.getAttr("X")],
                lower_soe_envelope=[soe / self._factor_soe for soe in self._lower_soe_envelope.getAttr("X")],
            )
        else:
            return None

    def _set_variables(self) -> None:
        bidirectional_charging_mask: np.ndarray
        # Boolean numpy array that masks for each vehicle and timestep if charging is bidirectional
        # Example:
        #   [[1, 1, 1, ..., 1],
        #    [0, 0, 0, ..., 0],
        #           ⋮
        #    [1, 1, 1, ..., 1]]
        assert self._input_data.is_battery is not None
        if self._config.bidirectional_charging:
            bidirectional_charging_mask = np.ones(
                (self._input_data.num_vehicles, self._input_data.num_timesteps),
                dtype=bool
            )
        else:
            bidirectional_charging_mask = np.hstack(
                [np.array(self._input_data.is_battery).reshape(-1, 1)] * self._input_data.num_timesteps
            )

        depot_charge_mask: np.ndarray = np.array(self._input_data.depot_charge, dtype=bool)

        self._charging_power = self._model.addMVar(
            (self._input_data.num_vehicles, self._input_data.num_timesteps),
            lb=-bidirectional_charging_mask.astype(float) * depot_charge_mask.astype(float),
            ub=depot_charge_mask.astype(float),
            name="charging_power",
            vtype=GRB.CONTINUOUS,
        )

        self._effective_charging_power = self._model.addMVar(
            (self._input_data.num_vehicles, self._input_data.num_timesteps),
            lb=-2.0 * bidirectional_charging_mask.astype(float) * depot_charge_mask.astype(float),
            ub=depot_charge_mask.astype(float),
            name="effective_charging_power",
            vtype=GRB.CONTINUOUS,
        )

        self._total_charging_power = self._model.addMVar(
            (self._input_data.num_timesteps, ),
            lb=0.0,
            ub=1.0 * self._input_data.num_vehicles,
            name="total_charging_power",
            vtype=GRB.CONTINUOUS,
        )

        self._max_charging_power = self._model.addMVar(
            (1, ),
            lb=0.0,
            ub=1.0 * self._input_data.num_vehicles,
            name="max_charging_power",
            vtype=GRB.CONTINUOUS,
        )

        self._state_of_energy = self._model.addMVar(
            (self._input_data.num_vehicles, self._input_data.num_timesteps + 1),
            lb=0.0,
            ub=1.0,
            name="state_of_energy",
            vtype=GRB.CONTINUOUS,
        )

        self._lower_soe_envelope = self._model.addMVar(
            (self._input_data.num_vehicles, self._input_data.num_timesteps + 1),
            lb=0.0,
            ub=1.0,
            name="lower_soe_envelope",
            vtype=GRB.CONTINUOUS,
        )

        if self._config.initial_soe is not None:
            self._initial_soe = np.array(self._config.initial_soe, dtype=float) * self._factor_soe

        self._energy_demand = np.array(self._input_data.energy_demand, dtype=float) * self._factor_soe

        z = float(norm.ppf((1 - self._config.confidence_level) / 2))
        self._realistic_worst_case = self._energy_demand + np.abs(
            self._energy_demand * self._config.energy_std_dev * z
        )

        self._time_delta = np.vstack(
            [np.array(
                [t2 - t1 for t1, t2 in zip([0] + self._input_data.time[:-1], self._input_data.time)],
                dtype=float,
            )] * self._input_data.num_vehicles
        )

        assert self._input_data.energy_price is not None
        self._energy_price = np.array(self._input_data.energy_price, dtype=float) * self._factor_ep

        assert self._input_data.grid_tariff is not None
        self._grid_tariff = self._input_data.grid_tariff * self._factor_ep

    def _set_constraints(self) -> None:
        # assert all optimization variables are initialized
        assert isinstance(self._charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._effective_charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._total_charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._max_charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._state_of_energy, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._lower_soe_envelope, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._energy_demand, np.ndarray), "uninitialized optimization variable"
        assert isinstance(self._realistic_worst_case, np.ndarray), "uninitialized optimization variable"
        assert isinstance(self._time_delta, np.ndarray), "uninitialized optimization variable"

        self._model.addConstr(
            self._effective_charging_power
            <= self._config.max_efficiency * (
                    self._charging_power - (self._config.alpha / 2) * self._charging_power ** 2
            ),
            "charging_efficiency",
        )

        self._model.addConstr(
            self._total_charging_power == self._charging_power.sum(axis=0),
            "total_charging_power",
        )

        self._model.addConstr(
            self._max_charging_power >= self._total_charging_power,
            "max_charging_power_relaxation",
        )

        power_to_soe = self._factor_soe / self._factor_cp
        soe_previous = self._state_of_energy[:, :-1]
        soe_next = self._state_of_energy[:, 1:]
        assert isinstance(soe_previous, gp.MVar)
        assert isinstance(soe_next, gp.MVar)
        self._model.addConstr(
            soe_next ==
            soe_previous + self._effective_charging_power * self._time_delta * power_to_soe - self._energy_demand,
            "energy_flow",
        )

        lse_previous = self._lower_soe_envelope[:, :-1]
        lse_next = self._lower_soe_envelope[:, 1:]
        assert isinstance(lse_previous, gp.MVar)
        assert isinstance(lse_next, gp.MVar)
        self._model.addConstr(
            lse_next ==
            lse_previous + self._effective_charging_power * self._time_delta * power_to_soe - self._realistic_worst_case,
            "energy_flow_realistic_worst_case",
            )

        soe_first = self._state_of_energy[:, 0]
        soe_last = self._state_of_energy[:, -1]
        assert isinstance(soe_first, gp.MVar)
        assert isinstance(soe_last, gp.MVar)
        self._model.addConstr(
            soe_first <= soe_last,
            "soe_sustainability",
        )

        if isinstance(self._initial_soe, np.ndarray):
            self._model.addConstr(
                soe_first == self._initial_soe,
                "initial_soe",
            )

        lse_first = self._lower_soe_envelope[:, 0]
        assert isinstance(lse_first, gp.MVar)
        self._model.addConstr(
            lse_first == soe_first,
            "initial_soe_envelope",
        )

    def _set_objective(self) -> None:
        # assert all optimization variables are initialized
        assert isinstance(self._total_charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._max_charging_power, gp.MVar), "uninitialized optimization variable"
        assert isinstance(self._energy_demand, np.ndarray), "uninitialized optimization variable"
        assert isinstance(self._time_delta, np.ndarray), "uninitialized optimization variable"
        assert isinstance(self._energy_price, np.ndarray), "uninitialized optimization variable"
        assert isinstance(self._grid_tariff, float), "uninitialized optimization variable"

        energy_cost = (self._energy_price * self._total_charging_power * self._time_delta[0]).sum()
        power_cost = self._max_charging_power * self._grid_tariff

        self._model.setObjective(energy_cost + power_cost, GRB.MINIMIZE)
