from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from depot_charging_optimization.config import OptimizerConfig
from depot_charging_optimization.optimizer.spec import QuadraticExpression, VariableSpec, ConstraintSpec
from depot_charging_optimization.data_models import Input


OptVariable = TypeVar("OptVariable")


class Optimizer(ABC, Generic[OptVariable]):
    def __init__(self, input_data: Input, config: OptimizerConfig):
        self._input_data: Input = input_data
        self._config: OptimizerConfig = config

        self._built: bool = False

        # Optimization variables
        self._charging_power: list[list[OptVariable]] | None = None
        self._state_of_energy: list[list[OptVariable]] | None = None
        self._effective_charging_power: list[list[OptVariable]] | None = None
        self._total_charging_power: list[OptVariable] | None = None
        self._mcp: OptVariable | None = None

        # envelope for stochastic robustness
        self._lower_soe_envelope: list[list[OptVariable]] | None = None

        # Scaling factors
        self._factor_cp: float = 1.0 / input_data.max_charging_power
        self._factor_soe: float = 1.0 / max(input_data.battery_capacity)
        assert input_data.energy_price is not None, "Input data must contain energy prices"
        self._factor_ep: float = 1.0 / max(input_data.energy_price)

        # Variable bounds
        assert self.input_data.is_battery is not None
        vehicle_lower_bounds = [
            -1.0 if (is_battery or self.config.bidirectional_charging) else 0.0
            for is_battery in self.input_data.is_battery
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
                self.input_data.soe_lb[vehicle] * self.input_data.battery_capacity[vehicle] * self._factor_soe
                for _ in range(self._num_timesteps + 1)
            ]
            for vehicle in range(self._num_vehicles)
        ]
        self._ub_soe: list[list[float]] = [
            [
                self.input_data.soe_ub[vehicle] * self.input_data.battery_capacity[vehicle] * self._factor_soe
                for _ in range(self._num_timesteps + 1)
            ]
            for vehicle in range(self._num_vehicles)
        ]

        self._lb_mcp: float = 0.0
        self._ub_mcp: float = float(self._num_vehicles)

        self._lb_tcp: list[float] = [0.0 for _ in range(self._num_timesteps)]
        self._ub_tcp: list[float] = [float(self._num_vehicles) for _ in range(self._num_timesteps)]

    def build(self):
        pass

    @abstractmethod
    def _set_variable(self):
        pass

    @abstractmethod
    def _set_constraint(self):
        pass




