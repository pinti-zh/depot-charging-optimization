import casadi as ca
import numpy as np
from scipy.stats import norm

from depot_charging_optimization.config import OptimizerConfig
from depot_charging_optimization.data_models import Input, Solution


class CasadiOptimizer:
    def __init__(self, input_data: Input, config: OptimizerConfig=OptimizerConfig()):
        self._input_data: Input = input_data
        self._config: OptimizerConfig = config

        self._built: bool = False

        # optimization variables
        self._variables: dict[str, dict[str, ca.SX | np.ndarray | None]] = {
            "charging_power": {"var": None, "lb": None, "ub": None},
            "effective_charging_power": {"var": None, "lb": None, "ub": None},
            "total_charging_power": {"var": None, "lb": None, "ub": None},
            "max_charging_power": {"var": None, "lb": None, "ub": None},
            "state_of_energy": {"var": None, "lb": None, "ub": None},
            "lower_soe_envelope": {"var": None, "lb": None, "ub": None},
        }

        self._initial_soe: ca.DM | None = None                 # constant once initialized
        self._energy_demand: ca.DM | None = None               # constant once initialized
        self._realistic_worst_case: ca.DM | None = None        # constant once initialized
        self._time_delta: ca.DM | None = None                  # constant once initialized
        self._energy_price: ca.DM | None = None                # constant once initialized
        self._grid_tariff: float | None = None                 # constant once initialized

        self._constraints: dict[str, dict[str, ca.SX | np.ndarray]] = {}
        self._objective: ca.SX | None = None

        self._x = None
        self._lbx = None
        self._ubx = None

        self._g = None
        self._lbg = None
        self._ubg = None

        # auxiliary constants: scaling factors
        self._factor_cp: float = 1.0 / self._input_data.max_charging_power
        self._factor_soe: float = 1.0 / max(self._input_data.battery_capacity)
        assert self._input_data.energy_price is not None
        self._factor_ep: float = 1.0 / max(self._input_data.energy_price)

    def build(self) -> None:
        self._set_variables()
        self._set_constraints()
        self._set_objective()

        # stack variables
        x_list = []
        lbx_list = []
        ubx_list = []
        for name, vdict in self._variables.items():
            var = vdict["var"]
            x_list.append(ca.reshape(var, -1, 1))  # flatten to column
            lbx_list.append(ca.reshape(vdict["lb"], -1, 1))
            ubx_list.append(ca.reshape(vdict["ub"], -1, 1))

        self._x = ca.vertcat(*x_list)  # stacked variable vector
        self._lbx = ca.vertcat(*lbx_list)
        self._ubx = ca.vertcat(*ubx_list)

        # stack constraints
        g_list = []
        lbg_list = []
        ubg_list = []

        for name, cdict in self._constraints.items():
            expr = cdict["expr"]
            g_list.append(ca.reshape(expr, -1, 1))
            lbg_list.append(ca.reshape(cdict["lb"], -1, 1))
            ubg_list.append(ca.reshape(cdict["ub"], -1, 1))

        self._g = ca.vertcat(*g_list)
        self._lbg = ca.vertcat(*lbg_list)
        self._ubg = ca.vertcat(*ubg_list)

        self._built = True

    def solve(self) -> Solution | None:
        assert self._built, "optimizer not built"

        # assert all optimization variables are initialized
        assert isinstance(self._x, ca.SX)
        assert isinstance(self._objective, ca.SX)
        assert isinstance(self._g, ca.SX)
        assert isinstance(self._time_delta, ca.DM), "uninitialized optimization variable"
        assert isinstance(self._energy_price, ca.DM), "uninitialized optimization variable"
        assert isinstance(self._grid_tariff, float), "uninitialized optimization variable"

        nlp = {
            "x": self._x,
            "f": self._objective,
            "g": self._g,
        }

        solver = ca.nlpsol("solver", "ipopt", nlp)

        # Solve with bounds
        sol = solver(x0=ca.DM.zeros(self._x.size1(), 1), lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg)

        if not solver.stats()["success"]:
            return None

        solution = sol["x"]  # DM vector

        offset = 0
        var_values = {}

        for name, vdict in self._variables.items():
            var = vdict["var"]
            assert isinstance(var, ca.SX)
            n_var = var.numel()  # total number of elements in this variable

            # slice the solution vector
            var_val = solution[offset : offset + n_var]

            # reshape to original variable shape
            var_val = np.array(ca.reshape(var_val, *var.shape))

            var_values[name] = var_val
            offset += n_var

        factor = self._factor_cp * self._factor_ep

        time_delta = self._time_delta[0, :] # get one-dimensional vector
        assert isinstance(time_delta, ca.DM), "uninitialized optimization variable"

        energy_cost = ca.sum1(
            var_values["total_charging_power"] * time_delta.T * self._energy_price / factor
        )

        power_cost = var_values["max_charging_power"].item() * self._grid_tariff / factor

        return Solution(
            input_data=self._input_data,
            total_cost=energy_cost + power_cost,
            energy_cost=energy_cost,
            power_cost=power_cost,
            max_charging_power_used=var_values["max_charging_power"].item() / self._factor_cp,
            charging_power=[cp / self._factor_cp for cp in var_values["charging_power"]],
            effective_charging_power=[cp / self._factor_cp for cp in var_values["effective_charging_power"]],
            state_of_energy=[soe / self._factor_soe for soe in var_values["state_of_energy"]],
            lower_soe_envelope=[soe / self._factor_soe for soe in var_values["lower_soe_envelope"]],
        )


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

        self._variables["charging_power"]["var"] = ca.SX.sym(
            "charging_power",
            self._input_data.num_vehicles,
            self._input_data.num_timesteps,
        )
        self._variables["charging_power"]["lb"] = -(bidirectional_charging_mask
                                                   * depot_charge_mask).astype(float)
        self._variables["charging_power"]["ub"] = depot_charge_mask.astype(float)

        self._variables["effective_charging_power"]["var"] = ca.SX.sym(
            "effective_charging_power",
            self._input_data.num_vehicles,
            self._input_data.num_timesteps,
        )
        self._variables["effective_charging_power"]["lb"] = (-2 * bidirectional_charging_mask * depot_charge_mask).astype(float)
        self._variables["effective_charging_power"]["ub"] = depot_charge_mask.astype(float)

        self._variables["total_charging_power"]["var"] = ca.SX.sym(
            "total_charging_power",
            self._input_data.num_timesteps,
        )
        self._variables["total_charging_power"]["lb"] = np.zeros(self._input_data.num_timesteps, dtype=float)
        self._variables["total_charging_power"]["ub"] = (self._input_data.num_vehicles
                                                         * np.ones(self._input_data.num_timesteps, dtype=float))

        self._variables["max_charging_power"]["var"] = ca.SX.sym("max_charging_power", 1)
        self._variables["max_charging_power"]["lb"] = np.zeros((1, ), dtype=float)
        self._variables["max_charging_power"]["ub"] = (self._input_data.num_vehicles
                                                       * np.ones((1, ), dtype=float))

        self._variables["state_of_energy"]["var"] = ca.SX.sym(
            "state_of_energy",
            self._input_data.num_vehicles,
            self._input_data.num_timesteps + 1,
        )
        self._variables["state_of_energy"]["lb"] = np.zeros(
            (self._input_data.num_vehicles,
            self._input_data.num_timesteps + 1),
            dtype=float
        )
        self._variables["state_of_energy"]["ub"] = np.ones(
            (self._input_data.num_vehicles,
             self._input_data.num_timesteps + 1),
            dtype=float
        )

        self._variables["lower_soe_envelope"]["var"] = ca.SX.sym(
            "lower_soe_envelope",
            self._input_data.num_vehicles,
            self._input_data.num_timesteps + 1,
            )
        self._variables["lower_soe_envelope"]["lb"] = np.zeros(
            (self._input_data.num_vehicles,
             self._input_data.num_timesteps + 1),
            dtype=float
        )
        self._variables["lower_soe_envelope"]["ub"] = np.ones(
            (self._input_data.num_vehicles,
             self._input_data.num_timesteps + 1),
            dtype=float
        )

        if self._config.initial_soe is not None:
            self._initial_soe = ca.DM(np.array(self._config.initial_soe, dtype=float) * self._factor_soe)

        self._energy_demand = ca.DM(np.array(self._input_data.energy_demand, dtype=float) * self._factor_soe)

        z = float(norm.ppf((1 - self._config.confidence_level) / 2))
        self._realistic_worst_case = self._energy_demand + ca.DM(np.abs(
            self._energy_demand * self._config.energy_std_dev * z
        ))

        self._time_delta = ca.DM(np.vstack(
            [np.array(
                [t2 - t1 for t1, t2 in zip([0] + self._input_data.time[:-1], self._input_data.time)],
                dtype=float,
            )] * self._input_data.num_vehicles
        ))

        assert self._input_data.energy_price is not None
        self._energy_price = ca.DM(np.array(self._input_data.energy_price, dtype=float) * self._factor_ep)

        assert self._input_data.grid_tariff is not None
        self._grid_tariff = self._input_data.grid_tariff * self._factor_ep

    def _set_constraints(self) -> None:
        # assert all optimization variables are initialized
        assert isinstance(self._variables["charging_power"]["var"], ca.SX), "uninitialized optimization variable"
        assert isinstance(self._variables["effective_charging_power"]["var"], ca.SX), "uninitialized optimization variable"
        assert isinstance(self._variables["total_charging_power"]["var"], ca.SX), "uninitialized optimization variable"
        assert isinstance(self._variables["max_charging_power"]["var"], ca.SX), "uninitialized optimization variable"
        assert isinstance(self._variables["state_of_energy"]["var"], ca.SX), "uninitialized optimization variable"
        assert isinstance(self._variables["lower_soe_envelope"]["var"], ca.SX), "uninitialized optimization variable"
        assert isinstance(self._energy_demand, ca.DM), "uninitialized optimization variable"
        assert isinstance(self._realistic_worst_case, ca.DM), "uninitialized optimization variable"
        assert isinstance(self._time_delta, ca.DM), "uninitialized optimization variable"

        self._constraints["charging_efficiency"] = {
            "expr": self._variables["effective_charging_power"]["var"] - self._config.max_efficiency
            * (self._variables["charging_power"]["var"] - (self._config.alpha / 2)
               * self._variables["charging_power"]["var"] ** 2),
            "lb": -float("inf") * np.ones(
                (self._input_data.num_vehicles, self._input_data.num_timesteps), dtype=float
            ),
            "ub": np.zeros((self._input_data.num_vehicles, self._input_data.num_timesteps), dtype=float),
        }

        self._constraints["total_charging_power"] = {
            "expr": ca.reshape(ca.sum1(self._variables["charging_power"]["var"]), -1, 1) - self._variables["total_charging_power"]["var"],
            "lb": -float("inf") * np.ones(self._input_data.num_timesteps, dtype=float),
            "ub": np.zeros(self._input_data.num_timesteps, dtype=float),
        }

        self._constraints["max_charging_power_relaxation"] = {
            "expr": self._variables["total_charging_power"]["var"] - self._variables["max_charging_power"]["var"],
            "lb": -float("inf") * np.ones(self._input_data.num_timesteps, dtype=float),
            "ub": np.zeros(self._input_data.num_timesteps, dtype=float),
        }

        power_to_soe = self._factor_soe / self._factor_cp
        self._constraints["energy_flow"] = {
            "expr": self._variables["state_of_energy"]["var"][:, 1:]
            - self._variables["state_of_energy"]["var"][:, :-1]
            - self._variables["effective_charging_power"]["var"] * self._time_delta * power_to_soe
            + self._energy_demand,
            "lb": np.zeros((self._input_data.num_vehicles, self._input_data.num_timesteps), dtype=float),
            "ub": np.zeros((self._input_data.num_vehicles, self._input_data.num_timesteps), dtype=float),
        }

        self._constraints["energy_flow_realistic_worst_case"] = {
            "expr": self._variables["lower_soe_envelope"]["var"][:, 1:]
            - self._variables["lower_soe_envelope"]["var"][:, :-1]
            - self._variables["effective_charging_power"]["var"] * self._time_delta * power_to_soe
            + self._realistic_worst_case,
            "lb": np.zeros((self._input_data.num_vehicles, self._input_data.num_timesteps), dtype=float),
            "ub": np.zeros((self._input_data.num_vehicles, self._input_data.num_timesteps), dtype=float),
        }

        self._constraints["soe_sustainability"] = {
            "expr": self._variables["state_of_energy"]["var"][:, 0] - self._variables["state_of_energy"]["var"][:, -1],
            "lb": -float("inf") * np.ones(self._input_data.num_vehicles, dtype=float),
            "ub": np.zeros(self._input_data.num_vehicles, dtype=float),
        }

        if isinstance(self._initial_soe, ca.DM):
            for i in range(self._input_data.num_vehicles):
                if not np.isnan(self._initial_soe[i]):
                    self._constraints[f"initial_soe_{i}"] = {
                        "expr": self._variables["state_of_energy"]["var"][i, 0] - self._initial_soe[i],
                        "lb": np.zeros(1, dtype=float),
                        "ub": np.zeros(1, dtype=float),
                    }

        self._constraints["initial_soe_envelope"] = {
            "expr": self._variables["state_of_energy"]["var"][:, 0]
            - self._variables["lower_soe_envelope"]["var"][:, 0],
            "lb": np.zeros(self._input_data.num_vehicles, dtype=float),
            "ub": np.zeros(self._input_data.num_vehicles, dtype=float),
        }

    def _set_objective(self) -> None:
        # assert all optimization variables are initialized
        assert isinstance(self._variables["total_charging_power"]["var"], ca.SX), "uninitialized optimization variable"
        assert isinstance(self._variables["max_charging_power"]["var"], ca.SX), "uninitialized optimization variable"
        assert isinstance(self._energy_demand, ca.DM), "uninitialized optimization variable"
        assert isinstance(self._time_delta, ca.DM), "uninitialized optimization variable"
        assert isinstance(self._energy_price, ca.DM), "uninitialized optimization variable"
        assert isinstance(self._grid_tariff, float), "uninitialized optimization variable"

        time_delta = self._time_delta[0, :] # get one-dimensional vector
        assert isinstance(time_delta, ca.DM), "uninitialized optimization variable"

        energy_cost = ca.sum1(
                self._energy_price * self._variables["total_charging_power"]["var"] * time_delta.T
        )
        power_cost = self._variables["max_charging_power"]["var"] * self._grid_tariff

        self._objective = energy_cost + power_cost
