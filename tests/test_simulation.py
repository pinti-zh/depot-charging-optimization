import pytest

from depot_charging_optimization.config import EnvironmentConfig
from depot_charging_optimization.environment import Environment, State


class TestState:
    def test_valid_state_creation(self, valid_initial_state):
        state_instance = valid_initial_state.model_copy()
        assert state_instance.num_vehicles == len(state_instance.state_of_energy)
        assert state_instance.num_vehicles == len(state_instance.in_depot)
        assert state_instance.num_vehicles == len(state_instance.battery_capacity)

    @pytest.mark.parametrize(
        "field",
        ["state_of_energy", "in_depot", "battery_capacity"],
    )
    def test_invalid_state_creation_long_list(self, valid_initial_state_data, field):
        data = valid_initial_state_data.copy()
        data[field] *= 2
        with pytest.raises(ValueError):
            State(**data)

    def test_invalid_state_creation_wrong_num_vehicles(self, valid_initial_state_data):
        data = valid_initial_state_data.copy()
        data["num_vehicles"] += 1
        with pytest.raises(ValueError):
            State(**data)

    @pytest.mark.parametrize(
        "energy_delta",
        [-1.0, -5.0, -10.0, -20.0],
    )
    def test_state_evolution(self, valid_initial_state, energy_delta):
        state_instance = valid_initial_state.model_copy()
        state_instance.update_soe([energy_delta, energy_delta])
        assert state_instance.state_of_energy == [soe + energy_delta for soe in state_instance.battery_capacity]

    def test_state_evolution_over_capacity(self, valid_initial_state):
        state_instance = valid_initial_state.model_copy()
        state_instance.update_soe([1.0, 1.0])
        assert state_instance.state_of_energy == state_instance.battery_capacity

    def test_longer_state_evolution(self, valid_initial_state):
        state_instance = valid_initial_state.model_copy()
        for energy_delta in [-1.0, 5.0, -5.0, -2.0, 3.0, -1.0]:
            state_instance.update_soe([energy_delta, energy_delta])
        assert state_instance.state_of_energy == [5.0, 5.0]

    @pytest.mark.parametrize(
        "energy_delta",
        [1.0, -1.0, -10.0],
    )
    def test_state_validation_ok(self, valid_initial_state, energy_delta):
        state_instance = valid_initial_state.model_copy()
        state_instance.update_soe([energy_delta, energy_delta])
        assert state_instance.is_valid()

    @pytest.mark.parametrize(
        "energy_delta",
        [-11.0, -20.0],
    )
    def test_state_validation_not_ok(self, valid_initial_state, energy_delta):
        state_instance = valid_initial_state.model_copy()
        state_instance.update_soe([energy_delta, energy_delta])
        assert not state_instance.is_valid()


class TestEnvironment:
    def test_state_evolution_zero_policy(self, valid_input):
        input_instance = valid_input.model_copy()
        env = Environment(input_instance, EnvironmentConfig())

        env.reset(input_instance.battery_capacity)

        policy = [0.0, 0.0, 0.0]
        for t in range(input_instance.num_timesteps):
            env.step(policy)

        assert env.state.state_of_energy == [10.0, -10.0, -10.0]

    def test_state_evolution_constant_policy(self, valid_input):
        input_instance = valid_input.model_copy()
        env = Environment(input_instance, EnvironmentConfig(charger_max_charging_power=1.0))

        env.reset(input_instance.battery_capacity)

        policy = [1e-2] * 3
        for t in range(input_instance.num_timesteps):
            env.step(policy)

        assert env.state.state_of_energy == [25.0, 5.0, -4.0]

    def test_simulation_solution_constant_policy(self, valid_input):
        input_instance = valid_input.model_copy()
        input_instance = input_instance.add_energy_price([300, 600, 1200, 1500, 3000], [1.0] * 5)
        input_instance = input_instance.add_grid_tariff(5.0)
        env = Environment(input_instance, EnvironmentConfig(charger_max_charging_power=1.0))

        env.reset(input_instance.battery_capacity)

        policy = [1e-2] * 3
        for t in range(input_instance.num_timesteps):
            env.step(policy)

        solution = env.get_solution()
        assert solution.total_cost == 36 * 1.0 + 5.0 * 2e-2
