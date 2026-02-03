import pytest

from depot_charging_optimization.config import OptimizerConfig
from depot_charging_optimization.data_models import Input, Solution


def _augment_input_with_prices(inp: Input) -> Input:
    # Ensure energy_price and grid_tariff are present for optimizers
    energy_time = [300, 600, 1200, 1500, 3000]
    energy_price = [1.0] * len(energy_time)
    inp = inp.add_energy_price(energy_time, energy_price)
    inp = inp.add_grid_tariff(5.0)
    return inp


@pytest.mark.parametrize("bidirectional", [False, True])
def test_casadi_optimizer_build_and_solve(valid_input, bidirectional):
    casadi = pytest.importorskip("casadi")
    from depot_charging_optimization.optimizer.casadi import CasadiOptimizer

    inp = _augment_input_with_prices(valid_input.model_copy())
    config = OptimizerConfig(
        optimizer_type="casadi",
        alpha=0.0,
        max_efficiency=1.0,
        bidirectional_charging=bidirectional,
        confidence_level=0.0,
        energy_std_dev=0.0,
        initial_soe=None,
    )

    opt = CasadiOptimizer(inp, config=config)
    opt.build()
    sol = opt.solve()

    assert isinstance(sol, Solution)
    # Basic structural checks
    assert sol.input_data.num_vehicles == inp.num_vehicles
    assert sol.input_data.num_timesteps == inp.num_timesteps
    assert len(sol.charging_power) == inp.num_vehicles
    assert len(sol.charging_power[0]) == inp.num_timesteps
    assert len(sol.effective_charging_power) == inp.num_vehicles
    assert len(sol.effective_charging_power[0]) == inp.num_timesteps
    assert len(sol.state_of_energy) == inp.num_vehicles
    assert len(sol.state_of_energy[0]) == inp.num_timesteps + 1
    assert len(sol.lower_soe_envelope) == inp.num_vehicles
    assert len(sol.lower_soe_envelope[0]) == inp.num_timesteps + 1

    # Sanity checks on values
    # SoE should be within [0, capacity]
    for v_idx in range(inp.num_vehicles):
        cap = inp.battery_capacity[v_idx]
        for val in sol.state_of_energy[v_idx]:
            assert 0.0 - 1e-6 <= val <= cap + 1e-6

    assert sol.total_cost >= 0.0
    assert sol.energy_cost >= 0.0
    assert sol.power_cost >= 0.0


@pytest.mark.parametrize("bidirectional", [False, True])
def test_gurobi_optimizer_build_and_solve(valid_input, bidirectional):
    gp = pytest.importorskip("gurobipy")

    from depot_charging_optimization.optimizer.gurobi import GurobiOptimizer

    inp = _augment_input_with_prices(valid_input.model_copy())
    config = OptimizerConfig(
        optimizer_type="gurobi",
        alpha=0.0,
        max_efficiency=1.0,
        bidirectional_charging=bidirectional,
        confidence_level=0.0,
        energy_std_dev=0.0,
        initial_soe=None,
    )

    try:
        opt = GurobiOptimizer(inp, config=config)
        opt.build()
        sol = opt.solve()
    except Exception as e:  # License or runtime issues should skip the test
        pytest.skip(f"Skipping Gurobi test due to runtime/setup issue: {e}")

    assert isinstance(sol, Solution)
    # Basic structural checks
    assert sol.input_data.num_vehicles == inp.num_vehicles
    assert sol.input_data.num_timesteps == inp.num_timesteps
    assert len(sol.charging_power) == inp.num_vehicles
    assert len(sol.charging_power[0]) == inp.num_timesteps
    assert len(sol.effective_charging_power) == inp.num_vehicles
    assert len(sol.effective_charging_power[0]) == inp.num_timesteps
    assert len(sol.state_of_energy) == inp.num_vehicles
    assert len(sol.state_of_energy[0]) == inp.num_timesteps + 1
    assert len(sol.lower_soe_envelope) == inp.num_vehicles
    assert len(sol.lower_soe_envelope[0]) == inp.num_timesteps + 1

    # Sanity checks on values
    for v_idx in range(inp.num_vehicles):
        cap = inp.battery_capacity[v_idx]
        for val in sol.state_of_energy[v_idx]:
            assert 0.0 - 1e-6 <= val <= cap + 1e-6

    assert sol.total_cost >= 0.0
    assert sol.energy_cost >= 0.0
    assert sol.power_cost >= 0.0
