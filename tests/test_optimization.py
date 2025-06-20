import polars as pl

from optimization.optimization import (
    GreedyOptimizationModel,
    OptimizationInput,
    OptimizationModel,
)

EPS = 1e-6


class TestOptimizationSingleVehicle:
    def test_optimization_low_tariff(self):
        data = pl.DataFrame(
            {
                "time": [5, 10, 15],
                "energy_demand": [0.0, 30.0, 0.0],
                "depot_charge": [True, False, True],
                "battery_capacity": [50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 6.0],
            }
        )
        energy_price = pl.DataFrame({"time": [5, 10, 15], "energy_price": [1.0, 0.0, 2.0]})
        grid_tariff = 4.0

        opt_input = OptimizationInput([data], energy_price, grid_tariff)
        assert opt_input.is_feasible()[0]
        opt_model = OptimizationModel(opt_input)
        opt_model.set_variables()
        opt_model.set_constraints()
        opt_model.set_objective()

        solution = opt_model.solve()
        for expected, value in zip([6.0, 0.0, 0.0], opt_model.get_charging_power()[0]):
            assert abs(expected - value) < EPS
        assert abs(solution.total_cost - 54.0) < EPS

    def test_optimization_high_tariff(self):
        data = pl.DataFrame(
            {
                "time": [5, 10, 15],
                "energy_demand": [0.0, 30.0, 0.0],
                "depot_charge": [True, False, True],
                "battery_capacity": [50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 6.0],
            }
        )
        energy_price = pl.DataFrame({"time": [5, 10, 15], "energy_price": [1.0, 0.0, 2.0]})
        grid_tariff = 6.0

        opt_input = OptimizationInput([data], energy_price, grid_tariff)
        assert opt_input.is_feasible()[0]
        opt_model = OptimizationModel(opt_input)
        opt_model.set_variables()
        opt_model.set_constraints()
        opt_model.set_objective()

        solution = opt_model.solve()
        for expected, value in zip([3.0, 0.0, 3.0], opt_model.get_charging_power()[0]):
            assert abs(expected - value) < EPS
        assert abs(solution.total_cost - 63.0) < EPS

    def test_optimization_no_solution(self):
        data = pl.DataFrame(
            {
                "time": [5, 10, 15],
                "energy_demand": [0.0, 30.0, 0.0],
                "depot_charge": [True, False, True],
                "battery_capacity": [50.0, 50.0, 50.0],
                "max_charging_power": [3.0, 0.0, 2.0],
            }
        )
        energy_price = pl.DataFrame({"time": [5, 10, 15], "energy_price": [1.0, 0.0, 2.0]})
        grid_tariff = 4.0

        opt_input = OptimizationInput([data], energy_price, grid_tariff)
        assert opt_input.is_feasible() == (
            False,
            {"not enough time to charge": [0], "not enough battery capacity": []},
        )
        opt_model = OptimizationModel(opt_input)
        opt_model.set_variables()
        opt_model.set_constraints()
        opt_model.set_objective()

        solution = opt_model.solve()
        assert solution is None


class TestOptimizationNaiveGreedySolution:
    def test_simple(self):
        data = pl.DataFrame(
            {
                "time": [5, 10, 15],
                "energy_demand": [0.0, 30.0, 0.0],
                "depot_charge": [True, False, True],
                "battery_capacity": [50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 6.0],
            }
        )
        energy_price = pl.DataFrame({"time": [5, 10, 15], "energy_price": [1.0, 0.0, 2.0]})

        opt_input = OptimizationInput([data], energy_price, 0.0)
        assert opt_input.is_feasible()[0]
        opt_model = GreedyOptimizationModel(opt_input)
        opt_model.set_variables()
        opt_model.set_constraints()
        opt_model.set_objective()
        solution = opt_model.solve()
        assert abs(solution.total_cost - 60.0) < EPS

    def test_complex(self):
        data = pl.DataFrame(
            {
                "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                "energy_demand": [0.0, 20.0, 10.0, 0.0, 10.0, 0.0, 0.0, 20.0, 0.0, 0.0],
                "depot_charge": [True, False, False, True, False, True, True, False, True, True],
                "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 5.0, 4.0, 5.0, 4.0, 6.0, 0.0, 2.0, 4.0],
            }
        )
        energy_price = pl.DataFrame(
            {
                "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                "energy_price": [1.0, 0.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 5.0, 2.0],
            }
        )

        opt_input = OptimizationInput([data], energy_price, 0.0)
        assert opt_input.is_feasible()[0]
        opt_model = GreedyOptimizationModel(opt_input)
        opt_model.set_variables()
        opt_model.set_constraints()
        opt_model.set_objective()
        solution = opt_model.solve()
        assert abs(solution.total_cost - 170.0) < EPS
