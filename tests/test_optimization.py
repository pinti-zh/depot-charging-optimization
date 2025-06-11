import polars as pl

from optimization.optimization import OptimizationInput, OptimizationModel


class TestOptimization:
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

        opt_input = OptimizationInput(data, energy_price, grid_tariff)
        opt_model = OptimizationModel(opt_input)
        opt_model.set_variables()
        opt_model.set_constraints()
        opt_model.set_objective()

        solution = opt_model.solve()
        for expected, value in zip([6.0, 0.0], opt_model.get_charging_power()):
            assert expected == value
        assert solution == 54.0

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

        opt_input = OptimizationInput(data, energy_price, grid_tariff)
        opt_model = OptimizationModel(opt_input)
        opt_model.set_variables()
        opt_model.set_constraints()
        opt_model.set_objective()

        solution = opt_model.solve()
        for expected, value in zip([3.0, 3.0], opt_model.get_charging_power()):
            assert expected == value
        assert solution == 63.0

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

        opt_input = OptimizationInput(data, energy_price, grid_tariff)
        opt_model = OptimizationModel(opt_input)
        opt_model.set_variables()
        opt_model.set_constraints()
        opt_model.set_objective()

        solution = opt_model.solve()
        assert solution is None
