import polars as pl

from optimization.optimization import OptimizationInput


class TestOptimizationInput:
    def test_ok(self):
        data = pl.DataFrame(
            {
                "time": [5, 10, 15, 20, 25, 30],
                "energy_demand": [0.0, 30.0, 0.0, 10.0, 10.0, 0.0],
                "depot_charge": [True, False, True, False, False, True],
                "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 6.0, 0.0, 0.0, 6.0],
            }
        )
        energy_price = pl.DataFrame({"time": [5, 10, 15, 20, 25, 30], "energy_price": [1.0] * 6})
        grid_tariff = 1.0

        opt_input = OptimizationInput(data, energy_price, grid_tariff)
        assert opt_input.is_feasible()[0]

    def test_not_enough_battery_capacity(self):
        data = pl.DataFrame(
            {
                "time": [5, 10, 15, 20, 25, 30],
                "energy_demand": [0.0, 20.0, 0.0, 20.0, 20.0, 0.0],
                "depot_charge": [True, False, True, False, False, True],
                "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 6.0, 0.0, 0.0, 6.0],
            }
        )
        energy_price = pl.DataFrame({"time": [5, 10, 15, 20, 25, 30], "energy_price": [1.0] * 6})
        grid_tariff = 1.0

        opt_input = OptimizationInput(data, energy_price, grid_tariff)
        assert opt_input.is_feasible() == (False, "not enough battery capacity")

    def test_not_enough_time_to_charge(self):
        data = pl.DataFrame(
            {
                "time": [5, 10, 15, 20, 25, 30],
                "energy_demand": [0.0, 30.0, 0.0, 10.0, 10.0, 0.0],
                "depot_charge": [True, False, True, False, False, True],
                "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                "max_charging_power": [2.0, 0.0, 2.0, 0.0, 0.0, 2.0],
            }
        )
        energy_price = pl.DataFrame({"time": [5, 10, 15, 20, 25, 30], "energy_price": [1.0] * 6})
        grid_tariff = 1.0

        opt_input = OptimizationInput(data, energy_price, grid_tariff)
        assert opt_input.is_feasible() == (False, "not enough time to charge")


class TestOptimizationInputNaiveGreedySolution:
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

        opt_input = OptimizationInput(data, energy_price, 0.0)
        assert opt_input.is_feasible()[0]
        assert opt_input.naive_greedy_solution().total_cost == 60.0

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

        opt_input = OptimizationInput(data, energy_price, 0.0)
        assert opt_input.is_feasible()[0]
        assert opt_input.naive_greedy_solution().total_cost == 170.0
