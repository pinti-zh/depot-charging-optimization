import polars as pl

from optimization.optimization import OptimizationInput


class TestOptimizationInput:
    def test_feasible(self):
        data = [
            pl.DataFrame(
                {
                    "time": [5, 10, 15, 20, 25, 30],
                    "energy_demand": [0.0, 30.0, 0.0, 10.0, 10.0, 0.0],
                    "depot_charge": [True, False, True, False, False, True],
                    "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                    "max_charging_power": [6.0, 0.0, 6.0, 0.0, 0.0, 6.0],
                }
            ),
            pl.DataFrame(
                {
                    "time": [5, 10, 15, 20, 25, 30],
                    "energy_demand": [0.0, 20.0, 0.0, 20.0, 10.0, 0.0],
                    "depot_charge": [True, False, True, False, False, True],
                    "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                    "max_charging_power": [6.0, 0.0, 6.0, 0.0, 0.0, 6.0],
                }
            ),
            pl.DataFrame(
                {
                    "time": [5, 10, 15, 20, 25, 30],
                    "energy_demand": [0.0, 30.0, 0.0, 10.0, 10.0, 0.0],
                    "depot_charge": [True, False, True, False, False, True],
                    "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                    "max_charging_power": [2.0, 0.0, 6.0, 0.0, 0.0, 2.0],
                }
            ),
        ]
        energy_price = pl.DataFrame({"time": [5, 10, 15, 20, 25, 30], "energy_price": [1.0] * 6})
        grid_tariff = 1.0

        opt_input = OptimizationInput(data, energy_price, grid_tariff)
        ok, _ = opt_input.is_feasible()
        assert ok

    def test_infeasible(self):
        data = [
            pl.DataFrame(
                {
                    "time": [5, 10, 15, 20, 25, 30],
                    "energy_demand": [0.0, 30.0, 0.0, 10.0, 10.0, 0.0],
                    "depot_charge": [True, False, True, False, False, True],
                    "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                    "max_charging_power": [6.0, 0.0, 6.0, 0.0, 0.0, 6.0],
                }
            ),
            pl.DataFrame(
                {
                    "time": [5, 10, 15, 20, 25, 30],
                    "energy_demand": [0.0, 20.0, 0.0, 20.0, 20.0, 0.0],
                    "depot_charge": [True, False, True, False, False, True],
                    "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                    "max_charging_power": [6.0, 0.0, 6.0, 0.0, 0.0, 6.0],
                }
            ),
            pl.DataFrame(
                {
                    "time": [5, 10, 15, 20, 25, 30],
                    "energy_demand": [0.0, 10.0, 0.0, 10.0, 10.0, 0.0],
                    "depot_charge": [True, False, True, False, False, True],
                    "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                    "max_charging_power": [1.0, 0.0, 2.0, 0.0, 0.0, 2.0],
                }
            ),
        ]
        energy_price = pl.DataFrame({"time": [5, 10, 15, 20, 25, 30], "energy_price": [1.0] * 6})
        grid_tariff = 1.0

        opt_input = OptimizationInput(data, energy_price, grid_tariff)
        ok, reasons = opt_input.is_feasible()
        assert not ok
        assert reasons["not enough battery capacity"] == [1]
        assert reasons["not enough time to charge"] == [2]
