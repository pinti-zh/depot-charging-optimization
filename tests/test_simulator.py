import pandas as pd

from depot_charging_optimization.data_models import Input
from depot_charging_optimization.simulator import GreedySimulator, PeakShavingSimulator

EPS = 1e-4


class TestGreedySimulator:
    def test_simple(self):
        df = pd.DataFrame(
            {
                "time": [5, 10, 15, 20],
                "energy_demand": [0.0, 30.0, 0.0, 0.0],
                "depot_charge": [True, False, True, True],
                "battery_capacity": [50.0, 50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 6.0, 6.0],
            }
        )
        energy_price = pd.DataFrame({"time": [5, 10, 15, 20], "energy_price": [1.0, 0.0, 2.0, 1.0]})
        grid_tariff = 0.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        simulator = GreedySimulator(input_data)

        solution = simulator.run()
        assert abs(solution.total_cost - 60.0) < EPS

    def test_complex(self):
        df = pd.DataFrame(
            {
                "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                "energy_demand": [0.0, 20.0, 10.0, 0.0, 10.0, 0.0, 0.0, 20.0, 0.0, 0.0],
                "depot_charge": [True, False, False, True, False, True, True, False, True, True],
                "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                "max_charging_power": [5.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0, 5.0, 5.0],
            }
        )
        energy_price = pd.DataFrame(
            {
                "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                "energy_price": [1.0, 0.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 5.0, 2.0],
            }
        )
        grid_tariff = 0.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        simulator = GreedySimulator(input_data)

        solution = simulator.run()
        assert abs(solution.total_cost - 195.0) < EPS


class TestPeakShavingSimulator:
    def test_simple(self):
        df = pd.DataFrame(
            {
                "time": [5, 10, 15, 20],
                "energy_demand": [0.0, 30.0, 0.0, 0.0],
                "depot_charge": [True, False, True, True],
                "battery_capacity": [50.0, 50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 6.0, 6.0],
            }
        )
        energy_price = pd.DataFrame({"time": [5, 10, 15, 20], "energy_price": [1.0, 0.0, 2.0, 1.0]})
        grid_tariff = 1.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        simulator = PeakShavingSimulator(input_data, 6.0)

        solution = simulator.run()
        assert abs(solution.total_cost - 42.0) < EPS

    def test_complex(self):
        df = pd.DataFrame(
            {
                "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                "energy_demand": [0.0, 20.0, 10.0, 0.0, 10.0, 0.0, 0.0, 20.0, 0.0, 0.0],
                "depot_charge": [True, False, False, True, False, True, True, False, True, True],
                "battery_capacity": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                "max_charging_power": [5.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0, 5.0, 5.0],
            }
        )
        energy_price = pd.DataFrame(
            {
                "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                "energy_price": [1.0, 0.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 5.0, 2.0],
            }
        )
        grid_tariff = 1.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        simulator = PeakShavingSimulator(input_data, 4.0)

        solution = simulator.run()
        assert abs(solution.total_cost - 412.0 / 3.0) < EPS
