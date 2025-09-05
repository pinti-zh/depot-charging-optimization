import pandas as pd

from depot_charging_optimization.config import OptimizerConfig
from depot_charging_optimization.core import CasadiOptimizer
from depot_charging_optimization.data_models import Input

EPS = 1e-5


class TestOptimizationSingleVehicle:
    def test_optimization_low_tariff(self):
        df = pd.DataFrame(
            {
                "time": [5, 10, 15],
                "energy_demand": [0.0, 30.0, 0.0],
                "depot_charge": [True, False, True],
                "battery_capacity": [50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 6.0],
            }
        )
        energy_price = pd.DataFrame({"time": [5, 10, 15], "energy_price": [1.0, 0.0, 2.0]})
        grid_tariff = 4.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        optimizer = CasadiOptimizer(input_data)
        optimizer.build()

        solution = optimizer.solve()
        for expected, value in zip([6.0, 0.0, 0.0], solution.charging_power[0]):
            assert abs(expected - value) < EPS
        assert abs(solution.total_cost - 54.0) < EPS

    def test_optimization_high_tariff(self):
        df = pd.DataFrame(
            {
                "time": [5, 10, 15],
                "energy_demand": [0.0, 30.0, 0.0],
                "depot_charge": [True, False, True],
                "battery_capacity": [50.0, 50.0, 50.0],
                "max_charging_power": [6.0, 0.0, 6.0],
            }
        )
        energy_price = pd.DataFrame({"time": [5, 10, 15], "energy_price": [1.0, 0.0, 2.0]})
        grid_tariff = 6.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        optimizer = CasadiOptimizer(input_data)
        optimizer.build()

        solution = optimizer.solve()
        for expected, value in zip([3.0, 0.0, 3.0], solution.charging_power[0]):
            assert abs(expected - value) < EPS
        assert abs(solution.total_cost - 63.0) < EPS

    def test_optimization_no_solution(self):
        df = pd.DataFrame(
            {
                "time": [5, 10, 15],
                "energy_demand": [0.0, 30.0, 0.0],
                "depot_charge": [True, False, True],
                "battery_capacity": [50.0, 50.0, 50.0],
                "max_charging_power": [2.0, 0.0, 2.0],
            }
        )
        energy_price = pd.DataFrame({"time": [5, 10, 15], "energy_price": [1.0, 0.0, 2.0]})
        grid_tariff = 4.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        optimizer = CasadiOptimizer(input_data)
        optimizer.build()

        solution = optimizer.solve()
        assert solution is None


class TestOptimiazationChargingEfficiency:
    def test_constant_charging_efficiency(self):
        df = pd.DataFrame(
            {
                "time": [5, 10],
                "energy_demand": [0.0, 18.0],
                "depot_charge": [True, False],
                "battery_capacity": [50.0, 50.0],
                "max_charging_power": [10.0, 0.0],
            }
        )
        energy_price = pd.DataFrame({"time": [5, 10], "energy_price": [1.0, 1.0]})
        grid_tariff = 0.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        optimizer = CasadiOptimizer(input_data)
        optimizer.build(ce_function_type="constant", alpha=0.6)

        solution = optimizer.solve()
        assert abs(solution.total_cost - 30.0) < EPS

    def test_quadratic_charging_efficiency(self):
        df = pd.DataFrame(
            {
                "time": [5, 10],
                "energy_demand": [0.0, 44],
                "depot_charge": [True, False],
                "battery_capacity": [100.0, 100.0],
                "max_charging_power": [25.0, 0.0],
            }
        )
        energy_price = pd.DataFrame({"time": [5, 10], "energy_price": [1.0, 1.0]})
        grid_tariff = 0.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        optimizer = CasadiOptimizer(input_data)
        optimizer.build(ce_function_type="quadratic", alpha=0.4)

        solution = optimizer.solve()
        assert abs(solution.total_cost - 50.0) < EPS

    def test_quadratic_charging_efficiency_max(self):
        df = pd.DataFrame(
            {
                "time": [5, 10],
                "energy_demand": [0.0, 35.0],
                "depot_charge": [True, False],
                "battery_capacity": [100.0, 100.0],
                "max_charging_power": [10.0, 0.0],
            }
        )
        energy_price = pd.DataFrame({"time": [5, 10], "energy_price": [1.0, 1.0]})
        grid_tariff = 0.0

        input_data = Input.from_dataframe(df)
        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        optimizer = CasadiOptimizer(input_data)
        optimizer.build(ce_function_type="quadratic", alpha=0.4)

        solution = optimizer.solve()
        assert abs(solution.total_cost - 50.0) < EPS


class TestInitialSoE:
    def test_simple(self):
        input_data = Input(
            num_vehicles=1,
            time=[1, 2, 3],
            energy_demand=[[0.0, 1.0, 0.0]],
            soe_lb=[0.2],
            soe_ub=[0.8],
            max_charging_power=1.0,
            battery_capacity=[10.0],
            depot_charge=[[True, False, True]],
            is_battery=[False],
        )

        energy_price = pd.DataFrame({"time": [1, 2, 3], "energy_price": [2.0, 1.0, 1.0]})
        grid_tariff = 0.0

        input_data = input_data.add_energy_price(
            energy_price["time"].to_list(), energy_price["energy_price"].to_list()
        )
        input_data = input_data.add_grid_tariff(grid_tariff)

        config = OptimizerConfig(initial_soe=[2.0])

        optimizer = CasadiOptimizer(input_data, config=config)
        optimizer.build()

        solution = optimizer.solve()
        assert abs(solution.total_cost - 2.0) < EPS
