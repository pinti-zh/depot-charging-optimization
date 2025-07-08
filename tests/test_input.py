import pandas as pd

from depot_charging_optimization.data_models import Input


class TestInputCreation:

    def test_creation_from_dataframe(self):
        df = pd.DataFrame(
            {
                "time": [200, 500, 600, 1000],
                "energy_demand": [20.0, 30.0, 0.0, 40.0],
                "depot_charge": [False, False, True, False],
                "battery_capacity": [100.0, 100.0, 100.0, 100.0],
                "max_charging_power": [0.0, 10.0, 2.0, 0.0],
            }
        )

        input_data = Input.from_dataframe(df)

        assert input_data.num_timesteps == 4
        assert input_data.num_vehicles == 1
        assert input_data.time == [200, 500, 600, 1000]
        assert input_data.energy_demand == [[20.0, 30.0, 0.0, 40.0]]
        assert input_data.soe_lb == [0.2]
        assert input_data.soe_ub == [0.8]
        assert input_data.max_charging_power == 2.0
        assert input_data.battery_capacity == [100.0]
        assert input_data.depot_charge == [[False, False, True, False]]


class TestCombine:

    def test_combination_from_singles(self):
        dataframes = []
        dataframes.append(
            pd.DataFrame(
                {
                    "time": [200, 500, 600, 1000],
                    "energy_demand": [20.0, 30.0, 0.0, 40.0],
                    "depot_charge": [False, False, True, False],
                    "battery_capacity": [100.0, 100.0, 100.0, 100.0],
                    "max_charging_power": [0.0, 10.0, 2.0, 0.0],
                }
            )
        )
        dataframes.append(
            pd.DataFrame(
                {
                    "time": [100, 400, 600, 1000],
                    "energy_demand": [10.0, 0.0, 20.0, 40.0],
                    "depot_charge": [False, True, False, False],
                    "battery_capacity": [200.0, 200.0, 200.0, 200.0],
                    "max_charging_power": [0.0, 2.0, 0.0, 0.0],
                }
            )
        )
        dataframes.append(
            pd.DataFrame(
                {
                    "time": [500, 600, 700, 1000],
                    "energy_demand": [0.0, 10.0, 0.0, 30.0],
                    "depot_charge": [True, False, True, False],
                    "battery_capacity": [100.0, 100.0, 100.0, 100.0],
                    "max_charging_power": [2.0, 0.0, 2.0, 0.0],
                }
            )
        )

        inputs = [Input.from_dataframe(df) for df in dataframes]
        input_data = Input.combine(inputs)

        assert input_data.num_timesteps == 7
        assert input_data.num_vehicles == 3
        assert input_data.time == [100, 200, 400, 500, 600, 700, 1000]
        assert input_data.energy_demand == [
            [10.0, 10.0, 20.0, 10.0, 0.0, 10.0, 30.0],
            [10.0, 0.0, 0.0, 10.0, 10.0, 10.0, 30.0],
            [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 30.0],
        ]
        assert input_data.soe_lb == [0.2, 0.2, 0.2]
        assert input_data.soe_ub == [0.8, 0.8, 0.8]
        assert input_data.max_charging_power == 2.0
        assert input_data.battery_capacity == [100.0, 200.0, 100.0]
        assert input_data.depot_charge == [
            [False, False, False, False, True, False, False],
            [False, True, True, False, False, False, False],
            [True, True, True, True, False, True, False],
        ]

    def test_sequential_combination(self):
        dataframes = []
        dataframes.append(
            pd.DataFrame(
                {
                    "time": [200, 500, 600, 1000],
                    "energy_demand": [20.0, 30.0, 0.0, 40.0],
                    "depot_charge": [False, False, True, False],
                    "battery_capacity": [100.0, 100.0, 100.0, 100.0],
                    "max_charging_power": [0.0, 10.0, 2.0, 0.0],
                }
            )
        )
        dataframes.append(
            pd.DataFrame(
                {
                    "time": [100, 400, 600, 1000],
                    "energy_demand": [10.0, 0.0, 20.0, 40.0],
                    "depot_charge": [False, True, False, False],
                    "battery_capacity": [200.0, 200.0, 200.0, 200.0],
                    "max_charging_power": [0.0, 2.0, 0.0, 0.0],
                }
            )
        )
        dataframes.append(
            pd.DataFrame(
                {
                    "time": [500, 600, 700, 1000],
                    "energy_demand": [0.0, 10.0, 0.0, 30.0],
                    "depot_charge": [True, False, True, False],
                    "battery_capacity": [100.0, 100.0, 100.0, 100.0],
                    "max_charging_power": [2.0, 0.0, 2.0, 0.0],
                }
            )
        )

        inputs = [Input.from_dataframe(df) for df in dataframes]
        print("first combine")
        input_data = Input.combine([inputs[0], inputs[1]])
        print("second combine")
        input_data = Input.combine([input_data, inputs[2]])

        assert input_data.num_timesteps == 7
        assert input_data.num_vehicles == 3
        assert input_data.time == [100, 200, 400, 500, 600, 700, 1000]
        assert input_data.energy_demand == [
            [10.0, 10.0, 20.0, 10.0, 0.0, 10.0, 30.0],
            [10.0, 0.0, 0.0, 10.0, 10.0, 10.0, 30.0],
            [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 30.0],
        ]
        assert input_data.soe_lb == [0.2, 0.2, 0.2]
        assert input_data.soe_ub == [0.8, 0.8, 0.8]
        assert input_data.max_charging_power == 2.0
        assert input_data.battery_capacity == [100.0, 200.0, 100.0]
        assert input_data.depot_charge == [
            [False, False, False, False, True, False, False],
            [False, True, True, False, False, False, False],
            [True, True, True, True, False, True, False],
        ]
