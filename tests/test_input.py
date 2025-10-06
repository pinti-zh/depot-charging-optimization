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
        assert input_data.soe_lb == [0.0]
        assert input_data.soe_ub == [1.0]
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
        assert input_data.soe_lb == [0.0, 0.0, 0.0]
        assert input_data.soe_ub == [1.0, 1.0, 1.0]
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
        input_data = Input.combine([inputs[0], inputs[1]])
        input_data = Input.combine([input_data, inputs[2]])

        assert input_data.num_timesteps == 7
        assert input_data.num_vehicles == 3
        assert input_data.time == [100, 200, 400, 500, 600, 700, 1000]
        assert input_data.energy_demand == [
            [10.0, 10.0, 20.0, 10.0, 0.0, 10.0, 30.0],
            [10.0, 0.0, 0.0, 10.0, 10.0, 10.0, 30.0],
            [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 30.0],
        ]
        assert input_data.soe_lb == [0.0, 0.0, 0.0]
        assert input_data.soe_ub == [1.0, 1.0, 1.0]
        assert input_data.max_charging_power == 2.0
        assert input_data.battery_capacity == [100.0, 200.0, 100.0]
        assert input_data.depot_charge == [
            [False, False, False, False, True, False, False],
            [False, True, True, False, False, False, False],
            [True, True, True, True, False, True, False],
        ]


class TestConcatenate:
    def test_simple(self):
        input_data_1 = Input(
            num_vehicles=1,
            time=[1, 2, 3],
            energy_demand=[[0.0, 1.0, 0.0]],
            soe_lb=[0.0],
            soe_ub=[1.0],
            max_charging_power=1.0,
            battery_capacity=[1.0],
            depot_charge=[[True, False, True]],
            is_battery=[False],
        )
        input_data_2 = Input(
            num_vehicles=1,
            time=[1, 2, 3],
            energy_demand=[[1.0, 0.0, 0.0]],
            soe_lb=[0.0],
            soe_ub=[1.0],
            max_charging_power=1.0,
            battery_capacity=[1.0],
            depot_charge=[[False, True, True]],
            is_battery=[False],
        )

        concatenated = Input.concatenate([input_data_1, input_data_2])

        assert concatenated.num_vehicles == 1
        assert concatenated.time == [1, 2, 3]
        assert concatenated.energy_demand == [[1.0, 1.0, 0.0]]
        assert concatenated.soe_lb == [0.0]
        assert concatenated.soe_ub == [1.0]
        assert concatenated.max_charging_power == 1.0
        assert concatenated.battery_capacity == [1.0]
        assert concatenated.depot_charge == [[False, False, True]]
        assert concatenated.is_battery == [False]

    def test_complex(self):
        input_data_1 = Input(
            num_vehicles=1,
            time=[1, 2, 3, 4, 10],
            energy_demand=[[0.0, 1.0, 0.0, 1.0, 0.0]],
            soe_lb=[0.0],
            soe_ub=[1.0],
            max_charging_power=1.0,
            battery_capacity=[1.0],
            depot_charge=[[True, False, True, False, True]],
            is_battery=[False],
        )
        input_data_2 = Input(
            num_vehicles=1,
            time=[5, 8, 10],
            energy_demand=[[0.0, 1.0, 0.0]],
            soe_lb=[0.0],
            soe_ub=[1.0],
            max_charging_power=1.0,
            battery_capacity=[1.0],
            depot_charge=[[True, False, True]],
            is_battery=[False],
        )

        concatenated = Input.concatenate([input_data_1, input_data_2])

        assert concatenated.num_vehicles == 1
        assert concatenated.time == [1, 2, 3, 4, 5, 8, 10]
        assert concatenated.energy_demand == [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
        assert concatenated.soe_lb == [0.0]
        assert concatenated.soe_ub == [1.0]
        assert concatenated.max_charging_power == 1.0
        assert concatenated.battery_capacity == [1.0]
        assert concatenated.depot_charge == [[True, False, True, False, True, False, True]]
        assert concatenated.is_battery == [False]


class TestRotation:
    def test_simple(self):
        input_data = Input(
            num_vehicles=1,
            time=[5, 8, 10],
            energy_demand=[[0.0, 1.0, 0.0]],
            soe_lb=[0.0],
            soe_ub=[1.0],
            max_charging_power=1.0,
            battery_capacity=[1.0],
            depot_charge=[[True, False, True]],
            is_battery=[False],
        )

        rotated = input_data.rotate()
        assert rotated.time == [3, 5, 10]
        assert rotated.energy_demand == [[1.0, 0.0, 0.0]]
        assert rotated.depot_charge == [[False, True, True]]


class TestEnergyPrice:
    def test_energy_price_single(self):
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

        energy_time = [300, 600, 1000]
        energy_price = [1.0, 2.0, 1.0]

        input_data = input_data.add_energy_price(energy_time, energy_price)

        assert input_data.num_timesteps == 5
        assert input_data.num_vehicles == 1
        assert input_data.time == [200, 300, 500, 600, 1000]
        assert input_data.energy_demand == [[20.0, 10.0, 20.0, 0.0, 40.0]]
        assert input_data.soe_lb == [0.0]
        assert input_data.soe_ub == [1.0]
        assert input_data.max_charging_power == 2.0
        assert input_data.battery_capacity == [100.0]
        assert input_data.depot_charge == [[False, False, False, True, False]]
        assert input_data.energy_price == [1.0, 1.0, 2.0, 2.0, 1.0]

    def test_energy_price_multiple(self):
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
        input_data = Input.combine([inputs[0], inputs[1]])
        input_data = Input.combine([input_data, inputs[2]])

        energy_time = [300, 800, 1000]
        energy_price = [1.0, 2.0, 1.0]

        input_data = input_data.add_energy_price(energy_time, energy_price)

        assert input_data.num_timesteps == 9
        assert input_data.num_vehicles == 3
        assert input_data.time == [100, 200, 300, 400, 500, 600, 700, 800, 1000]
        assert input_data.energy_demand == [
            [10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 10.0, 10.0, 20.0],
            [10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 20.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 20.0],
        ]
        assert input_data.soe_lb == [0.0, 0.0, 0.0]
        assert input_data.soe_ub == [1.0, 1.0, 1.0]
        assert input_data.max_charging_power == 2.0
        assert input_data.battery_capacity == [100.0, 200.0, 100.0]
        assert input_data.depot_charge == [
            [False, False, False, False, False, True, False, False, False],
            [False, True, True, True, False, False, False, False, False],
            [True, True, True, True, True, False, True, False, False],
        ]
        assert input_data.energy_price == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]
