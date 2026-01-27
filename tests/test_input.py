import pytest

from depot_charging_optimization.data_models import Input


@pytest.fixture
def valid_input_data():
    data = {
        "num_vehicles": 3,
        "time": [300, 600, 1200, 1500, 3000],
        "energy_demand": [
            [0.0, 0.0, 10.0, 10.0, 0.0],
            [0.0, 10.0, 20.0, 10.0, 0.0],
            [20.0, 0.0, 10.0, 0.0, 10.0],
        ],
        "max_charging_power": 1.0,
        "battery_capacity": [30.0, 30.0, 30.0],
        "depot_charge": [
            [True, True, False, False, True],
            [True, False, False, False, True],
            [False, True, False, True, False],
        ],
        "energy_price": [1.0, 1.0, 2.0, 3.0, 1.0],
        "grid_tariff": 2.0,
        "is_battery": [False, False, False],
    }
    return data


@pytest.fixture
def valid_input(valid_input_data):
    return Input(**valid_input_data)


class TestInputCreation:
    def test_valid_input(self, valid_input):
        input_instance = valid_input.model_copy()
        assert input_instance.num_timesteps == len(input_instance.time)

    def test_invalid_num_vehicles(self, valid_input_data):
        data = valid_input_data.copy()
        data["num_vehicles"] += 1
        with pytest.raises(ValueError):
            Input(**data)

    @pytest.mark.parametrize(
        "invalid_time",
        [
            [-60, 0, 60, 120, 180],  # negative
            [0, 60, 120, 180, 240],  # zero
            [0, 60, 60, 120, 180],  # non-increasing
            [60, 120, 180, 240],  # wrong length
        ],
    )
    def test_invalid_time(self, valid_input_data, invalid_time):
        data = valid_input_data.copy()
        data["time"] = invalid_time
        with pytest.raises(ValueError):
            Input(**data)

    @pytest.mark.parametrize(
        "invalid_energy_demand",
        [
            [],  # empty
            [[0.0] * 5, [0.0] * 5],  # wrong amount of lists
            [[0.0] * 4, [0.0] * 4, [0.0] * 4],  # lists with wrong length
        ],
    )
    def test_invalid_energy_demand(self, valid_input_data, invalid_energy_demand):
        data = valid_input_data.copy()
        data["energy_demand"] = invalid_energy_demand
        with pytest.raises(ValueError):
            Input(**data)

    @pytest.mark.parametrize(
        "invalid_depot_charge",
        [
            [],  # empty
            [
                [True, True, False, False, True],
                [True, False, False, False, True],
            ],  # wrong amount of lists
            [
                [True, True, False, False],
                [True, False, False, False],
                [False, True, False, True],
            ],  # lists with wrong length
            [
                [True, True, False, False, True],
                [True, False, False, False, True],
                [False, True, False, True, True],  # last true has nonzero energy demand
            ],  # depot charging with energy demand
        ],
    )
    def test_invalid_depot_charge(self, valid_input_data, invalid_depot_charge):
        data = valid_input_data.copy()
        data["depot_charge"] = invalid_depot_charge
        with pytest.raises(ValueError):
            Input(**data)

    @pytest.mark.parametrize(
        "invalid_battery_capacity",
        [
            [],  # empty
            [30.0, 30.0],  # wrong length
            [0.0, 30.0, 30.0],  # zero capacity
            [-10.0, 30.0, 30.0],  # negative capacity
        ],
    )
    def test_invalid_battery_capacity(self, valid_input_data, invalid_battery_capacity):
        data = valid_input_data.copy()
        data["battery_capacity"] = invalid_battery_capacity
        with pytest.raises(ValueError):
            Input(**data)

    @pytest.mark.parametrize(
        "invalid_is_battery",
        [
            [],  # empty
            [False, False],  # wrong length
        ],
    )
    def test_invalid_is_battery(self, valid_input_data, invalid_is_battery):
        data = valid_input_data.copy()
        data["is_battery"] = invalid_is_battery
        with pytest.raises(ValueError):
            Input(**data)


# test rotate

# test truncate

# test loop

# test maximum possible equal timestep

# test equalize timesteps

# test combine

# test add energy price

# test add grid price
