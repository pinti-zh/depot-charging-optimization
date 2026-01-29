import pytest

from depot_charging_optimization.data_models import Input
from depot_charging_optimization.environment import State

"""
Valid Input Data:

----------------------------------------------------------------------------------------
Vehicle 1 (CAP = 30.0):   [DC]  [DC]     10.0     10.0              [DC]
----------------------------------------------------------------------------------------
Vehicle 2 (CAP = 30.0):   [DC]  10.0     20.0     10.0              [DC]
----------------------------------------------------------------------------------------
Vehicle 3 (CAP = 30.0):   20.0  [DC]     10.0     [DC]              10.0
----------------------------------------------------------------------------------------
                         | . . | . . | . . . . . | . . | . . . . . . . . . . . . . . |
                         0    300   600         1200  1500                          3000
"""


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
        "is_battery": [False, False, False],
    }
    return data


@pytest.fixture
def valid_input(valid_input_data):
    return Input(**valid_input_data)


@pytest.fixture
def valid_initial_state_data():
    return {
        "num_vehicles": 2,
        "state_of_energy": [10.0, 10.0],
        "in_depot": [True, True],
        "battery_capacity": [10.0, 10.0],
    }


@pytest.fixture
def valid_initial_state(valid_initial_state_data):
    return State(**valid_initial_state_data)
