import numpy as np

from depot_charging_optimization.utils import (
    expand_values,
    group_vehicles_by_index,
    natural_keys,
)


class TestExpandValues:
    def test_bool(self):
        time = [120, 240]
        values = [True, False]
        assert expand_values(time, values, 120) == [True, False]
        assert expand_values(time, values, 60) == [True, True, False, False]

    def test_string(self):
        time = [120, 240]
        values = ["a", "b"]
        assert expand_values(time, values, 120) == ["a", "b"]
        assert expand_values(time, values, 60) == ["a", "a", "b", "b"]

    def test_int(self):
        time = [120, 240]
        values = [2, 4]
        assert expand_values(time, values, 120) == [2, 4]
        assert expand_values(time, values, 60) == [2, 2, 4, 4]
        assert expand_values(time, values, 60, interpolation="split") == [1, 1, 2, 2]
        assert expand_values(time, values, 60, interpolation="linear") == [1, 2, 3, 4]

    def test_float(self):
        time = [120, 240]
        values = [2.4, 3.6]
        assert expand_values(time, values, 120) == [2.4, 3.6]
        assert expand_values(time, values, 60) == [2.4, 2.4, 3.6, 3.6]
        assert expand_values(time, values, 60, interpolation="split") == [1.2, 1.2, 1.8, 1.8]
        assert expand_values(time, values, 60, interpolation="linear") == [1.2, 2.4, 3.0, 3.6]


class TestNaturalKeysSorting:
    def test_simple(self):
        values = ["2", "1", "3"]
        assert sorted(values, key=natural_keys) == ["1", "2", "3"]

    def test_complex(self):
        values = ["a10", "b101", "a003", "a11", "a7", "b5", "b10", "a5b7"]
        assert sorted(values, key=natural_keys) == ["a003", "a5b7", "a7", "a10", "a11", "b5", "b10", "b101"]


class TestGroupVehiclesByIndex:
    def test_simple(self):
        values = [
            np.array([5, 7]),
            np.array([4, 6]),
        ]
        grouped = group_vehicles_by_index(values)
        assert set(grouped.keys()) == set([4, 5, 6, 7])
        assert grouped[4] == [1]
        assert grouped[5] == [0]
        assert grouped[6] == [1]
        assert grouped[7] == [0]

    def test_complex(self):
        values = [
            np.array([0, 3, 4, 6]),
            np.array([0, 3, 5, 6]),
            np.array([1, 2, 6]),
            np.array([1, 4, 8]),
        ]
        grouped = group_vehicles_by_index(values)
        expected_keys = [0, 1, 2, 3, 4, 5, 6, 8]
        assert set(grouped.keys()) == set(expected_keys)
        assert grouped[0] == [0, 1]
        assert grouped[1] == [2, 3]
        assert grouped[2] == [2]
        assert grouped[3] == [0, 1]
        assert grouped[4] == [0, 3]
        assert grouped[5] == [1]
        assert grouped[6] == [0, 1, 2]
        assert grouped[8] == [3]
