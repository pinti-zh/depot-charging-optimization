from optimization.utils import expand_values, natural_keys


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
        assert expand_values(time, values, 60, interpolation="linear") == [1, 1, 2, 2]

    def test_float(self):
        time = [120, 240]
        values = [2.4, 3.6]
        assert expand_values(time, values, 120) == [2.4, 3.6]
        assert expand_values(time, values, 60) == [2.4, 2.4, 3.6, 3.6]
        assert expand_values(time, values, 60, interpolation="linear") == [1.2, 1.2, 1.8, 1.8]


class TestNaturalKeysSorting:
    def test_simple(self):
        values = ["2", "1", "3"]
        assert sorted(values, key=natural_keys) == ["1", "2", "3"]

    def test_complex(self):
        values = ["a10", "b101", "a003", "a11", "a7", "b5", "b10", "a5b7"]
        assert sorted(values, key=natural_keys) == ["a003", "a5b7", "a7", "a10", "a11", "b5", "b10", "b101"]
