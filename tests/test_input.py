import pytest

from depot_charging_optimization.data_models import Input


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


class TestInputMethods:
    def test_add_valid_energy_price(self, valid_input):
        input_instance = valid_input.model_copy()
        input_instance = input_instance.add_energy_price([300, 900, 1200, 1500, 3000], [1.0, 1.0, 2.0, 3.0, 1.0])
        assert input_instance.energy_price == [1.0, 1.0, 1.0, 2.0, 3.0, 1.0]  # 300 - 600 and 600 - 900 extension

    @pytest.mark.parametrize(
        "energy_price",
        [
            ([], []),  # empty
            ([-60, 3000], [1.0, 1.0]),  # negative time
            ([0, 3000], [1.0, 1.0]),  # zero time
            ([60, 6000], [1.0, 1.0]),  # too long time horizon
            ([60, 1000], [1.0, 1.0]),  # too short time horizon
            ([60, 60, 3000], [1.0, 1.0, 1.0]),  # non-increasing time
            ([60, 3000], [1.0]),  # list length missmatch
        ],
    )
    def test_add_invalid_energy_price(self, valid_input, energy_price):
        input_instance = valid_input.model_copy()
        with pytest.raises(ValueError):
            input_instance.add_energy_price(*energy_price)

    @pytest.mark.parametrize(
        "grid_tariff",
        [
            -1.0,  # negative
            0.0,  # zero
            1.0,  # positive
        ],
    )
    def test_add_grid_tariff(self, valid_input, grid_tariff):
        input_instance = valid_input.model_copy()
        input_instance = input_instance.add_grid_tariff(grid_tariff)
        assert input_instance.grid_tariff == grid_tariff

    def test_rotate(self, valid_input):
        input_instance = valid_input.model_copy()
        time_delta = [t2 - t1 for t1, t2 in zip([0] + input_instance.time[:-1], input_instance.time)]
        rotated_input_instance = input_instance.rotate()
        rotated_time_delta = [
            t2 - t1 for t1, t2 in zip([0] + rotated_input_instance.time[:-1], rotated_input_instance.time)
        ]

        rotation_map = [4, 0, 1, 2, 3]
        for i, rotated_i in enumerate(rotation_map):
            assert time_delta[i] == rotated_time_delta[rotated_i]
            for ed, rotated_ed in zip(input_instance.energy_demand, rotated_input_instance.energy_demand):
                assert ed[i] == rotated_ed[rotated_i]
            for dc, rotated_dc in zip(input_instance.depot_charge, rotated_input_instance.depot_charge):
                assert dc[i] == rotated_dc[rotated_i]

    @pytest.mark.parametrize(
        "num",
        [
            3,  # regular
            10,  # too large but okay
        ],
    )
    def test_valid_truncate(self, valid_input, num):
        input_instance = valid_input.model_copy()
        truncated_input_instance = input_instance.truncate(num)
        assert len(truncated_input_instance.time) == min(num, len(input_instance.time))

    @pytest.mark.parametrize(
        "num",
        [
            -1,  # negative
            0,  # zero
        ],
    )
    def test_invalid_truncate(self, valid_input, num):
        input_instance = valid_input.model_copy()
        with pytest.raises(ValueError):
            input_instance.truncate(num)

    @pytest.mark.parametrize(
        "num",
        [
            1,
            10,
        ],
    )
    def test_valid_loop(self, valid_input, num):
        input_instance = valid_input.model_copy()
        looped_input_instance = input_instance.loop(num)
        assert len(looped_input_instance.time) == num * len(input_instance.time)
        original_length = len(input_instance.time)
        for i, t in enumerate(looped_input_instance.time):
            assert t == input_instance.time[i % original_length] + (i // original_length) * input_instance.time[-1]

        for ed, looped_ed in zip(input_instance.energy_demand, looped_input_instance.energy_demand):
            for i, value in enumerate(looped_ed):
                assert ed[i % len(ed)] == value

        for dc, looped_dc in zip(input_instance.depot_charge, looped_input_instance.depot_charge):
            for i, value in enumerate(looped_dc):
                assert dc[i % len(dc)] == value

    @pytest.mark.parametrize(
        "num",
        [
            -1,  # negative
            0,  # zero
        ],
    )
    def test_invalid_loop(self, valid_input, num):
        input_instance = valid_input.model_copy()
        with pytest.raises(ValueError):
            input_instance.loop(num)

    def test_maximum_possible_equal_timestep(self, valid_input):
        input_instance = valid_input.model_copy()
        assert input_instance.maximum_possible_equal_timestep() == 300

    def test_equalize_timesteps(self, valid_input):
        input_instance = valid_input.model_copy()
        equalized_input_instance = input_instance.equalize_timesteps()
        for t1, t2 in zip([0] + equalized_input_instance.time[:-1], equalized_input_instance.time):
            assert t2 - t1 == 300

    def test_combine(self, valid_input_data):
        input_instances = []

        for i in range(3):
            data = valid_input_data.copy()
            data["time"] = [100 * (i + 1), 600, 1200, 1500, 3000]
            input_instances.append(Input(**data))

        combined_input_instance = Input.combine(input_instances)
        assert combined_input_instance.time == [100, 200, 300, 600, 1200, 1500, 3000]
