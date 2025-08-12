from depot_charging_optimization.data_models import Input
from depot_charging_optimization.scripts.compose_cycles import composable


def generate_input(time, depot_charge):
    return Input.model_validate(
        {
            "num_vehicles": 1,
            "time": time,
            "energy_demand": [[1.0 if not dc else 0.0 for dc in depot_charge]],
            "depot_charge": [depot_charge],
            "soe_lb": [0.2],
            "soe_ub": [0.8],
            "max_charging_power": 1.0,
            "battery_capacity": [10.0],
            "is_battery": [False],
        }
    )


class TestCycleCompositionEqualTime:
    def test_cycle_composition_depot_only(self):
        time = [3600 * (i + 1) for i in range(24)]
        cycle_1 = generate_input(time, [True] * 24)
        cycle_2 = generate_input(time, [True] * 24)
        assert composable([cycle_1, cycle_2])

    def test_cycle_composition_demand_only(self):
        time = [3600 * (i + 1) for i in range(24)]
        cycle_1 = generate_input(time, [False] * 24)
        cycle_2 = generate_input(time, [False] * 24)
        assert not composable([cycle_1, cycle_2])

    def test_partial_overlap(self):
        time = [3600 * (i + 1) for i in range(24)]
        cycle_1 = generate_input(time, [True] * 4 + [False] * 4 + [True] * 16)
        cycle_2 = generate_input(time, [True] * 2 + [False] * 4 + [True] * 18)
        assert not composable([cycle_1, cycle_2])

    def test_exact_overlap(self):
        time = [3600 * (i + 1) for i in range(24)]
        cycle_1 = generate_input(time, [True] * 4 + [False] * 4 + [True] * 16)
        cycle_2 = generate_input(time, [True] * 4 + [False] * 4 + [True] * 16)
        assert not composable([cycle_1, cycle_2])

    def test_total_overlap(self):
        time = [3600 * (i + 1) for i in range(24)]
        cycle_1 = generate_input(time, [True] * 4 + [False] * 4 + [True] * 16)
        cycle_2 = generate_input(time, [True] * 5 + [False] * 2 + [True] * 17)
        assert not composable([cycle_1, cycle_2])


class TestCycleCompositionDifferentTime:
    def test_cycle_composition_depot_only(self):
        time_1 = [3600 * (i + 1) for i in range(24)]
        time_2 = [1800 + 3600 * i for i in range(23)] + [24 * 3600]
        cycle_1 = generate_input(time_1, [True] * 24)
        cycle_2 = generate_input(time_2, [True] * 24)
        assert composable([cycle_1, cycle_2])

    def test_cycle_composition_demand_only(self):
        time_1 = [3600 * (i + 1) for i in range(24)]
        time_2 = [1800 + 3600 * i for i in range(23)] + [24 * 3600]
        cycle_1 = generate_input(time_1, [False] * 24)
        cycle_2 = generate_input(time_2, [False] * 24)
        assert not composable([cycle_1, cycle_2])

    def test_partial_overlap(self):
        time_1 = [3600 * (i + 1) for i in range(24)]
        time_2 = [1800 + 3600 * i for i in range(23)] + [24 * 3600]
        cycle_1 = generate_input(time_1, [True] * 4 + [False] * 4 + [True] * 16)
        cycle_2 = generate_input(time_2, [True] * 3 + [False] * 4 + [True] * 17)
        assert not composable([cycle_1, cycle_2])

    def test_exact_overlap(self):
        time_1 = [3600 * (i + 1) for i in range(24)]
        time_2 = [1800, 5400, 2 * 3600, 3 * 3600, 4 * 3600] + [1800 + 3600 * i for i in range(5, 23)] + [24 * 3600]
        cycle_1 = generate_input(time_1, [True] * 2 + [False] * 2 + [True] * 20)
        cycle_2 = generate_input(time_2, [True] * 2 + [False] * 2 + [True] * 20)
        assert not composable([cycle_1, cycle_2])

    def test_total_overlap(self):
        time_1 = [3600 * (i + 1) for i in range(24)]
        time_2 = [1800 + 3600 * i for i in range(23)] + [24 * 3600]
        cycle_1 = generate_input(time_1, [True] * 4 + [False] * 4 + [True] * 16)
        cycle_2 = generate_input(time_2, [True] * 5 + [False] * 2 + [True] * 17)
        assert not composable([cycle_1, cycle_2])
