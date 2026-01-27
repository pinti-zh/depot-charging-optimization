import pytest

from depot_charging_optimization.environment import Charger


class TestConstantEfficiencyCharger:
    @pytest.mark.parametrize(
        "max_efficiency",
        [0.1, 0.5, 1.0],
    )
    @pytest.mark.parametrize(
        "max_charging_power",
        [0.0, 1.0, 100.0],
    )
    def test_max_possible_effective_charging_power(self, max_efficiency, max_charging_power):
        charger = Charger(max_charging_power, max_efficiency, 0.0)
        assert charger.max_possible_effective_charging_power() == max_efficiency * max_charging_power

    @pytest.mark.parametrize(
        "max_efficiency",
        [0.1, 0.5, 0.8, 1.0],
    )
    @pytest.mark.parametrize(
        "charging_power",
        [1.0, 10.0, 100.0],
    )
    def test_effective_charging_power(self, max_efficiency, charging_power):
        max_charging_power = 100.0
        charger = Charger(max_charging_power, max_efficiency, 0.0)
        assert charger.effective_charging_power(charging_power) == charging_power * max_efficiency

    @pytest.mark.parametrize(
        "max_efficiency",
        [0.1, 0.5, 0.8, 1.0],
    )
    @pytest.mark.parametrize(
        "effective_charging_power",
        [1.0, 10.0],
    )
    def test_inverse_effective_charging_power(self, max_efficiency, effective_charging_power):
        max_charging_power = 100.0
        charger = Charger(max_charging_power, max_efficiency, 0.0)
        assert (
            charger.inverse_effective_charging_power(effective_charging_power) * max_efficiency
            == effective_charging_power
        )

    @pytest.mark.parametrize(
        "max_efficiency",
        [0.1, 0.5, 0.8, 1.0],
    )
    @pytest.mark.parametrize(
        "charging_power",
        [1.0, 10.0, 100.0],
    )
    def test_inversion(self, max_efficiency, charging_power):
        max_charging_power = 100.0
        charger = Charger(max_charging_power, max_efficiency, 0.0)
        assert (
            charger.inverse_effective_charging_power(charger.effective_charging_power(charging_power))
            == charging_power
        )


class TestQuadraticEfficiencyCharger:
    @pytest.mark.parametrize(
        "max_efficiency",
        [0.5, 1.0],
    )
    @pytest.mark.parametrize(
        "loss_coefficient",
        [0.0, 0.5, 1.0],
    )
    def test_max_possible_effective_charging_power(self, max_efficiency, loss_coefficient):
        max_charging_power = 100.0
        charger = Charger(max_charging_power, max_efficiency, loss_coefficient)
        assert charger.max_possible_effective_charging_power() == max_efficiency * 100.0 * (1 - loss_coefficient / 2)

    @pytest.mark.parametrize(
        "max_efficiency",
        [0.5, 1.0],
    )
    @pytest.mark.parametrize(
        "loss_coefficient",
        [0.0, 0.5, 1.0],
    )
    def test_effective_charging_power(self, max_efficiency, loss_coefficient):
        max_charging_power = 100.0
        charger = Charger(max_charging_power, max_efficiency, loss_coefficient)
        assert charger.effective_charging_power(0.0) == 0.0
        assert charger.effective_charging_power(max_charging_power) == charger.max_possible_effective_charging_power()

    @pytest.mark.parametrize(
        "max_efficiency",
        [0.5, 1.0],
    )
    @pytest.mark.parametrize(
        "loss_coefficient",
        [0.0, 0.5, 1.0],
    )
    def test_inverse_effective_charging_power(self, max_efficiency, loss_coefficient):
        max_charging_power = 100.0
        charger = Charger(max_charging_power, max_efficiency, loss_coefficient)
        assert charger.inverse_effective_charging_power(0.0) == 0.0
        assert charger.inverse_effective_charging_power(charger.max_possible_effective_charging_power()) == 100.0

    @pytest.mark.parametrize(
        "max_efficiency",
        [0.5, 1.0],
    )
    @pytest.mark.parametrize(
        "charging_power",
        [80.0, 100.0],
    )
    @pytest.mark.parametrize(
        "loss_coefficient",
        [0.0, 0.5, 1.0],
    )
    def test_inversion(self, max_efficiency, charging_power, loss_coefficient):
        max_charging_power = 100.0
        charger = Charger(max_charging_power, max_efficiency, loss_coefficient)
        assert (
            abs(
                charger.inverse_effective_charging_power(charger.effective_charging_power(charging_power))
                - charging_power
            )
            < 1e-6
        )
