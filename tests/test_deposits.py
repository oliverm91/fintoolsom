from datetime import date

import pytest

from fintoolsom.fixedIncome import Deposit


@pytest.fixture
def sample_deposit() -> Deposit:
    return Deposit("TEST", "usd", date(2024, 7, 1), 1000)


def test_deposit_get_value_scales_linearly_with_fx(sample_deposit):
    t = date(2024, 1, 1)
    rate_value = 0.05
    value = sample_deposit.get_value(t, rate_value)
    value_fx2 = sample_deposit.get_value(t, rate_value, fx=2)
    assert value_fx2 == pytest.approx(2 * value)


def test_deposit_get_duration_matches_act365_year_fraction(sample_deposit):
    t = date(2024, 1, 1)
    duration = sample_deposit.get_duration(t)
    expected = (sample_deposit.payment_date - t).days / 365
    assert duration == pytest.approx(expected)


def test_deposit_get_dv01_is_negative_for_positive_rate(sample_deposit):
    t = date(2024, 1, 1)
    dv01 = sample_deposit.get_dv01(t, 0.05)
    assert dv01 < 0
