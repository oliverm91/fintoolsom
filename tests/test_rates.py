from datetime import date, timedelta

import pytest

from fintoolsom.rates import Rate, RateConvention, LinearInterestConvention, CompoundedInterestConvention
from fintoolsom.dates import ActualDayCountConvention


def test_get_rate_from_df_linear_convention():
    days = 70
    df = 0.99
    base = 360
    time_fraction = days / base
    rate_value = LinearInterestConvention.get_rate_from_df(df, time_fraction)
    expected_value = (1 / df - 1) * (base / days)
    assert rate_value == pytest.approx(expected_value)


def test_convert_rate_convention_linear_to_compounded():
    rate_pct = 3
    days = 25
    t_start = date(2023, 10, 17)
    t_end = t_start + timedelta(days=days)

    linear_convention = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)
    rate = Rate(linear_convention, rate_pct / 100)

    compounded_base = 365
    rate.convert_rate_convention(RateConvention(CompoundedInterestConvention, ActualDayCountConvention, compounded_base), t_start, t_end)

    expected_value = (1 + rate_pct / 100 * days / 360) ** (compounded_base / days) - 1
    assert rate.rate_value == pytest.approx(expected_value)
