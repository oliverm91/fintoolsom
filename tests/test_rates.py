from datetime import date, timedelta

import pytest

from fintoolsom.rates import (
    Rate,
    RateConvention,
    LinearInterestConvention,
    CompoundedInterestConvention,
)
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

    linear_convention = RateConvention(
        LinearInterestConvention, ActualDayCountConvention, 360
    )
    rate = Rate(linear_convention, rate_pct / 100)

    compounded_base = 365
    rate.convert_rate_convention(
        RateConvention(
            CompoundedInterestConvention, ActualDayCountConvention, compounded_base
        ),
        t_start,
        t_end,
    )

    expected_value = (1 + rate_pct / 100 * days / 360) ** (compounded_base / days) - 1
    assert rate.rate_value == pytest.approx(expected_value)


def test_rate_get_accrued_interest_matches_wealth_factor_formula():
    rate_convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    rate = Rate(rate_convention, 0.05)
    t_start = date(2024, 1, 1)
    t_end = date(2024, 7, 1)
    notional = 1_000

    accrued_interest = rate.get_accrued_interest(notional, t_start, t_end)

    wf = rate.get_wealth_factor(t_start, t_end)
    assert accrued_interest == pytest.approx(notional * (wf - 1))


def test_rate_copy_is_independent_from_original():
    rate_convention = RateConvention(
        LinearInterestConvention, ActualDayCountConvention, 360
    )
    rate = Rate(rate_convention, 0.05)
    rate_copy = rate.copy()

    rate_copy.rate_value = 0.10
    rate_copy.rate_convention.time_fraction_base = 365

    assert rate.rate_value == 0.05
    assert rate.rate_convention.time_fraction_base == 360


def test_rate_convert_rate_convention_roundtrip_preserves_wealth_factor():
    t_start = date(2024, 1, 1)
    t_end = date(2024, 7, 1)
    linear_convention = RateConvention(
        LinearInterestConvention, ActualDayCountConvention, 360
    )
    compounded_convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )

    rate = Rate(linear_convention, 0.05)
    original_wf = rate.get_wealth_factor(t_start, t_end)

    rate.convert_rate_convention(compounded_convention, t_start, t_end)
    rate.convert_rate_convention(linear_convention, t_start, t_end)

    assert rate.get_wealth_factor(t_start, t_end) == pytest.approx(original_wf)
