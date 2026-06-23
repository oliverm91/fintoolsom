from datetime import date, timedelta

import pytest

from fintoolsom.rates import Rate, RateConvention, LinearInterestConvention, CompoundedInterestConvention, ExponentialInterestConvention
from fintoolsom.dates import ActualDayCountConvention


def _expected_converted_rate(rate_value, from_convention, to_convention, t_start, t_end):
    from_tf = from_convention.day_count_convention.get_time_fraction(t_start, t_end, from_convention.time_fraction_base)
    wf = from_convention.interest_convention.get_wf_from_rate(rate_value, from_tf)
    to_tf = to_convention.day_count_convention.get_time_fraction(t_start, t_end, to_convention.time_fraction_base)
    return to_convention.interest_convention.get_rate_from_wf(wf, to_tf)


def test_convert_rate_convention_compounded_to_linear():
    t_start = date(2024, 1, 15)
    t_end = t_start + timedelta(days=90)
    from_convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    to_convention = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)

    rate = Rate(from_convention, 0.045)
    expected = _expected_converted_rate(0.045, from_convention, to_convention, t_start, t_end)

    rate.convert_rate_convention(to_convention, t_start, t_end)

    assert rate.rate_value == pytest.approx(expected)
    assert rate.rate_convention is to_convention


def test_convert_rate_convention_compounded_to_exponential():
    t_start = date(2023, 6, 1)
    t_end = t_start + timedelta(days=180)
    from_convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    to_convention = RateConvention(ExponentialInterestConvention, ActualDayCountConvention, 365)

    rate = Rate(from_convention, 0.06)
    expected = _expected_converted_rate(0.06, from_convention, to_convention, t_start, t_end)

    rate.convert_rate_convention(to_convention, t_start, t_end)

    assert rate.rate_value == pytest.approx(expected)


def test_convert_rate_convention_exponential_to_compounded():
    t_start = date(2022, 3, 10)
    t_end = t_start + timedelta(days=365)
    from_convention = RateConvention(ExponentialInterestConvention, ActualDayCountConvention, 365)
    to_convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)

    rate = Rate(from_convention, 0.03)
    expected = _expected_converted_rate(0.03, from_convention, to_convention, t_start, t_end)

    rate.convert_rate_convention(to_convention, t_start, t_end)

    assert rate.rate_value == pytest.approx(expected)


def test_convert_rate_convention_linear_to_exponential():
    t_start = date(2021, 11, 5)
    t_end = t_start + timedelta(days=45)
    from_convention = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)
    to_convention = RateConvention(ExponentialInterestConvention, ActualDayCountConvention, 365)

    rate = Rate(from_convention, 0.02)
    expected = _expected_converted_rate(0.02, from_convention, to_convention, t_start, t_end)

    rate.convert_rate_convention(to_convention, t_start, t_end)

    assert rate.rate_value == pytest.approx(expected)


def test_convert_rate_convention_exponential_to_linear():
    t_start = date(2020, 7, 20)
    t_end = t_start + timedelta(days=200)
    from_convention = RateConvention(ExponentialInterestConvention, ActualDayCountConvention, 365)
    to_convention = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)

    rate = Rate(from_convention, 0.04)
    expected = _expected_converted_rate(0.04, from_convention, to_convention, t_start, t_end)

    rate.convert_rate_convention(to_convention, t_start, t_end)

    assert rate.rate_value == pytest.approx(expected)


def test_convert_rate_convention_roundtrip_returns_original_value():
    t_start = date(2024, 2, 1)
    t_end = t_start + timedelta(days=120)
    compounded = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    linear = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)

    original_value = 0.0525
    rate = Rate(compounded, original_value)
    rate.convert_rate_convention(linear, t_start, t_end)
    rate.convert_rate_convention(compounded, t_start, t_end)

    assert rate.rate_value == pytest.approx(original_value)
