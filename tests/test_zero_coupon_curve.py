from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.rates import ZeroCouponCurve, ZeroCouponCurvePoint, Rate, RateConvention, CompoundedInterestConvention
from fintoolsom.dates import ActualDayCountConvention


def test_get_df_at_known_point_matches_input(curve_date, sample_zero_coupon_curve):
    known_date = date(2024, 7, 11)
    df = sample_zero_coupon_curve.get_df(known_date)
    assert df == pytest.approx(0.99982, abs=1e-5)


def test_get_dfs_for_multiple_known_points(sample_zero_coupon_curve):
    known_dates = [date(2024, 7, 11), date(2025, 7, 10), date(2029, 7, 10)]
    expected = [0.99982, 0.98432, 0.89386]
    dfs = sample_zero_coupon_curve.get_dfs(known_dates)
    assert dfs == pytest.approx(expected, abs=1e-5)


def test_get_df_interpolates_between_known_points(sample_zero_coupon_curve):
    mid_date = date(2024, 7, 14)
    df_before = sample_zero_coupon_curve.get_df(date(2024, 7, 11))
    df_mid = sample_zero_coupon_curve.get_df(mid_date)
    df_after = sample_zero_coupon_curve.get_df(date(2024, 7, 17))
    assert df_after < df_mid < df_before


def test_get_df_before_first_point_uses_first_point_rate(curve_date, sample_zero_coupon_curve):
    short_date = curve_date + timedelta(days=1)
    expected = sample_zero_coupon_curve.curve_points[0].rate.get_discount_factor(curve_date, short_date)
    df = sample_zero_coupon_curve.get_df(short_date)
    assert df == pytest.approx(expected)


def test_get_df_after_last_point_uses_last_point_rate(curve_date, sample_zero_coupon_curve):
    far_date = sample_zero_coupon_curve.curve_points[-1].date + timedelta(days=365)
    expected = sample_zero_coupon_curve.curve_points[-1].rate.get_discount_factor(curve_date, far_date)
    df = sample_zero_coupon_curve.get_df(far_date)
    assert df == pytest.approx(expected)


def test_get_wf_is_inverse_of_get_df(sample_zero_coupon_curve):
    some_date = date(2026, 1, 10)
    df = sample_zero_coupon_curve.get_df(some_date)
    wf = sample_zero_coupon_curve.get_wf(some_date)
    assert wf == pytest.approx(1 / df)


def test_get_df_fwd_equals_ratio_of_dfs(sample_zero_coupon_curve):
    start_date = date(2025, 7, 10)
    end_date = date(2027, 7, 10)
    fwd = sample_zero_coupon_curve.get_df_fwd(start_date, end_date)
    expected = sample_zero_coupon_curve.get_df(end_date) / sample_zero_coupon_curve.get_df(start_date)
    assert fwd == pytest.approx(expected)


def test_add_point_updates_df_at_that_date(curve_date, sample_zero_coupon_curve):
    new_date = date(2030, 7, 10)
    convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    new_point = ZeroCouponCurvePoint(new_date, Rate(convention, 0.05))
    original_length = len(sample_zero_coupon_curve)

    sample_zero_coupon_curve.add_point(new_point)

    assert len(sample_zero_coupon_curve) == original_length + 1
    expected_df = new_point.rate.get_discount_factor(curve_date, new_date)
    assert sample_zero_coupon_curve.get_df(new_date) == pytest.approx(expected_df)


def test_delete_point_reduces_curve_length(sample_zero_coupon_curve):
    original_length = len(sample_zero_coupon_curve)
    date_to_remove = sample_zero_coupon_curve.curve_points[0].date

    sample_zero_coupon_curve.delete_point(date_to_remove)

    assert len(sample_zero_coupon_curve) == original_length - 1
    assert date_to_remove not in sample_zero_coupon_curve.dates


def test_parallel_bump_rates_bps_increases_zero_rates(sample_zero_coupon_curve):
    original_rates = sample_zero_coupon_curve.get_zero_rates_values().copy()

    sample_zero_coupon_curve.parallel_bump_rates_bps(50)

    bumped_rates = sample_zero_coupon_curve.get_zero_rates_values()
    assert bumped_rates == pytest.approx(original_rates + 0.005)


def test_get_aged_curve_dfs_match_original_forward_dfs(sample_zero_coupon_curve):
    aging_date = date(2025, 1, 10)
    aged_curve = sample_zero_coupon_curve.get_aged_curve(aging_date)

    for cp in aged_curve.curve_points:
        expected_df = sample_zero_coupon_curve.get_df_fwd(aging_date, cp.date)
        assert cp.rate.get_discount_factor(aging_date, cp.date) == pytest.approx(expected_df)
