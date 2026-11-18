"""ZeroCouponCurve mechanics: pillars, extrapolation, cache, edits and repr.

The Curve interface itself (scalar vs vector queries, forwards, aging, copies, bumps)
is the contract in test_LegacyCurve.py.
"""

from copy import copy
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.rates import (
    InterpolationMethod,
    ZeroCouponCurve,
    ZeroCouponCurvePoint,
)
from tests.factories import curve_factory
from tests.factories.rate_factory import comp_act365, exp_act365, rate


# ── Construction and pillars ────────────────────────────────────────────────


def test_curve_needs_points_or_dfs_and_a_valid_method(curve_date):
    with pytest.raises(ValueError):
        ZeroCouponCurve(curve_date)
    with pytest.raises(TypeError):
        ZeroCouponCurve(
            curve_date,
            date_dfs=[(curve_date + timedelta(days=30), 0.99)],
            df_interpolation_method="linear",
        )


def test_sort_drops_all_points_not_after_curve_date(curve_date):
    curve = ZeroCouponCurve(
        curve_date,
        date_dfs=[
            (curve_date - timedelta(days=7), 0.999),
            (curve_date - timedelta(days=3), 0.998),
        ],
    )
    assert curve.curve_points == []


def test_curve_point_copy_and_str(curve_date):
    point = ZeroCouponCurvePoint(curve_date + timedelta(days=30), rate(0.05))
    duplicate = copy(point)
    assert duplicate is not point and duplicate.rate is not point.rate
    assert point.simple_str() == f"({point.date}|0.05)"


@pytest.mark.parametrize(
    "method", [InterpolationMethod.LogLinear, InterpolationMethod.HermiteCubicSpline]
)
def test_every_interpolation_reprices_its_pillars(method):
    curve = curve_factory.sample_zero_coupon_curve(method)
    for t_i, df in curve.date_dfs:
        assert curve.get_df(t_i) == pytest.approx(df, rel=1e-9)


def test_get_dfs_for_multiple_known_points(sample_zero_coupon_curve):
    known_dates = [date(2024, 7, 11), date(2025, 7, 10), date(2029, 7, 10)]
    expected = [0.99982, 0.98432, 0.89386]
    assert sample_zero_coupon_curve.get_dfs(known_dates) == pytest.approx(
        expected, abs=1e-5
    )


# ── Interpolation and extrapolation ─────────────────────────────────────────


def test_get_df_interpolates_between_known_points(sample_zero_coupon_curve):
    df_before = sample_zero_coupon_curve.get_df(date(2024, 7, 11))
    df_mid = sample_zero_coupon_curve.get_df(date(2024, 7, 14))
    df_after = sample_zero_coupon_curve.get_df(date(2024, 7, 17))
    assert df_after < df_mid < df_before


def test_get_df_outside_the_pillars_uses_the_first_and_last_rates(
    curve_date, sample_zero_coupon_curve
):
    curve = sample_zero_coupon_curve
    short, far = curve_date + timedelta(days=1), curve.dates[-1] + timedelta(days=365)
    first, last = curve.curve_points[0].rate, curve.curve_points[-1].rate
    assert curve.get_df(short) == pytest.approx(first.get_discount_factor(curve_date, short))
    assert curve.get_df(far) == pytest.approx(last.get_discount_factor(curve_date, far))


@pytest.mark.parametrize("method", ["LogLinear", "HermiteCubicSpline"])
def test_get_dfs_does_not_depend_on_the_order_of_the_dates(curve_date, method):
    # Dates before the first pillar, between pillars and after the last one, shuffled.
    curve = curve_factory.sample_zero_coupon_curve(InterpolationMethod[method])
    first, last = curve.curve_points[0].date, curve.curve_points[-1].date
    dates = [
        curve_date + timedelta(days=1),
        date(2025, 1, 15),
        last + timedelta(days=400),
        first,
        date(2027, 3, 3),
        last,
    ]
    one_by_one = [curve.get_df(d) for d in dates]
    for order in (dates, dates[::-1], sorted(dates), [dates[2], *dates[:2], *dates[3:]]):
        dfs = curve.get_dfs(order)
        assert list(dfs) == [one_by_one[dates.index(d)] for d in order]


def test_get_dfs_hands_out_a_copy_of_its_cached_values(sample_zero_coupon_curve):
    dates = [date(2025, 7, 10), date(2026, 1, 5)]
    first = sample_zero_coupon_curve.get_dfs(dates)
    expected = first.copy()
    first *= 0.5  # a caller writing into the result must not change the curve
    again = sample_zero_coupon_curve.get_dfs(dates)
    assert list(again) == list(expected)
    again[0] = -1.0
    assert list(sample_zero_coupon_curve.get_dfs(dates)) == list(expected)


# ── Edits ───────────────────────────────────────────────────────────────────


def test_add_point_adds_or_replaces_a_pillar_and_rejects_past_dates(
    curve_date, sample_zero_coupon_curve
):
    curve = sample_zero_coupon_curve
    new_point = ZeroCouponCurvePoint(date(2030, 7, 10), rate(0.05, comp_act365()))
    length = len(curve)
    curve.add_point(new_point)
    assert len(curve) == length + 1
    assert curve.get_df(new_point.date) == pytest.approx(
        new_point.rate.get_discount_factor(curve_date, new_point.date)
    )
    pillar = curve.curve_points[3].date
    curve.add_point(ZeroCouponCurvePoint(pillar, rate(0.10)))
    assert len(curve) == length + 1  # replaced, not added
    assert curve.get_df(pillar) == pytest.approx(
        1 / 1.10 ** ((pillar - curve_date).days / 365)
    )
    with pytest.raises(ValueError):
        curve.add_point(ZeroCouponCurvePoint(curve_date - timedelta(days=1), rate(0.05)))


def test_delete_point_reduces_curve_length(sample_zero_coupon_curve):
    length = len(sample_zero_coupon_curve)
    removed = sample_zero_coupon_curve.curve_points[0].date
    sample_zero_coupon_curve.delete_point(removed)
    assert len(sample_zero_coupon_curve) == length - 1
    assert removed not in sample_zero_coupon_curve.dates


def test_parallel_bump_shifts_every_pillar_rate(sample_zero_coupon_curve):
    original = sample_zero_coupon_curve.get_zero_rates_values().copy()
    sample_zero_coupon_curve.parallel_bump_rates_bps(50)
    assert sample_zero_coupon_curve.get_zero_rates_values() == pytest.approx(
        original + 0.005
    )
    # A one-element array (what a solver passes) is a scalar bump.
    sample_zero_coupon_curve.parallel_bump_rates_bps(np.array([-50.0]))
    assert sample_zero_coupon_curve.get_zero_rates_values() == pytest.approx(original)


# ── Rates, aging and repr ───────────────────────────────────────────────────


def test_zero_rates_in_their_own_and_another_convention(curve_date):
    curve = curve_factory.flat_zero_coupon_curve(curve_date, 0.05)
    own = curve.get_zero_rates(curve.curve_points[0].rate.rate_convention)
    assert [r.rate for r in own] == pytest.approx(
        [cp.rate.rate for cp in curve.curve_points]
    )
    continuous = curve.get_zero_rates(exp_act365())
    assert [r.rate for r in continuous] == pytest.approx(
        [0.05] * len(curve.curve_points), rel=1e-9
    )
    assert (
        curve.get_zero_rate(curve_date + timedelta(days=500)).rate_convention
        == curve.curve_points[0].rate.rate_convention
    )


def test_forward_wealth_factors_need_matching_lengths(curve_date, sample_zero_coupon_curve):
    with pytest.raises(ValueError):
        sample_zero_coupon_curve.get_wfs_fwds(
            [curve_date], [curve_date + timedelta(days=10), curve_date + timedelta(days=20)]
        )


def test_aged_curve_cannot_start_after_the_last_pillar(sample_zero_coupon_curve):
    with pytest.raises(ValueError):
        sample_zero_coupon_curve.get_aged_curve(
            sample_zero_coupon_curve.curve_points[-1].date + timedelta(days=1)
        )


def test_curve_str_lists_its_pillars(curve_date):
    text = str(curve_factory.flat_zero_coupon_curve(curve_date, 0.05, years=2))
    assert (
        text.startswith(f"ZeroCouponCurve class. Curve date {curve_date}.")
        and text.count("|") == 3
    )
