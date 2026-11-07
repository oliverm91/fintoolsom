"""Rates, ZeroCouponCurve and ProjectionCurve: accessors, conversions and validation."""
import math
from copy import copy
from datetime import timedelta

import numpy as np
import pytest

from fintoolsom.rates import (
    CompoundedInterestConvention,
    DiscountProjectionView,
    ExponentialInterestConvention,
    InterpolationMethod,
    LinearInterestConvention,
    ProjectionCurve,
    ProjectionInterpolationMethod,
    Rate,
    RateConvention,
    ZeroCouponCurve,
    ZeroCouponCurvePoint,
)
from tests.factories.curve_factory import SAMPLE_CURVE_DATE, flat_zero_coupon_curve, sample_zero_coupon_curve
from tests.factories.rate_factory import comp_act365, exp_act365, lin_act360, rate

T = SAMPLE_CURVE_DATE


# ── Rate and conventions ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "convention, name",
    [(LinearInterestConvention, "l"), (CompoundedInterestConvention, "c"), (ExponentialInterestConvention, "e")],
)
def test_interest_conventions_have_short_names(convention, name):
    assert convention.__str__() == name


def test_rate_convention_str_and_copy():
    conv = lin_act360()
    assert str(conv) == "lact360"
    duplicate = copy(conv)
    assert duplicate == conv and duplicate is not conv


def test_rate_value_deprecation_message_names_the_old_attribute():
    with pytest.warns(DeprecationWarning, match=r"Rate\.rate_value is deprecated; use Rate\.rate"):
        _ = Rate(lin_act360(), 0.05).rate_value


def test_rate_constructor_argument_errors():
    with pytest.raises(TypeError):
        Rate(lin_act360())
    with pytest.raises(TypeError), pytest.warns(DeprecationWarning):
        Rate(lin_act360(), 0.05, rate_value=0.05)


def test_rate_vectorised_discount_factors():
    r = rate(0.05)
    ends = [T + timedelta(days=365), T + timedelta(days=730)]
    assert r.get_discount_factor(T, ends) == pytest.approx([1 / 1.05, 1 / 1.05**2])


# ── ZeroCouponCurve ─────────────────────────────────────────────────────────

def test_curve_point_copy_and_str():
    point = ZeroCouponCurvePoint(T + timedelta(days=30), rate(0.05))
    duplicate = copy(point)
    assert duplicate is not point and duplicate.rate is not point.rate
    assert point.simple_str() == f"({point.date}|0.05)"


def test_curve_needs_points_or_dfs_and_a_valid_method():
    with pytest.raises(ValueError):
        ZeroCouponCurve(T)
    with pytest.raises(TypeError):
        ZeroCouponCurve(T, date_dfs=[(T + timedelta(days=30), 0.99)], df_interpolation_method="linear")


def test_add_point_replaces_a_pillar_and_rejects_past_dates():
    curve = sample_zero_coupon_curve()
    pillar = curve.curve_points[3].date
    curve.add_point(ZeroCouponCurvePoint(pillar, rate(0.10)))
    assert curve.get_df(pillar) == pytest.approx(1 / 1.10 ** ((pillar - T).days / 365))
    with pytest.raises(ValueError):
        curve.add_point(ZeroCouponCurvePoint(T - timedelta(days=1), rate(0.05)))


def test_parallel_bump_accepts_a_numpy_scalar_array():
    curve = flat_zero_coupon_curve(T, 0.05)
    d = T + timedelta(days=730)
    before = curve.get_df(d)
    curve.parallel_bump_rates_bps(np.array([10.0]))
    assert curve.get_df(d) < before
    curve.parallel_bump_rates_bps(-10.0)
    assert curve.get_df(d) == pytest.approx(before, rel=1e-12)


def test_forward_wealth_factors_need_matching_lengths():
    curve = sample_zero_coupon_curve()
    with pytest.raises(ValueError):
        curve.get_wfs_fwds([T], [T + timedelta(days=10), T + timedelta(days=20)])


def test_forward_rate_values_scalar_and_vector():
    curve = flat_zero_coupon_curve(T, 0.05)
    s, e = T + timedelta(days=365), T + timedelta(days=730)
    one = curve.get_forward_rates_values(s, e, exp_act365())
    many = curve.get_forward_rates_values([s, s], [e, e], exp_act365())
    assert one == pytest.approx(0.05, rel=1e-9)
    assert np.asarray(many) == pytest.approx([0.05, 0.05], rel=1e-9)


def test_zero_rates_in_their_own_and_another_convention():
    curve = flat_zero_coupon_curve(T, 0.05)
    own = curve.get_zero_rates(curve.curve_points[0].rate.rate_convention)
    assert [r.rate for r in own] == pytest.approx([cp.rate.rate for cp in curve.curve_points])
    continuous = curve.get_zero_rates(exp_act365())
    assert [r.rate for r in continuous] == pytest.approx([0.05] * len(curve.curve_points), rel=1e-9)
    assert curve.get_zero_rate(T + timedelta(days=500)).rate_convention == curve.curve_points[0].rate.rate_convention


def test_aged_curve_must_lie_inside_the_curve():
    curve = sample_zero_coupon_curve()
    with pytest.raises(ValueError):
        curve.get_aged_curve(T)
    with pytest.raises(ValueError):
        curve.get_aged_curve(curve.curve_points[-1].date + timedelta(days=1))


def test_curve_str_lists_its_pillars():
    text = str(flat_zero_coupon_curve(T, 0.05, years=2))
    assert text.startswith(f"ZeroCouponCurve class. Curve date {T}.") and text.count("|") == 3


@pytest.mark.parametrize("method", list(InterpolationMethod))
def test_every_interpolation_reprices_its_pillars(method):
    curve = sample_zero_coupon_curve(method)
    for t_i, df in [(d, df) for d, df in curve.date_dfs]:
        assert curve.get_df(t_i) == pytest.approx(df, rel=1e-9)


# ── ProjectionCurve ─────────────────────────────────────────────────────────

KNOTS = [T, T + timedelta(days=1), T + timedelta(days=180), T + timedelta(days=720)]
FWDS = np.array([0.05, 0.055, 0.06])


def test_projection_curve_pseudo_dfs_str_and_equivalent_rate():
    pc = ProjectionCurve(T, KNOTS, FWDS, ProjectionInterpolationMethod.PiecewiseConstant)
    dfs = pc.pseudo_dfs(KNOTS[1:])
    assert np.all(np.diff(dfs) < 0) and dfs[0] == pytest.approx(math.exp(-0.05 / 365))
    assert str(pc).startswith("ProjectionCurve(PiecewiseConstant)") and len(pc) == 3
    s, e = KNOTS[2], KNOTS[3]
    fwd = pc.get_equivalent_forward_rate(s, e, exp_act365())
    assert fwd.rate == pytest.approx(0.06, rel=1e-9)
    assert pc.get_equivalent_forward_rate(s, e).rate_convention == RateConvention()


def test_discount_projection_view_reads_the_discount_curve():
    curve = flat_zero_coupon_curve(T, 0.05)
    view = DiscountProjectionView(curve)
    dates = [T + timedelta(days=100), T + timedelta(days=400)]
    assert view.pseudo_dfs(dates) == pytest.approx(curve.get_dfs(dates))
    assert view.get_equivalent_forward_rate(dates[0], dates[1], exp_act365()).rate == pytest.approx(0.05, rel=1e-3)
    assert view.get_equivalent_forward_rate(dates[0], dates[1]).rate_convention == RateConvention()
    assert comp_act365() == RateConvention()
