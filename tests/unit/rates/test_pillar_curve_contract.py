"""The PillarCurve contract on fixed knots and forwards: methods, aging, bumps.

The spot rate and construction from pillars have their own files.
"""

import math
from datetime import timedelta

import numpy as np
import pytest

from fintoolsom.rates import (
    PillarCurve,
    InterpolationMethod,
    ZeroCouponCurve,
)
from tests.factories.curve_factory import (
    curve_from_zero_coupon_curve,
)

from .test_PillarCurve import DFS, FORWARDS, PILLARS, T

PC = InterpolationMethod.PiecewiseConstant
EVERY_METHOD = sorted(PillarCurve.allowed_methods, key=lambda m: m.name)
KNOTS = [T, *PILLARS]
INSIDE = [T + timedelta(days=n) for n in (1, 10, 95, 400, 1100, 2500, 3655)]

_BUILD = {"pillar": lambda method: PillarCurve(T, KNOTS, FORWARDS, method)}
_FROM_ZCC = {"pillar": curve_from_zero_coupon_curve}


@pytest.fixture(params=list(_BUILD))
def role(request) -> str:
    return request.param


def _curve(role: str, method=PC):
    return _BUILD[role](method)


def _zero_shift(curve, bumped, dates) -> list[float]:
    return [
        (math.log(curve.get_df(d)) - math.log(bumped.get_df(d))) / ((d - T).days / 365)
        for d in dates
    ]


# ── Methods ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", EVERY_METHOD)
def test_the_dfs_at_the_knots_do_not_depend_on_the_method(role, method):
    np.testing.assert_allclose(_curve(role, method).get_dfs(PILLARS), DFS, rtol=1e-13)


def test_log_linear_is_only_for_the_legacy_curve(role):
    with pytest.raises(ValueError, match="does not support interpolation 'LogLinear'"):
        _curve(role, InterpolationMethod.LogLinear)


def test_piecewise_constant_matches_a_log_linear_legacy_curve(role):
    zcc = ZeroCouponCurve(T, date_dfs=list(zip(PILLARS, DFS)))
    converted = _FROM_ZCC[role](zcc)
    assert converted.interpolation_method is PC
    assert converted.get_dfs(INSIDE) == pytest.approx(zcc.get_dfs(INSIDE), rel=1e-12)
    assert converted.forward_rates == pytest.approx(FORWARDS, rel=1e-10)


def test_cubic_spline_forward_joins_the_flat_extrapolation_smoothly(role):
    curve = _curve(role, InterpolationMethod.CubicSpline)
    last, day = PILLARS[-1], timedelta(days=1)
    before = math.log(curve.get_df(last - day)) - math.log(curve.get_df(last))
    after = math.log(curve.get_df(last)) - math.log(curve.get_df(last + day))
    assert before == pytest.approx(after, abs=1e-8)


# ── Bumps and copies ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "method",
    [PC, InterpolationMethod.PiecewiseLinear, InterpolationMethod.CubicSpline],
)
def test_parallel_bump_shifts_every_continuous_zero_rate_and_returns_a_copy(
    role, method
):
    curve = _curve(role, method)
    bumped = curve.parallel_bump_bps(2)
    dates = INSIDE + [PILLARS[-1] + timedelta(days=500)]
    assert _zero_shift(curve, bumped, dates) == pytest.approx(
        [2e-4] * len(dates), abs=1e-11
    )
    assert bumped.interpolation_method is method
    assert curve.forward_rates == pytest.approx(FORWARDS)  # the original is untouched
    assert curve.copy().get_dfs(dates) == pytest.approx(curve.get_dfs(dates))


# ── Aging ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad", [T, T - timedelta(days=1), PILLARS[-1], PILLARS[-1] + timedelta(days=1)]
)
def test_aging_outside_the_curve_raises(role, bad):
    with pytest.raises(ValueError):
        _curve(role).get_aged_curve(bad)


@pytest.mark.parametrize("offset", [0, 1])
def test_aging_needs_two_segments_left(role, offset):
    with pytest.raises(ValueError, match="second-to-last knot"):
        _curve(role).get_aged_curve(PILLARS[-2] + timedelta(days=offset))
    _curve(role).get_aged_curve(PILLARS[-2] - timedelta(days=1))


@pytest.mark.parametrize("method", EVERY_METHOD)
def test_aged_dfs_at_the_knots_are_forward_dfs_for_every_method(role, method):
    curve = _curve(role, method)
    age = T + timedelta(days=200)
    aged = curve.get_aged_curve(age)
    kept = [p for p in PILLARS if p > age]
    assert type(aged) is type(curve)
    assert aged.curve_date == age and aged.knot_dates == [age, *kept]
    assert aged.get_df(age) == pytest.approx(1.0)
    np.testing.assert_allclose(
        aged.get_dfs(kept), curve.get_dfs(kept) / curve.get_df(age), rtol=1e-12
    )


@pytest.mark.parametrize("age_days", [30, 100, 300])
def test_piecewise_constant_aged_df_is_the_forward_df_between_knots_too(role, age_days):
    curve = _curve(role)
    t1 = T + timedelta(days=age_days)
    aged = curve.get_aged_curve(t1)
    for t2 in [t1 + timedelta(days=n) for n in (1, 50, 200, 900, 3000)]:
        assert curve.get_df(t1) * aged.get_df(t2) == pytest.approx(
            curve.get_df(t2), rel=1e-12
        )
        assert aged.get_df(t2) == pytest.approx(curve.get_df_fwd(t1, t2), rel=1e-12)
