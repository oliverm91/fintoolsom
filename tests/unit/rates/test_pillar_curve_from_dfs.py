"""PillarCurve from pillars: from_dfs / from_zero_rates, dfs as a result, extrapolation."""

import math
from datetime import date, timedelta

import pytest

from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.rates import (
    CompoundedInterestConvention,
    InterpolationMethod,
    LinearInterestConvention,
    PillarCurve,
    Rate,
    RateConvention,
)
from tests.factories.curve_factory import curve_from_dfs

T = date(2024, 7, 8)
PILLARS = [T + timedelta(days=n) for n in (1, 30, 182, 365, 730, 1825, 3650)]
# A gently humped curve so interpolation choices show.
DFS = [
    math.exp(-(0.04 + 0.01 * math.sin(i)) * (p - T).days / 365)
    for i, p in enumerate(PILLARS)
]
INSIDE = [T + timedelta(days=n) for n in (1, 10, 95, 400, 1100, 2500, 3650)]


def _curve(method=InterpolationMethod.PiecewiseConstant) -> PillarCurve:
    return curve_from_dfs(T, PILLARS, DFS, method)


def test_from_dfs_puts_a_knot_at_the_curve_date_and_one_per_pillar():
    curve = PillarCurve.from_dfs(T, PILLARS, DFS, InterpolationMethod.PiecewiseConstant)
    assert curve.knot_dates == [T, *PILLARS] and curve.last_date == PILLARS[-1]
    assert curve.spot_rate is None and curve.effective_date == T


def test_from_dfs_needs_one_df_per_date():
    with pytest.raises(ValueError, match="differ in length"):
        PillarCurve.from_dfs(T, PILLARS, DFS[:-1], InterpolationMethod.CubicSpline)


def test_from_zero_rates_reads_each_rate_in_its_own_convention():
    linear = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)
    compounded = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    rates = [
        Rate(linear if i % 2 else compounded, 0.04 + 0.001 * i)
        for i in range(len(PILLARS))
    ]
    curve = PillarCurve.from_zero_rates(
        T, PILLARS, rates, InterpolationMethod.HermiteCubicSpline
    )
    expected = [float(r.get_discount_factor(T, p)) for r, p in zip(rates, PILLARS)]
    assert curve.get_dfs(PILLARS) == pytest.approx(expected, rel=1e-14)
    for rate, pillar in zip(rates, PILLARS):
        zero = curve.get_zero_rate(pillar, rate.rate_convention)
        assert zero.rate == pytest.approx(rate.rate, rel=1e-11)


@pytest.mark.parametrize(
    "method",
    [
        InterpolationMethod.PiecewiseConstant,
        InterpolationMethod.HermiteCubicSpline,
        InterpolationMethod.CubicSpline,
    ],
)
def test_reprices_its_pillars_and_is_one_at_the_curve_date(method):
    curve = _curve(method)
    assert curve.get_dfs(PILLARS) == pytest.approx(DFS, rel=1e-14)
    assert curve.get_df(T) == pytest.approx(1.0, abs=1e-15)


def test_extrapolates_the_last_forward_flat():
    curve = _curve()
    tau = [(d - T).days / 365 for d in PILLARS[-2:]]
    last_forward = -(math.log(DFS[-1]) - math.log(DFS[-2])) / (tau[1] - tau[0])
    beyond = PILLARS[-1] + timedelta(days=365)
    assert curve.get_df(beyond) == pytest.approx(
        DFS[-1] * math.exp(-last_forward * 1.0), rel=1e-13
    )


def test_cubic_spline_differs_from_piecewise_constant_between_pillars():
    between = T + timedelta(days=2500)
    assert _curve(InterpolationMethod.CubicSpline).get_df(between) != pytest.approx(
        _curve().get_df(between), rel=1e-6
    )


def test_knots_and_forwards_are_validated():
    with pytest.raises(ValueError, match="strictly increasing"):
        PillarCurve(
            T,
            [T, PILLARS[1], PILLARS[0]],
            [0.05, 0.05],
            InterpolationMethod.PiecewiseConstant,
        )
    with pytest.raises(ValueError, match="one entry per segment"):
        PillarCurve(
            T,
            [T, PILLARS[0], PILLARS[1]],
            [0.05],
            InterpolationMethod.PiecewiseConstant,
        )
    with pytest.raises(ValueError, match="at least 2 segments"):
        PillarCurve(
            T,
            [T, PILLARS[0]],
            [0.05],
            InterpolationMethod.PiecewiseConstant,
        )
    with pytest.raises(ValueError, match="does not support interpolation 'LogLinear'"):
        PillarCurve(
            T,
            [T, PILLARS[0]],
            [0.05],
            InterpolationMethod.LogLinear,
        )


# ── aging ───────────────────────────────────────────────────────────────────


# ── forward rates are the parameters ────────────────────────────────────────


def test_is_built_from_forward_rates_and_dfs_are_their_result():
    knots = [T, T + timedelta(days=365), T + timedelta(days=730)]
    curve = PillarCurve(
        T,
        knots,
        [0.05, 0.03],
        InterpolationMethod.PiecewiseConstant,
    )
    assert curve.last_date == knots[-1]
    assert curve.get_dfs(knots) == pytest.approx(
        [1.0, math.exp(-0.05), math.exp(-0.08)], rel=1e-14
    )


def test_knots_and_forwards_fully_define_the_curve():
    curve = _curve()
    rebuilt = PillarCurve(
        T,
        curve.knot_dates,
        curve.forward_rates,
        InterpolationMethod.PiecewiseConstant,
    )
    assert rebuilt.get_dfs(PILLARS) == pytest.approx(DFS, rel=1e-14)
    assert not hasattr(curve, "dfs") and not hasattr(curve, "pillars")


def test_a_first_knot_after_the_curve_date_extends_the_first_forward_back():
    late = [T + timedelta(days=1), T + timedelta(days=90), T + timedelta(days=365)]
    curve = PillarCurve(T, late, [0.05, 0.04], InterpolationMethod.PiecewiseConstant)
    assert curve.get_df(late[0]) == pytest.approx(math.exp(-0.05 / 365), rel=1e-14)
    assert curve.get_df(T) == pytest.approx(1.0, abs=1e-15)
