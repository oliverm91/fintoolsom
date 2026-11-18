"""PillarCurve: forwards as state, dfs derived, spot segment, bumps, validation."""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.market.currencies import Currency, CurrencyName
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.rates import (
    ExponentialInterestConvention,
    PillarCurve,
    InterpolationMethod,
    LinearInterestConvention,
    Rate,
    RateConvention,
)

T = date(2026, 6, 29)
SOFR = OvernightRateIndex("SOFR", currency=Currency(CurrencyName.USD))
KNOTS = [
    T,
    T + timedelta(days=90),
    T + timedelta(days=180),
    T + timedelta(days=365),
    T + timedelta(days=730),
]
FWDS = np.array([0.055, 0.052, 0.050, 0.045])  # continuously compounded, act/365
CONTINUOUS = RateConvention(
    ExponentialInterestConvention, ActualDayCountConvention, 365
)
SPOT = Rate(
    RateConvention(LinearInterestConvention, ActualDayCountConvention, 360), 0.053
)


def _curve(
    interpolation_method=InterpolationMethod.PiecewiseConstant, **kwargs
) -> PillarCurve:
    return PillarCurve(T, KNOTS, FWDS, interpolation_method, **kwargs)


def _tau(d: date) -> float:
    return (d - T).days / 365


def test_df_is_one_at_the_curve_date():
    curve = _curve()
    assert curve.get_df(T) == pytest.approx(1.0, abs=1e-15)
    assert curve.effective_date == T and curve.last_date == KNOTS[-1]


def test_dfs_integrate_the_forwards():
    curve = _curve()
    d = T + timedelta(days=250)  # inside the third segment
    integral = (
        FWDS[0] * _tau(KNOTS[1])
        + FWDS[1] * (_tau(KNOTS[2]) - _tau(KNOTS[1]))
        + FWDS[2] * (_tau(d) - _tau(KNOTS[2]))
    )
    assert curve.get_df(d) == pytest.approx(math.exp(-integral), rel=1e-14)


def test_equivalent_forward_rate_recovers_a_segment_and_the_flat_extrapolation():
    curve = _curve()
    assert curve.get_forward_rate(KNOTS[1], KNOTS[2], CONTINUOUS).rate == pytest.approx(
        0.052, abs=1e-12
    )
    beyond = (KNOTS[-1] + timedelta(days=100), KNOTS[-1] + timedelta(days=200))
    assert curve.get_forward_rate(*beyond, CONTINUOUS).rate == pytest.approx(
        FWDS[-1], abs=1e-12
    )


def test_accrual_is_the_wealth_factor_minus_one_scalar_and_vector():
    curve = _curve()
    starts = [T + timedelta(days=d) for d in (10, 100, 200)]
    ends = [T + timedelta(days=d) for d in (100, 200, 400)]
    vec = curve.get_accrual(starts, ends)
    assert isinstance(vec, np.ndarray)
    for i, (s, e) in enumerate(zip(starts, ends)):
        assert vec[i] == pytest.approx(curve.get_df(s) / curve.get_df(e) - 1, abs=1e-15)
        assert curve.get_accrual(s, e) == pytest.approx(vec[i], abs=1e-15)


def test_the_spot_rate_covers_the_curve_date_to_the_effective_date():
    effective = T + timedelta(days=3)
    knots = [effective, *KNOTS[1:]]
    curve = PillarCurve(
        T,
        knots,
        FWDS,
        InterpolationMethod.PiecewiseConstant,
        SPOT,
    )
    df_spot = 1 / (1 + 0.053 * 3 / 360)
    assert curve.get_df(effective) == pytest.approx(df_spot, rel=1e-14)
    assert curve.get_df(T + timedelta(days=1)) == pytest.approx(
        1 / (1 + 0.053 / 360), rel=1e-14
    )
    d = KNOTS[1]
    assert curve.get_df(d) == pytest.approx(
        df_spot * math.exp(-FWDS[0] * (_tau(d) - _tau(effective))), rel=1e-14
    )


def test_without_a_spot_rate_the_first_forward_extends_back_to_the_curve_date():
    effective = T + timedelta(days=2)  # e.g. a term index: t + spot lag
    curve = PillarCurve(
        T,
        [effective, *KNOTS[1:]],
        FWDS,
        InterpolationMethod.PiecewiseConstant,
    )
    assert curve.get_df(effective) == pytest.approx(
        math.exp(-FWDS[0] * _tau(effective)), rel=1e-14
    )


@pytest.mark.parametrize("spot", [None, SPOT], ids=["no-spot", "spot"])
def test_parallel_bump_shifts_every_continuous_zero_rate_and_returns_a_copy(spot):
    effective = T + timedelta(days=3)
    curve = PillarCurve(
        T,
        [effective, *KNOTS[1:]],
        FWDS,
        InterpolationMethod.PiecewiseConstant,
        spot,
    )
    bumped = curve.parallel_bump_bps(1)
    dates = [effective, *(T + timedelta(days=n) for n in (50, 300, 900))]
    shift = [
        (math.log(curve.get_df(d)) - math.log(bumped.get_df(d))) / _tau(d)
        for d in dates
    ]
    assert shift == pytest.approx([1e-4] * len(dates), abs=1e-12)
    assert curve.forward_rates == pytest.approx(FWDS)  # the original is untouched


def test_copy_is_equal_and_independent():
    curve = _curve(spot_rate=SPOT)
    twin = curve.copy()
    twin.forward_rates[0] = 0.0
    assert curve.forward_rates[0] == FWDS[0] and twin.spot_rate is not curve.spot_rate


def _linear_integral(d: date) -> float:
    """Integral of forwards flat over the first segment, then linear through FWDS."""
    nodes = [FWDS[0], *FWDS]
    taus = [_tau(k) for k in KNOTS]
    forward_at = float(np.interp(_tau(d), taus, nodes))
    ends = [(tau, f) for tau, f in zip(taus, nodes) if tau < _tau(d)]
    ends.append((_tau(d), forward_at))
    return sum((f0 + f1) / 2 * (t1 - t0) for (t0, f0), (t1, f1) in zip(ends, ends[1:]))


def _linear_curve() -> PillarCurve:
    """Linear curve whose forwards at the knots after the first are FWDS."""
    averages = np.concatenate((FWDS[:1], (FWDS[:-1] + FWDS[1:]) / 2))
    return PillarCurve(
        T,
        KNOTS,
        averages,
        InterpolationMethod.PiecewiseLinear,
    )


@pytest.mark.parametrize("days", [45, 90, 135, 180, 250, 365, 500, 730])
def test_linear_integrates_forwards_linear_between_knots(days):
    curve = _linear_curve()
    d = T + timedelta(days=days)
    assert curve.get_df(d) == pytest.approx(math.exp(-_linear_integral(d)), rel=1e-13)


def test_linear_extends_the_last_forward_flat():
    curve = _linear_curve()
    beyond = KNOTS[-1] + timedelta(days=365)
    assert curve.get_df(beyond) == pytest.approx(
        curve.get_df(KNOTS[-1]) * math.exp(-FWDS[-1]), rel=1e-13
    )


def test_cubic_spline_keeps_the_knot_dfs_of_piecewise_constant_forwards():
    spline = _curve(interpolation_method=InterpolationMethod.CubicSpline)
    plain = _curve()
    assert spline.get_dfs(KNOTS) == pytest.approx(plain.get_dfs(KNOTS), rel=1e-13)
    between = T + timedelta(days=250)
    assert spline.get_df(between) != pytest.approx(plain.get_df(between), rel=1e-7)


def test_validation_errors():
    with pytest.raises(ValueError, match="at least 2 segments"):
        PillarCurve(T, [T], [], InterpolationMethod.PiecewiseConstant)
    with pytest.raises(ValueError, match="at least 2 segments"):
        PillarCurve(T, KNOTS[:2], FWDS[:1], InterpolationMethod.PiecewiseConstant)
    with pytest.raises(ValueError, match="one entry per segment"):
        PillarCurve(T, KNOTS, [0.05, 0.05], InterpolationMethod.PiecewiseConstant)
    with pytest.raises(ValueError, match="strictly increasing"):
        PillarCurve(
            T,
            [T, T, T + timedelta(days=10)],
            [0.05, 0.05],
            InterpolationMethod.PiecewiseConstant,
        )
    with pytest.raises(ValueError, match="before the curve date"):
        PillarCurve(
            T,
            [T - timedelta(days=1), T + timedelta(days=10), T + timedelta(days=20)],
            [0.05, 0.05],
            InterpolationMethod.PiecewiseConstant,
        )
    with pytest.raises(ValueError, match="does not support interpolation 'LogLinear'"):
        PillarCurve(
            T,
            KNOTS,
            FWDS,
            "LogLinear",  # type: ignore[arg-type],
        )


# ── aging ───────────────────────────────────────────────────────────────────


def test_aged_before_the_effective_date_averages_the_spot_forward():
    effective = T + timedelta(days=3)
    knots = [effective, *KNOTS[1:]]
    curve = PillarCurve(
        T,
        knots,
        FWDS,
        InterpolationMethod.PiecewiseConstant,
        SPOT,
    )
    age = T + timedelta(days=1)
    aged = curve.get_aged_curve(age)
    assert aged.spot_rate is None and aged.knot_dates[:2] == [age, effective]
    assert aged.get_df(effective) == pytest.approx(
        curve.get_df(effective) / curve.get_df(age)
    )
