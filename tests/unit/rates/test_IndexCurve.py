"""IndexCurve: forwards as state, dfs derived, spot segment, bumps, validation."""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.market.currencies import Currency, CurrencyName
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.rates import (
    ExponentialInterestConvention,
    IndexCurve,
    InterpolationMethod,
    LinearInterestConvention,
    Rate,
    RateConvention,
    ZeroCouponCurve,
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


def _curve(**kwargs) -> IndexCurve:
    return IndexCurve(T, SOFR, KNOTS, FWDS, **kwargs)


def _tau(d: date) -> float:
    return (d - T).days / 365


def test_id_is_the_index_name_and_df_is_one_at_the_curve_date():
    curve = _curve()
    assert curve.get_id() == "SOFR" and curve.get_df(T) == pytest.approx(1.0, abs=1e-15)
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
    curve = IndexCurve(T, SOFR, knots, FWDS, SPOT)
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
    curve = IndexCurve(T, SOFR, [effective, *KNOTS[1:]], FWDS)
    assert curve.get_df(effective) == pytest.approx(
        math.exp(-FWDS[0] * _tau(effective)), rel=1e-14
    )


@pytest.mark.parametrize("spot", [None, SPOT], ids=["no-spot", "spot"])
def test_parallel_bump_shifts_every_continuous_zero_rate_and_returns_a_copy(spot):
    effective = T + timedelta(days=3)
    curve = IndexCurve(T, SOFR, [effective, *KNOTS[1:]], FWDS, spot)
    bumped = curve.parallel_bump_bps(1)
    dates = [effective, *(T + timedelta(days=n) for n in (50, 300, 900))]
    shift = [
        (math.log(curve.get_df(d)) - math.log(bumped.get_df(d))) / _tau(d)
        for d in dates
    ]
    assert shift == pytest.approx([1e-4] * len(dates), abs=1e-12)
    assert curve.forward_rates == pytest.approx(FWDS)  # the original is untouched


def test_matches_a_log_linear_zero_coupon_curve_on_the_same_pillars():
    zcc = ZeroCouponCurve(T, date_dfs=[(k, _curve().get_df(k)) for k in KNOTS[1:]])
    converted = IndexCurve.from_zero_coupon_curve(zcc, SOFR)
    dates = [T + timedelta(days=n) for n in (1, 45, 200, 500, 729)]
    assert converted.get_dfs(dates) == pytest.approx(zcc.get_dfs(dates), rel=1e-12)
    assert converted.forward_rates == pytest.approx(FWDS, rel=1e-10)


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


@pytest.mark.parametrize("days", [45, 90, 135, 180, 250, 365, 500, 730])
def test_linear_integrates_forwards_linear_between_knots(days):
    curve = _curve(interpolation_method=InterpolationMethod.Linear)
    d = T + timedelta(days=days)
    assert curve.get_df(d) == pytest.approx(math.exp(-_linear_integral(d)), rel=1e-13)


def test_linear_extends_the_last_forward_flat():
    curve = _curve(interpolation_method=InterpolationMethod.Linear)
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


def test_cubic_spline_forward_joins_the_flat_extrapolation_smoothly():
    curve = _curve(interpolation_method=InterpolationMethod.CubicSpline)
    last, day = KNOTS[-1], timedelta(days=1)
    before = math.log(curve.get_df(last - day)) - math.log(curve.get_df(last))
    after = math.log(curve.get_df(last)) - math.log(curve.get_df(last + day))
    assert before == pytest.approx(after, abs=1e-8)


@pytest.mark.parametrize(
    "method", [InterpolationMethod.Linear, InterpolationMethod.CubicSpline]
)
def test_parallel_bump_and_copy_keep_the_shape(method):
    curve = _curve(interpolation_method=method)
    bumped = curve.parallel_bump_bps(1)
    dates = [T + timedelta(days=n) for n in (1, 45, 200, 500, 729, 1000)]
    shift = (np.log(curve.get_dfs(dates)) - np.log(bumped.get_dfs(dates))) / [
        _tau(d) for d in dates
    ]
    assert shift == pytest.approx([1e-4] * len(dates), abs=1e-12)
    assert bumped.interpolation_method is method
    assert curve.copy().get_dfs(dates) == pytest.approx(curve.get_dfs(dates))


@pytest.mark.parametrize(
    "method", [InterpolationMethod.Linear, InterpolationMethod.CubicSpline]
)
def test_a_single_segment_is_a_constant_forward(method):
    one = IndexCurve(T, SOFR, KNOTS[:2], FWDS[:1], interpolation_method=method)
    beyond = KNOTS[1] + timedelta(days=100)
    assert one.get_df(beyond) == pytest.approx(
        math.exp(-FWDS[0] * _tau(beyond)), rel=1e-13
    )


def test_validation_errors():
    with pytest.raises(ValueError, match="at least 2 knots"):
        IndexCurve(T, SOFR, [T], [])
    with pytest.raises(ValueError, match="one entry per segment"):
        IndexCurve(T, SOFR, KNOTS, [0.05, 0.05])
    with pytest.raises(ValueError, match="strictly increasing"):
        IndexCurve(T, SOFR, [T, T, T + timedelta(days=10)], [0.05, 0.05])
    with pytest.raises(ValueError, match="before the curve date"):
        IndexCurve(T, SOFR, [T - timedelta(days=1), T + timedelta(days=10)], [0.05])
    with pytest.raises(ValueError, match="does not support interpolation 'LogLinear'"):
        IndexCurve(
            T, SOFR, KNOTS, FWDS, interpolation_method=InterpolationMethod.LogLinear
        )
