"""DiscountCurve: forwards as parameters, dfs as a result, extrapolation, bumps."""

import math
from datetime import date, timedelta

import pytest

from fintoolsom.market.currencies import Currency, CurrencyName
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.rates import DiscountCurve, InterpolationMethod, ZeroCouponCurve
from tests.factories.curve_factory import discount_curve_from_dfs
from tests.factories.curve_factory import discount_curve_from_zero_coupon_curve

T = date(2024, 7, 8)
USD = Currency(CurrencyName.USD)
CLP = Currency(CurrencyName.CLP)
ICP = OvernightRateIndex("ICP", currency=CLP)
PILLARS = [T + timedelta(days=n) for n in (1, 30, 182, 365, 730, 1825, 3650)]
# A gently humped curve so interpolation choices show.
DFS = [
    math.exp(-(0.04 + 0.01 * math.sin(i)) * (p - T).days / 365)
    for i, p in enumerate(PILLARS)
]
INSIDE = [T + timedelta(days=n) for n in (1, 10, 95, 400, 1100, 2500, 3650)]


def _curve(method=InterpolationMethod.PiecewiseConstant) -> DiscountCurve:
    return discount_curve_from_dfs(T, USD, ICP, PILLARS, DFS, method)


def test_id_names_the_currency_and_the_risk_free_index():
    assert _curve().get_id() == "USD_ICP" and _curve().last_date == PILLARS[-1]


def test_a_currency_with_its_own_risk_free_index_has_no_discount_curve():
    with pytest.raises(ValueError, match="IndexCurve"):
        discount_curve_from_dfs(T, CLP, ICP, PILLARS, DFS)


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


def test_piecewise_constant_matches_a_log_linear_legacy_curve():
    zcc = ZeroCouponCurve(T, date_dfs=list(zip(PILLARS, DFS)))
    converted = discount_curve_from_zero_coupon_curve(zcc, USD, ICP)
    assert converted.get_dfs(INSIDE) == pytest.approx(zcc.get_dfs(INSIDE), rel=1e-12)
    assert converted.interpolation_method is InterpolationMethod.PiecewiseConstant


def test_extrapolates_the_last_forward_flat():
    curve = _curve()
    tau = [(d - T).days / 365 for d in PILLARS[-2:]]
    last_forward = -(math.log(DFS[-1]) - math.log(DFS[-2])) / (tau[1] - tau[0])
    beyond = PILLARS[-1] + timedelta(days=365)
    assert curve.get_df(beyond) == pytest.approx(
        DFS[-1] * math.exp(-last_forward * 1.0), rel=1e-13
    )


def test_parallel_bump_shifts_every_continuous_zero_rate_and_returns_a_copy():
    curve = _curve()
    bumped = curve.parallel_bump_bps(2)
    for d in INSIDE + [PILLARS[-1] + timedelta(days=500)]:
        tau = (d - T).days / 365
        assert (
            math.log(curve.get_df(d)) - math.log(bumped.get_df(d))
        ) / tau == pytest.approx(2e-4, abs=1e-12)
    assert curve.get_dfs(PILLARS) == pytest.approx(DFS)


def test_cubic_spline_forward_joins_the_flat_extrapolation_smoothly():
    curve = _curve(InterpolationMethod.CubicSpline)
    last, day = PILLARS[-1], timedelta(days=1)
    before = math.log(curve.get_df(last - day)) - math.log(curve.get_df(last))
    after = math.log(curve.get_df(last)) - math.log(curve.get_df(last + day))
    assert before == pytest.approx(after, abs=1e-8)


def test_cubic_spline_differs_from_piecewise_constant_between_pillars():
    between = T + timedelta(days=2500)
    assert _curve(InterpolationMethod.CubicSpline).get_df(between) != pytest.approx(
        _curve().get_df(between), rel=1e-6
    )


def test_cubic_spline_bump_shifts_every_zero_rate():
    curve = _curve(InterpolationMethod.CubicSpline)
    bumped = curve.parallel_bump_bps(2)
    assert bumped.interpolation_method is InterpolationMethod.CubicSpline
    for d in INSIDE + [PILLARS[-1] + timedelta(days=500)]:
        tau = (d - T).days / 365
        assert (
            math.log(curve.get_df(d)) - math.log(bumped.get_df(d))
        ) / tau == pytest.approx(2e-4, abs=1e-11)


def test_knots_and_forwards_are_validated():
    with pytest.raises(ValueError, match="strictly increasing"):
        DiscountCurve(T, USD, ICP, [T, PILLARS[1], PILLARS[0]], [0.05, 0.05])
    with pytest.raises(ValueError, match="one entry per segment"):
        DiscountCurve(T, USD, ICP, [T, PILLARS[0], PILLARS[1]], [0.05])
    with pytest.raises(ValueError, match="at least 2 segments"):
        DiscountCurve(T, USD, ICP, [T, PILLARS[0]], [0.05])
    with pytest.raises(ValueError, match="does not support interpolation 'LogLinear'"):
        DiscountCurve(
            T, USD, ICP, [T, PILLARS[0]], [0.05], InterpolationMethod.LogLinear
        )


# ── aging ───────────────────────────────────────────────────────────────────
AGE = T + timedelta(days=100)


@pytest.mark.parametrize(
    "method",
    [
        InterpolationMethod.PiecewiseConstant,
        InterpolationMethod.HermiteCubicSpline,
        InterpolationMethod.CubicSpline,
    ],
)
def test_aged_curve_discounts_with_forward_factors_at_the_pillars(method):
    curve = _curve(method)
    aged = curve.get_aged_curve(AGE)
    kept = [p for p in PILLARS if p > AGE]
    assert aged.curve_date == AGE and aged.knot_dates == [AGE, *kept]
    assert aged.get_df(AGE) == pytest.approx(1.0)
    assert aged.get_dfs(kept) == pytest.approx(curve.get_dfs(kept) / curve.get_df(AGE))


def test_piecewise_constant_aged_curve_is_exact_between_pillars_too():
    curve = _curve()
    aged = curve.get_aged_curve(AGE)
    dates = [AGE + timedelta(days=n) for n in (1, 50, 200, 900, 3000)]
    assert aged.get_dfs(dates) == pytest.approx(
        curve.get_dfs(dates) / curve.get_df(AGE)
    )


@pytest.mark.parametrize(
    "bad", [T, T - timedelta(days=1), PILLARS[-1], PILLARS[-1] + timedelta(days=1)]
)
def test_aging_outside_the_curve_raises(bad):
    with pytest.raises(ValueError):
        _curve().get_aged_curve(bad)


def test_df_t1_times_aged_df_t2_is_df_t2_and_aged_df_is_the_forward_df():
    curve = _curve()  # PiecewiseConstant: exact between pillars too
    t1 = AGE
    aged = curve.get_aged_curve(t1)
    for t2 in [t1 + timedelta(days=n) for n in (1, 50, 200, 900, 3000)]:
        assert curve.get_df(t1) * aged.get_df(t2) == pytest.approx(
            curve.get_df(t2), rel=1e-12
        )
        assert aged.get_df(t2) == pytest.approx(curve.get_df_fwd(t1, t2), rel=1e-12)


# ── forward rates are the parameters ────────────────────────────────────────


def test_is_built_from_forward_rates_and_dfs_are_their_result():
    knots = [T, T + timedelta(days=365), T + timedelta(days=730)]
    curve = DiscountCurve(T, USD, ICP, knots, [0.05, 0.03])
    assert curve.last_date == knots[-1]
    assert curve.get_dfs(knots) == pytest.approx(
        [1.0, math.exp(-0.05), math.exp(-0.08)], rel=1e-14
    )


def test_knots_and_forwards_fully_define_the_curve():
    curve = _curve()
    rebuilt = DiscountCurve(T, USD, ICP, curve.knot_dates, curve.forward_rates)
    assert rebuilt.get_dfs(PILLARS) == pytest.approx(DFS, rel=1e-14)
    assert not hasattr(curve, "dfs") and not hasattr(curve, "pillars")


def test_the_first_knot_must_be_the_curve_date():
    late = [T + timedelta(days=1), T + timedelta(days=90), T + timedelta(days=365)]
    with pytest.raises(ValueError, match="must be the curve date"):
        DiscountCurve(T, USD, ICP, late, [0.05, 0.05])
