"""DiscountCurve: dfs at pillars, interpolation, flat-forward extrapolation, bumps."""

import math
from datetime import date, timedelta

import pytest

from fintoolsom.market.currencies import Currency, CurrencyName
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.rates import DiscountCurve, InterpolationMethod, ZeroCouponCurve

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


def _curve(method=InterpolationMethod.LogLinear) -> DiscountCurve:
    return DiscountCurve(T, USD, ICP, PILLARS, DFS, method)


def test_id_names_the_currency_and_the_risk_free_index():
    assert _curve().get_id() == "USD_ICP" and _curve().last_date == PILLARS[-1]


def test_a_currency_with_its_own_risk_free_index_has_no_discount_curve():
    with pytest.raises(ValueError, match="IndexCurve"):
        DiscountCurve(T, CLP, ICP, PILLARS, DFS)


@pytest.mark.parametrize(
    "method",
    [
        InterpolationMethod.LogLinear,
        InterpolationMethod.HermiteCubicSpline,
        InterpolationMethod.CubicSpline,
    ],
)
def test_reprices_its_pillars_and_is_one_at_the_curve_date(method):
    curve = _curve(method)
    assert curve.get_dfs(PILLARS) == pytest.approx(DFS, rel=1e-14)
    assert curve.get_df(T) == pytest.approx(1.0, abs=1e-15)


def test_log_linear_matches_the_legacy_curve_on_the_same_pillars():
    zcc = ZeroCouponCurve(T, date_dfs=list(zip(PILLARS, DFS)))
    converted = DiscountCurve.from_zero_coupon_curve(zcc, USD, ICP)
    assert converted.get_dfs(INSIDE) == pytest.approx(zcc.get_dfs(INSIDE), rel=1e-12)
    assert converted.interpolation_method is InterpolationMethod.LogLinear


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
    assert list(curve.dfs) == pytest.approx(DFS)


def test_cubic_spline_forward_joins_the_flat_extrapolation_smoothly():
    curve = _curve(InterpolationMethod.CubicSpline)
    last, day = PILLARS[-1], timedelta(days=1)
    before = math.log(curve.get_df(last - day)) - math.log(curve.get_df(last))
    after = math.log(curve.get_df(last)) - math.log(curve.get_df(last + day))
    assert before == pytest.approx(after, abs=1e-8)


def test_cubic_spline_differs_from_log_linear_between_pillars():
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


def test_linear_is_only_for_index_curves():
    with pytest.raises(ValueError, match="does not support interpolation 'Linear'"):
        DiscountCurve(T, USD, ICP, PILLARS, DFS, InterpolationMethod.Linear)


def test_pillars_are_sorted_and_validated():
    shuffled = DiscountCurve(T, USD, ICP, PILLARS[::-1], DFS[::-1])
    assert shuffled.pillar_dates == PILLARS
    with pytest.raises(ValueError, match="after the curve date"):
        DiscountCurve(T, USD, ICP, [T], [1.0])
    with pytest.raises(ValueError, match="Duplicate"):
        DiscountCurve(T, USD, ICP, [PILLARS[0], PILLARS[0]], [0.99, 0.99])
    with pytest.raises(
        ValueError, match="does not support interpolation 'PiecewiseConstant'"
    ):
        DiscountCurve(T, USD, ICP, PILLARS, DFS, InterpolationMethod.PiecewiseConstant)
