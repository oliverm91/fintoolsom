"""Market curve storage: keys, the index / discount getters, get_dfs and get_accrual."""

import math
from datetime import date, timedelta

import pytest

from fintoolsom.market import Market
from fintoolsom.market.currencies import Currency, CurrencyName
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.rates import PillarCurve, InterpolationMethod, ZeroCouponCurve
from tests.factories.curve_factory import curve_from_dfs

T = date(2024, 7, 8)
CLP, USD = Currency(CurrencyName.CLP), Currency(CurrencyName.USD)
ICP = OvernightRateIndex("ICP", currency=CLP)
SOFR = OvernightRateIndex("SOFR", currency=USD)
END = T + timedelta(days=3650)
DATES = [T + timedelta(days=n) for n in (30, 365, 1800)]


def _flat_curve(rate):
    return PillarCurve(
        T,
        [T, T + (END - T) / 2, END],
        [rate, rate],
        InterpolationMethod.PiecewiseConstant,
    )


def _market() -> Market:
    usd_icp = curve_from_dfs(T, [END], [math.exp(-0.03 * 10)])
    return Market(
        T,
        risk_free_index=ICP,
        curves={ICP: _flat_curve(0.05), SOFR: _flat_curve(0.04), USD: usd_icp},
    )


def _flat(rate):
    return [math.exp(-rate * (d - T).days / 365) for d in DATES]


def test_curves_are_stored_under_their_keys():
    assert set(_market().curves) == {ICP, SOFR, USD}


def test_discount_curve_is_the_risk_free_index_curve_for_its_currency():
    market = _market()
    assert market.get_discount_curve(CLP) is market.curves[ICP]  # its own index curve
    assert market.get_discount_curve(USD) is market.curves[USD]


def test_a_currency_without_a_curve_raises():
    with pytest.raises(KeyError, match="EUR under 'ICP'"):
        _market().get_discount_curve(Currency(CurrencyName.EUR))


def test_get_dfs_uncollateralised_same_currency_and_cross_currency_collateral():
    market = _market()
    assert market.get_dfs(USD, DATES) == pytest.approx(_flat(0.03), rel=1e-13)
    assert market.get_dfs(USD, DATES, collateral=SOFR) == pytest.approx(
        _flat(0.04), rel=1e-13
    )
    # CLP under SOFR collateral: CLP_ICP * SOFR / USD_ICP.
    expected = [a * b / c for a, b, c in zip(_flat(0.05), _flat(0.04), _flat(0.03))]
    assert market.get_dfs(CLP, DATES, collateral=SOFR) == pytest.approx(
        expected, rel=1e-13
    )
    assert market.get_df(CLP, DATES[0], collateral=SOFR) == pytest.approx(
        expected[0], rel=1e-13
    )


def test_a_bump_of_the_collateral_curve_moves_every_dependent_discount_factor():
    market = _market()
    before = market.get_dfs(CLP, DATES, collateral=SOFR)
    market.set_curve(SOFR, market.curves[SOFR].parallel_bump_bps(1))
    after = market.get_dfs(CLP, DATES, collateral=SOFR)
    taus = [(d - T).days / 365 for d in DATES]
    assert [
        math.log(b / a) / t for a, b, t in zip(after, before, taus)
    ] == pytest.approx([1e-4] * 3, abs=1e-12)


def test_accrual_is_the_index_curve_wealth_factor_minus_one():
    market = _market()
    start, end = DATES[0], DATES[1]
    expected = math.exp(0.04 * (end - start).days / 365) - 1
    assert market.get_accrual(SOFR, start, end) == pytest.approx(expected, rel=1e-13)
    assert market.get_accrual(SOFR, [start, start], [end, DATES[2]]) == pytest.approx(
        [expected, math.exp(0.04 * (DATES[2] - start).days / 365) - 1], rel=1e-13
    )
    with pytest.raises(KeyError, match="TAB"):
        market.get_accrual(OvernightRateIndex("TAB", currency=CLP), start, end)


def test_index_curve_lookup_and_missing_curves():
    market = _market()
    assert market.get_index_curve(SOFR) is market.curves[SOFR]
    with pytest.raises(KeyError, match="TAB"):
        market.get_index_curve(OvernightRateIndex("TAB", currency=CLP))


def test_legacy_curves_are_rejected():
    with pytest.warns(DeprecationWarning, match="legacy"):
        zcc = ZeroCouponCurve(T, date_dfs=[(END, 0.6)])
    with pytest.raises(TypeError, match="use PillarCurve"):
        _market().set_curve(SOFR, zcc)


def test_curves_are_keyed_by_index_or_currency():
    with pytest.raises(TypeError, match="keyed by Index or Currency"):
        Market(T, risk_free_index=ICP, curves={"SOFR": _flat_curve(0.05)})


def test_the_risk_free_currency_is_not_a_key():
    with pytest.raises(ValueError, match="store the curve under the index"):
        Market(T, risk_free_index=ICP, curves={CLP: _flat_curve(0.05)})
