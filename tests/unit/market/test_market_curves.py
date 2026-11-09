"""Market curve storage: ids, the index / discount getters and get_dfs."""

import math
from datetime import date, timedelta

import pytest

from fintoolsom.market import Market
from fintoolsom.market.currencies import Currency, CurrencyName
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.rates import DiscountCurve, IndexCurve, ZeroCouponCurve

T = date(2024, 7, 8)
CLP, USD = Currency(CurrencyName.CLP), Currency(CurrencyName.USD)
ICP = OvernightRateIndex("ICP", currency=CLP)
SOFR = OvernightRateIndex("SOFR", currency=USD)
END = T + timedelta(days=3650)
DATES = [T + timedelta(days=n) for n in (30, 365, 1800)]


def _index_curve(index, rate):
    return IndexCurve(T, index, [T, END], [rate])


def _market() -> Market:
    usd_icp = DiscountCurve(T, USD, ICP, [END], [math.exp(-0.03 * 10)])
    return Market(
        T,
        risk_free_index=ICP,
        curves=[_index_curve(ICP, 0.05), _index_curve(SOFR, 0.04), usd_icp],
    )


def _flat(rate):
    return [math.exp(-rate * (d - T).days / 365) for d in DATES]


def test_curves_are_stored_under_their_ids():
    assert set(_market().curves) == {"ICP", "SOFR", "USD_ICP"}


def test_discount_curve_defaults_to_the_markets_risk_free_index():
    market = _market()
    assert market.get_discount_curve(CLP) is market.curves["ICP"]  # its own index curve
    assert market.get_discount_curve(USD) is market.curves["USD_ICP"]
    assert market.get_discount_curve(USD, risk_free_index=SOFR) is market.curves["SOFR"]


def test_a_numeraire_not_built_raises():
    with pytest.raises(KeyError, match="CLP_SOFR"):
        _market().get_discount_curve(CLP, risk_free_index=SOFR)


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
    market.set_curve(market.curves["SOFR"].parallel_bump_bps(1))
    after = market.get_dfs(CLP, DATES, collateral=SOFR)
    taus = [(d - T).days / 365 for d in DATES]
    assert [
        math.log(b / a) / t for a, b, t in zip(after, before, taus)
    ] == pytest.approx([1e-4] * 3, abs=1e-12)


def test_index_curve_lookup_and_missing_curves():
    market = _market()
    assert market.get_index_curve(SOFR) is market.curves["SOFR"]
    with pytest.raises(KeyError, match="TAB"):
        market.get_index_curve(OvernightRateIndex("TAB", currency=CLP))


def test_legacy_curves_are_rejected():
    with pytest.warns(DeprecationWarning, match="legacy"):
        zcc = ZeroCouponCurve(T, date_dfs=[(END, 0.6)])
    with pytest.raises(TypeError, match="from_zero_coupon_curve"):
        _market().set_curve(zcc)


def test_a_curve_under_a_foreign_key_is_rejected():
    with pytest.raises(ValueError, match="has id 'ICP'"):
        Market(T, risk_free_index=ICP, curves={"SOFR": _index_curve(ICP, 0.05)})
