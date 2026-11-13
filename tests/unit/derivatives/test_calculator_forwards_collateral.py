"""Forwards and NDFs under a CSA: collateral changes discounting, not the projected FX."""

from datetime import date
from typing import Any

import pytest

from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.forwards.forwards import NDF, Forward
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair
from tests.factories.swap_factory import (
    ESTR,
    FX_TO_CLP,
    ICP,
    SOFR,
    VAL_DATE,
    Ccy,
    _rate,
    adj_convention,
    build_market,
)

USD, CLP = Currency.USD, Currency.CLP
USD_CLP = CurrencyPair(USD, CLP)
SPOT = FX_TO_CLP[Ccy.USD]
NOTIONAL = 1_000_000.0
STRIKE = 980.0
PAYMENT = date(2025, 7, 10)
FIXING = date(2025, 7, 8)

COLLATERALS = [None, ICP, SOFR, ESTR]


@pytest.fixture(scope="module")
def market():
    return build_market()


def _fair_strike(market, d: date) -> float:
    """Forward FX on the funding curves: CSA-free by construction."""
    disc = market.get_discount_curve
    return SPOT * disc(USD).get_df(d) / disc(CLP).get_df(d)


def test_uncollateralised_forward_discounts_on_the_funding_curves(market):
    fwd = Forward(NOTIONAL, STRIKE, PAYMENT, is_buy=True, currency_pair=USD_CLP)
    disc = market.get_discount_curve
    expected = NOTIONAL * (
        SPOT * disc(USD).get_df(PAYMENT) - STRIKE * disc(CLP).get_df(PAYMENT)
    )
    assert Calculator.get_forward_mtm(fwd, market, CLP) == pytest.approx(
        expected, rel=1e-12
    )


def test_sofr_collateralised_forward_discounts_each_leg_under_the_csa(market):
    fwd = Forward(
        NOTIONAL,
        STRIKE,
        PAYMENT,
        is_buy=True,
        currency_pair=USD_CLP,
        collateral_index=SOFR,
    )
    disc = market.get_discount_curve
    df_sofr = market.get_index_curve(SOFR).get_df(PAYMENT)
    df_clp_sofr = disc(CLP).get_df(PAYMENT) * df_sofr / disc(USD).get_df(PAYMENT)
    expected = NOTIONAL * (SPOT * df_sofr - STRIKE * df_clp_sofr)
    mtm = Calculator.get_forward_mtm(fwd, market, CLP)
    assert mtm == pytest.approx(expected, rel=1e-12)
    uncollateralised = Calculator.get_forward_mtm(
        Forward(NOTIONAL, STRIKE, PAYMENT, is_buy=True, currency_pair=USD_CLP),
        market,
        CLP,
    )
    assert abs(mtm - uncollateralised) > 1.0


@pytest.mark.parametrize("collateral", COLLATERALS, ids=str)
def test_fair_strike_does_not_depend_on_the_collateral(market, collateral):
    fwd = Forward(
        NOTIONAL,
        _fair_strike(market, PAYMENT),
        PAYMENT,
        is_buy=True,
        currency_pair=USD_CLP,
        collateral_index=collateral,
    )
    assert Calculator.get_forward_mtm(fwd, market, CLP) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("collateral", COLLATERALS, ids=str)
def test_ndf_projects_fx_on_funding_curves_and_discounts_under_the_csa(
    market, collateral
):
    ndf = NDF(
        NOTIONAL,
        STRIKE,
        PAYMENT,
        is_buy=False,
        currency_pair=USD_CLP,
        fixing_date=FIXING,
        collateral_index=collateral,
    )
    expected = (
        -NOTIONAL
        * (_fair_strike(market, FIXING) - STRIKE)
        * market.get_df(CLP, PAYMENT, collateral=collateral)
    )
    assert Calculator.get_ndf_mtm(ndf, market, CLP) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("collateral", COLLATERALS, ids=str)
def test_ndf_fixing_on_payment_matches_forward_under_the_same_csa(market, collateral):
    kwargs: dict[str, Any] = dict(
        is_buy=True, currency_pair=USD_CLP, collateral_index=collateral
    )
    fwd = Forward(NOTIONAL, STRIKE, PAYMENT, **kwargs)
    ndf = NDF(NOTIONAL, STRIKE, PAYMENT, fixing_date=PAYMENT, **kwargs)
    assert Calculator.get_ndf_mtm(ndf, market, CLP) == pytest.approx(
        Calculator.get_forward_mtm(fwd, market, CLP), rel=1e-12
    )


def _zero_coupon_leg(currency: Currency, notional: float):
    """Bullet 0% fixed leg: a single final exchange of `notional`."""
    conv = adj_convention(Ccy.CLP)
    return builders.fixed_leg(
        rate=_rate(0.0),
        currency=currency,
        notional=notional,
        start_date=VAL_DATE,
        term=Term(1, TermUnit.Y, conv),
        frequency="ANNUAL",
        adj_convention=conv,
    )


@pytest.mark.parametrize("collateral", COLLATERALS, ids=str)
def test_forward_matches_the_equivalent_deliverable_swap(market, collateral):
    usd = _zero_coupon_leg(USD, NOTIONAL)
    clp = _zero_coupon_leg(CLP, NOTIONAL * STRIKE)
    assert usd.payment_dates == clp.payment_dates
    swap = Swap(
        receive_leg=usd, pay_leg=clp, is_deliverable=True, collateral_index=collateral
    )
    fwd = Forward(
        NOTIONAL,
        STRIKE,
        usd.payment_dates[-1],
        is_buy=True,
        currency_pair=USD_CLP,
        collateral_index=collateral,
    )
    assert Calculator.get_forward_mtm(fwd, market, CLP) == pytest.approx(
        swap.valuate(market, CLP), rel=1e-10
    )


def test_ndf_matches_the_equivalent_non_deliverable_swap(market):
    usd = _zero_coupon_leg(USD, NOTIONAL)
    clp = _zero_coupon_leg(CLP, NOTIONAL * STRIKE)
    swap = Swap(
        receive_leg=usd, pay_leg=clp, payment_currency=CLP, collateral_index=SOFR
    )
    pay = usd.payment_dates[-1]
    ndf = NDF(
        NOTIONAL,
        STRIKE,
        pay,
        is_buy=True,
        currency_pair=USD_CLP,
        fixing_date=usd.fx_fixing_dates[-1],
        collateral_index=SOFR,
    )
    assert Calculator.get_ndf_mtm(ndf, market, CLP) == pytest.approx(
        swap.valuate(market, CLP), rel=1e-10
    )


def test_ndf_positional_fixing_date_still_works():
    ndf = NDF(1.0, 1.0, PAYMENT, True, USD_CLP, FIXING)
    assert ndf.fixing_date == FIXING
    assert ndf.collateral_index is None
