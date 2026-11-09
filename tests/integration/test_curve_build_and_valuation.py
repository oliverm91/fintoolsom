"""End to end: bootstrap curves from a real quote set, then value the quoted swaps.

1. Every quoted instrument, valued on the curves built from it, has MTM ~ 0.
2. Moving a quoted fixed rate (or basis spread) by 1bp changes the MTM by roughly the
   DV01 of a fixed leg on that schedule: ``-notional * duration / 10_000``.
"""

from dataclasses import replace

import pytest

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps import LegSide
from fintoolsom.market import (
    CrossCurrencyFloatFloatQuote,
    Currency,
    ForwardPointsQuote,
    IRSQuote,
)
from fintoolsom.rates import Rate
from tests.factories.quote_factory import USDCLP_SPOT, built_market

CLP = Currency.CLP
QUOTE_NOTIONAL_CLP = 100 * USDCLP_SPOT  # quotes are on 100 units of the base currency
# 1M-1Y cross-currency quotes fall inside the range the forward points already pin, so
# the bootstrap cannot fit them exactly: the market data is ~10bp inconsistent there.
REDUNDANT_XCCY_TERMS = {"1M", "3M", "6M", "9M", "1Y"}


@pytest.fixture(scope="module")
def built():
    return built_market()


def _instrument(quote, market):
    if isinstance(quote, (CrossCurrencyFloatFloatQuote, ForwardPointsQuote)):
        return quote.get_instrument(market)
    return quote.get_instrument()


def _fx(currency: Currency) -> float:
    return USDCLP_SPOT if currency == Currency.USD else 1.0


def _irs(quotes, currency: Currency, term: str) -> IRSQuote:
    return next(
        q
        for q in quotes
        if isinstance(q, IRSQuote)
        and q.fixed_leg.currency == currency
        and str(q.term) == term
    )


def _xccy(quotes, term: str) -> CrossCurrencyFloatFloatQuote:
    return next(
        q
        for q in quotes
        if isinstance(q, CrossCurrencyFloatFloatQuote) and str(q.term) == term
    )


def _bumped(rate: Rate, bps: float = 1.0) -> Rate:
    return Rate(rate.rate_convention.copy(), rate.rate + bps / 10_000)


def test_every_quote_reprices_on_the_curves_built_from_it(built):
    market, quotes = built
    assert len(quotes) == 73
    for q in quotes:
        mtm = Calculator.valuate(_instrument(q, market), market, CLP)
        if (
            isinstance(q, CrossCurrencyFloatFloatQuote)
            and str(q.term) in REDUNDANT_XCCY_TERMS
        ):
            assert abs(mtm) < 1e-3 * QUOTE_NOTIONAL_CLP, f"XCCY {q.term}: {mtm}"
        else:
            assert abs(mtm) < 1e-8 * QUOTE_NOTIONAL_CLP, (
                f"{type(q).__name__} {q.term}: {mtm}"
            )


@pytest.mark.parametrize("currency", [Currency.CLP, Currency.USD])
@pytest.mark.parametrize("term", ["2Y", "5Y", "10Y", "20Y"])
def test_one_bp_on_the_fixed_rate_moves_mtm_by_the_fixed_leg_dv01(
    built, currency, term
):
    market, quotes = built
    quote = _irs(quotes, currency, term)
    swap = quote.get_instrument()
    fixed = swap.receive_leg  # the quotes receive the fixed leg
    base = Calculator.valuate(swap, market, CLP)
    bumped = replace(
        quote, fixed_leg=replace(quote.fixed_leg, rate=_bumped(quote.fixed_leg.rate))
    )
    change = Calculator.valuate(bumped.get_instrument(), market, CLP) - base

    dv01 = Calculator.get_leg_dv01(swap, LegSide.RECEIVE, market, CLP)
    assert (
        dv01 < 0 < change
    )  # receiving 1bp more is worth what 1bp lower rates would be
    assert change == pytest.approx(-dv01, rel=0.06)
    # Back-of-the-envelope version, with the notional in place of the leg PV.
    rough = (
        fixed.residuals[0]
        * _fx(currency)
        * Calculator.get_leg_duration(swap, LegSide.RECEIVE, market)
        / 10_000
    )
    assert change == pytest.approx(rough, rel=0.06)


@pytest.mark.parametrize("term", ["2Y", "5Y", "10Y"])
def test_one_bp_on_the_basis_spread_moves_mtm_by_a_fixed_leg_dv01(built, term):
    # The spread is quoted on the paid USD SOFR leg: 1bp more is a paid 1bp annuity,
    # i.e. the DV01 of the same-tenor USD fixed leg, scaled to the leg notional.
    market, quotes = built
    quote = _xccy(quotes, term)
    swap = quote.get_instrument(market)
    assert quote.pay_leg.currency == Currency.USD
    base = Calculator.valuate(swap, market, CLP)
    bumped = replace(
        quote, pay_leg=replace(quote.pay_leg, spread=_bumped(quote.pay_leg.spread))
    )
    change = Calculator.valuate(bumped.get_instrument(market), market, CLP) - base

    irs = _irs(quotes, Currency.USD, term).get_instrument()
    fixed = irs.receive_leg
    dv01 = (
        Calculator.get_leg_dv01(irs, LegSide.RECEIVE, market, CLP)
        * swap.pay_leg.residuals[0]
        / fixed.residuals[0]
    )
    assert change < 0
    assert change == pytest.approx(dv01, rel=0.05)
