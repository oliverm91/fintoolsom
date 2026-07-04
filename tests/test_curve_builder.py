from datetime import date

import pytest

from fintoolsom.curve_builder.builder import (
    InsufficientQuotesError,
    build_curves,
    _curves_needed,
)
from fintoolsom.market import Currency, Market, RateIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from fintoolsom.market.quotes import ForwardPriceQuote
from fintoolsom.rates import ZeroCouponCurve

USDCLP = CurrencyPair(Currency.USD, Currency.CLP)
RISKLESS = RateIndex("OIS")  # generic index, reused across currencies


def _market(t: date, spot: float, usd_df: float, payment_date: date) -> Market:
    market = Market(
        t=t,
        curves={(RISKLESS, Currency.USD): ZeroCouponCurve(t, date_dfs=[(payment_date, usd_df)])},
    )
    market.add_fx_rate(t, FX_Rate(USDCLP, spot))
    return market


def test_build_curves_solves_missing_currency_curve_from_one_forward_quote():
    t = date(2024, 1, 2)
    payment_date = date(2024, 7, 2)
    spot = 900.0
    usd_df = 0.98
    target_clp_df = 0.95
    strike = spot * usd_df / target_clp_df

    market = _market(t, spot, usd_df, payment_date)
    quote = ForwardPriceQuote(
        currency_pair=USDCLP, value=strike, is_buy=True,
        payment_date=payment_date, quote_date=t,
    )

    build_curves([quote], RISKLESS, market)

    assert (RISKLESS, Currency.CLP) in market.curves
    solved_df = market.curves[(RISKLESS, Currency.CLP)].get_df(payment_date)
    assert solved_df == pytest.approx(target_clp_df, abs=1e-6)
    # USD curve, already fully seeded at this maturity, must be left untouched.
    assert market.curves[(RISKLESS, Currency.USD)].get_df(payment_date) == pytest.approx(usd_df)


def test_build_curves_raises_insufficient_quotes_when_under_determined():
    t = date(2024, 1, 2)
    payment_date = date(2024, 7, 2)
    market = Market(t=t)
    market.add_fx_rate(t, FX_Rate(USDCLP, 900.0))
    quote = ForwardPriceQuote(
        currency_pair=USDCLP, value=928.0, is_buy=True,
        payment_date=payment_date, quote_date=t,
    )
    # Neither USD nor CLP curve exists yet: 1 quote, 2 unknown pillars.
    with pytest.raises(InsufficientQuotesError):
        build_curves([quote], RISKLESS, market)


def test_build_curves_raises_when_quote_date_mismatches_market_t():
    t = date(2024, 1, 2)
    market = Market(t=date(2024, 1, 3))
    quote = ForwardPriceQuote(
        currency_pair=USDCLP, value=900.0, is_buy=True,
        payment_date=date(2024, 7, 2), quote_date=t,
    )
    with pytest.raises(ValueError):
        build_curves([quote], RISKLESS, market)


def test_curves_needed_forward_quote_does_not_crash_on_missing_collateral_index():
    # ForwardPriceQuote has no collateral_index attribute at all; _curves_needed
    # must not assume every quote type carries one.
    quote = ForwardPriceQuote(
        currency_pair=USDCLP, value=900.0, is_buy=True,
        payment_date=date(2024, 7, 2), quote_date=date(2024, 1, 2),
    )
    keys = _curves_needed(quote, RISKLESS)
    assert keys == frozenset({(RISKLESS, Currency.USD), (RISKLESS, Currency.CLP)})
