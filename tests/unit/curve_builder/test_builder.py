"""End-to-end behaviour of build_curves and the _CurveBuilder orchestrator."""
from datetime import date
import pytest
from fintoolsom.curve_builder import InsufficientQuotesError, RepricingError, build_curves
from fintoolsom.curve_builder._keys import _curves_needed
from fintoolsom.curve_builder.builder import _CurveBuilder
from fintoolsom.rates import InterpolationMethod, ProjectionInterpolationMethod
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from fintoolsom.market.quotes import ForwardPriceQuote
from fintoolsom.rates import ZeroCouponCurve



USDCLP = CurrencyPair(Currency.USD, Currency.CLP)


RISKLESS = OvernightRateIndex("OIS", currency=Currency.CLP)  # funds CLP; USD has its own curve


def _market(t: date, spot: float, usd_df: float, payment_date: date) -> Market:
    market = Market(
        t=t,
        riskless_index=RISKLESS,
        curves={Currency.USD: ZeroCouponCurve(t, date_dfs=[(payment_date, usd_df)])},
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

    build_curves([quote], market)

    assert RISKLESS in market.curves
    solved_df = market.curves[RISKLESS].get_df(payment_date)
    assert solved_df == pytest.approx(target_clp_df, abs=1e-6)
    # USD curve, already fully seeded at this maturity, must be left untouched.
    assert market.curves[Currency.USD].get_df(payment_date) == pytest.approx(usd_df)


def test_build_curves_raises_insufficient_quotes_when_under_determined():
    t = date(2024, 1, 2)
    payment_date = date(2024, 7, 2)
    market = Market(t=t, riskless_index=RISKLESS)
    market.add_fx_rate(t, FX_Rate(USDCLP, 900.0))
    quote = ForwardPriceQuote(
        currency_pair=USDCLP, value=928.0, is_buy=True,
        payment_date=payment_date, quote_date=t,
    )
    # Neither USD nor CLP curve exists yet: 1 quote, 2 unknown pillars.
    with pytest.raises(InsufficientQuotesError):
        build_curves([quote], market)


def test_build_curves_raises_when_quote_date_mismatches_market_t():
    t = date(2024, 1, 2)
    market = Market(t=date(2024, 1, 3), riskless_index=RISKLESS)
    quote = ForwardPriceQuote(
        currency_pair=USDCLP, value=900.0, is_buy=True,
        payment_date=date(2024, 7, 2), quote_date=t,
    )
    with pytest.raises(ValueError):
        build_curves([quote], market)


def test_curves_needed_forward_quote_does_not_crash_on_missing_collateral_index():
    # ForwardPriceQuote has no collateral_index attribute at all; _curves_needed
    # must not assume every quote type carries one.
    quote = ForwardPriceQuote(
        currency_pair=USDCLP, value=900.0, is_buy=True,
        payment_date=date(2024, 7, 2), quote_date=date(2024, 1, 2),
    )
    keys = _curves_needed(quote, RISKLESS)
    assert keys == frozenset({Currency.USD, RISKLESS})


def _builder(market: Market, quotes=()) -> _CurveBuilder:
    return _CurveBuilder(
        list(quotes), market, InterpolationMethod.LogLinear, ProjectionInterpolationMethod.PiecewiseConstant
    )


def _forward_case():
    t, payment_date, spot, usd_df, target_clp_df = date(2024, 1, 2), date(2024, 7, 2), 900.0, 0.98, 0.95
    market = _market(t, spot, usd_df, payment_date)
    quote = ForwardPriceQuote(
        currency_pair=USDCLP, value=spot * usd_df / target_clp_df, is_buy=True,
        payment_date=payment_date, quote_date=t,
    )
    return market, quote


def test_quote_falls_back_to_the_curve_it_can_still_pin():
    # USD funding (the forward's preferred curve) already covers the maturity: the
    # forward's pillar goes to the unbuilt CLP curve instead.
    market, quote = _forward_case()
    builder = _builder(market, [quote])
    assert builder.plan.candidates[quote] == (Currency.USD, RISKLESS)
    assert builder.plan.assign_pillars([quote.get_instrument()], [builder.plan.candidates[quote]]) == [RISKLESS]


def test_every_pinning_quote_is_checked_after_the_build():
    market, quote = _forward_case()
    builder = _builder(market, [quote])
    builder.build()
    assert [i for i, _ in builder._pinning] == [quote.get_instrument()]


def test_a_quote_moved_off_par_after_its_solve_raises():
    market, quote = _forward_case()
    builder = _builder(market, [quote])
    builder.build()
    # A later solve moving the curve the forward was fitted on: simulate it.
    market.curves[RISKLESS] = ZeroCouponCurve(market.t, date_dfs=[(date(2024, 7, 2), 0.90)])
    with pytest.raises(RepricingError, match="no longer reprice"):
        builder._check_repricing()
