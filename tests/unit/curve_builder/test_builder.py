from datetime import date
import pytest
from fintoolsom.curve_builder.builder import (
    InsufficientQuotesError,
    build_curves,
    _curves_needed,
)
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


def _builder(market: Market, quotes=()):
    from fintoolsom.curve_builder.builder import _CurveBuilder
    from fintoolsom.rates import InterpolationMethod, ProjectionInterpolationMethod

    return _CurveBuilder(
        list(quotes), market, InterpolationMethod.LogLinear, ProjectionInterpolationMethod.PiecewiseConstant
    )


def test_forward_seed_is_covered_interest_parity_on_the_strike():
    # Same setup as the one-forward solve: the CLP seed must already be the solution.
    t, payment_date, spot, usd_df, target_clp_df = date(2024, 1, 2), date(2024, 7, 2), 900.0, 0.98, 0.95
    market = _market(t, spot, usd_df, payment_date)
    quote = ForwardPriceQuote(
        currency_pair=USDCLP, value=spot * usd_df / target_clp_df, is_buy=True,
        payment_date=payment_date, quote_date=t,
    )
    builder = _builder(market, [quote])
    (group,) = builder.groups
    _, free_dates, seeds = builder._partition_pillars(group, [quote.get_instrument()])
    assert free_dates == {RISKLESS: [payment_date]}
    assert seeds[RISKLESS][0] == pytest.approx(target_clp_df, rel=1e-12)


def test_fixed_leg_seed_compounds_the_coupons_to_the_leg_end():
    from types import SimpleNamespace
    from fintoolsom.rates import Rate, RateConvention

    t = date(2024, 1, 2)
    starts = [date(2024, 1, 4), date(2024, 7, 4)]
    ends = [date(2024, 7, 4), date(2025, 1, 6)]
    rate = Rate(RateConvention(), 0.05)  # annual compounding, act/365
    leg = SimpleNamespace(coupons=[
        SimpleNamespace(start_date=s, end_date=e, rate=rate) for s, e in zip(starts, ends)
    ])
    z = _builder(Market(t=t, riskless_index=RISKLESS))._fixed_leg_zero(leg)
    # Spot gap and both coupons at 5% annual over [t, last end]: continuous rate log(1.05).
    import math
    assert z == pytest.approx(math.log(1.05), rel=1e-12)


def test_basis_seed_shifts_the_index_curve_by_the_spread():
    # CLP leg ICP + s on the CLP funding curve (cancels) vs USD leg SOFR flat on the
    # unknown USD funding curve: z_USD = z_SOFR - s.
    import math
    from types import SimpleNamespace
    from collections import defaultdict
    from fintoolsom.rates import Rate, RateConvention

    t, maturity = date(2024, 1, 2), date(2026, 1, 2)
    sofr = OvernightRateIndex("SOFR", currency=Currency.USD)
    z_sofr, spread = 0.045, 0.003
    tau = (maturity - t).days / 365
    market = Market(
        t=t, riskless_index=RISKLESS,
        curves={
            RISKLESS: ZeroCouponCurve(t, date_dfs=[(maturity, math.exp(-0.06 * tau))]),
            sofr: ZeroCouponCurve(t, date_dfs=[(maturity, math.exp(-z_sofr * tau))]),
        },
    )
    clp_leg = SimpleNamespace(index=RISKLESS, currency=Currency.CLP, spread=Rate(RateConvention(), spread))
    usd_leg = SimpleNamespace(index=sofr, currency=Currency.USD, spread=None)
    swap = SimpleNamespace(receive_leg=clp_leg, pay_leg=usd_leg, leg_settlement_currency=lambda leg: leg.currency)
    implied = defaultdict(dict)
    free_dates = {Currency.USD: [maturity]}

    _builder(market)._basis_zero(
        swap, implied, {Currency.USD: []}, free_dates,
        lambda curve, _: maturity if curve == Currency.USD else None,
    )
    assert implied[Currency.USD][maturity] == pytest.approx(z_sofr - spread, rel=1e-9)


def test_short_end_anchor_supports_price_based_overnight_index():
    # A price-based overnight index (ICP levels) must still anchor the short end, via
    # OvernightInterestPriceHistory.spot_overnight_rate — not fall back to the flat guess.
    from fintoolsom.curve_builder.builder import _CurveBuilder
    from fintoolsom.market import OvernightInterestPriceIndex
    from fintoolsom.market.index_history import OvernightInterestPriceHistory
    from fintoolsom.rates import InterpolationMethod, ProjectionInterpolationMethod

    t = date(2024, 1, 4)
    icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
    history = OvernightInterestPriceHistory(
        index=icp,
        values={date(2024, 1, 2): 100.00, date(2024, 1, 3): 100.01, date(2024, 1, 4): 100.02},
    )
    market = Market(t=t, indexes_history={"ICP": history}, riskless_index=icp)
    builder = _CurveBuilder(
        [], market, InterpolationMethod.LogLinear, ProjectionInterpolationMethod.PiecewiseConstant
    )

    anchor, guess = builder._short_end_anchor(icp)
    assert anchor is not None  # not the flat-guess fallback
    anchor_date, df = anchor
    assert anchor_date == icp.get_maturity(t)  # next business day
    assert 0.0 < df < 1.0
    assert guess > 0.0
