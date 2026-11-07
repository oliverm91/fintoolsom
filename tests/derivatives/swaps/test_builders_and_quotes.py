"""Leg builders, quote instrument builders and curve-builder routing.

Covers the builder defaults (zero spread, term fixing lag), the floating-leg kind
chosen per index, non-deliverable cross-currency quotes and their curve routing, a
fixed term coupon keeping its notional repayment, and Market.get_rate fallback.
"""
from datetime import date

import pytest

from fintoolsom.curve_builder.builder import _curves_needed, _get_maturity
from fintoolsom.dates import TF_Actual360
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.legs import OvernightLeg, TermRateLeg, XCCYFloatingLeg
from fintoolsom.market import Currency, Market
from fintoolsom.market.conventions import FixedLegSpec, FloatingLegSpec, PaymentFrequency
from fintoolsom.market.quotes import (
    CrossCurrencyFixedFloatQuote,
    CrossCurrencyFloatFloatQuote,
    QuotedSide,
)

from .swap_factory import (
    CAMARA,
    ICP,
    RISKLESS,
    SOFR,
    SPREAD,
    TAB_3M,
    TERM_SOFR_3M,
    VAL_DATE,
    Ccy,
    _rate,
    adj_convention,
    build_market,
)

CLP, USD = Currency.CLP, Currency.USD
ADJ_CLP = adj_convention(Ccy.CLP)


def _leg(builder, index, start=date(2024, 7, 15), years=2, **kw):
    return builder(
        notional=1_000.0,
        start_date=start,
        term=Term(years, TermUnit.Y, ADJ_CLP),
        frequency="QUARTERLY",
        adj_convention=ADJ_CLP,
        index=index,
        **kw,
    )


# ── Leg builders ────────────────────────────────────────────────────────────

def test_spread_defaults_to_a_fresh_zero_rate_per_leg():
    a, b = _leg(builders.overnight_leg, CAMARA), _leg(builders.term_rate_leg, TAB_3M)
    assert a.spread.rate_value == 0.0 and b.spread.rate_value == 0.0
    assert a.spread is not b.spread  # bumping one leg's spread must not move another's
    assert list(a.spreads_values) == [0.0] * a.n_coupons


def test_term_leg_fixes_spot_lag_business_days_before_each_start():
    leg = _leg(builders.term_rate_leg, TAB_3M)
    calendar = ADJ_CLP.calendar
    assert TAB_3M.spot_lag == 2
    assert leg.fixing_dates == [calendar.add_business_days(s, -2) for s in leg.start_dates]
    assert _leg(builders.term_rate_leg, TAB_3M, fixing_lag=0).fixing_dates == leg.start_dates


@pytest.mark.parametrize(
    "index, cross_currency, expected",
    [
        (CAMARA, False, OvernightLeg),
        (ICP, False, OvernightLeg),
        (CAMARA, True, XCCYFloatingLeg),
        (TAB_3M, False, TermRateLeg),
        (TAB_3M, True, TermRateLeg),  # a term index never compounds daily
    ],
)
def test_floating_leg_builds_the_kind_the_index_calls_for(index, cross_currency, expected):
    leg = _leg(builders.floating_leg, index, cross_currency=cross_currency, spread=SPREAD)
    assert type(leg) is expected


# ── Calculator fixes ────────────────────────────────────────────────────────

def test_fixed_term_coupon_keeps_its_notional_repayment():
    market = build_market()
    # 1Y leg: the last coupon (2024-04-12 → 2024-07-12) fixed before VAL_DATE, unpaid.
    leg = _leg(builders.term_rate_leg, TAB_3M, start=date(2023, 7, 12), years=1, spread=SPREAD)
    last = leg.coupons[-1]
    assert leg.fixing_dates[-1] < VAL_DATE < last.payment_date and last.amortization == 1_000.0

    _, flows = Calculator._leg_flows(leg, market)
    rate = market.get_index(TAB_3M.name).get_rate(leg.fixing_dates[-1]).rate_value
    days = (last.end_date - last.start_date).days
    expected = 1_000.0 * (rate + SPREAD.rate_value) * days / 360 + 1_000.0
    assert flows[-1] == pytest.approx(expected, rel=1e-12)


def test_get_rate_falls_back_to_the_latest_past_fixing():
    d1, d2, d3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    market = Market(t=d3, interest_rates={"x": {d1: _rate(0.01), d3: _rate(0.03)}})
    assert market.get_rate(d2, "X", use_closest_past_rate=True).rate_value == 0.01
    with pytest.raises(ValueError):
        market.get_rate(d2, "X")
    with pytest.raises(ValueError):
        market.get_rate(date(2023, 1, 1), "X", use_closest_past_rate=True)


# ── Cross-currency quotes ───────────────────────────────────────────────────

def _fixed_spec(ccy: Currency) -> FixedLegSpec:
    return FixedLegSpec(
        currency=ccy,
        payment_frequency=PaymentFrequency.QUARTERLY,
        adj_convention=ADJ_CLP,
        time_fraction=TF_Actual360,
        rate=_rate(0.055),
    )


def _float_spec(index) -> FloatingLegSpec:
    return FloatingLegSpec(
        currency=index.currency,
        payment_frequency=PaymentFrequency.QUARTERLY,
        adj_convention=ADJ_CLP,
        time_fraction=TF_Actual360,
        index=index,
        spread=SPREAD,
    )


def _fixed_float_quote(index: object = SOFR, **kw) -> CrossCurrencyFixedFloatQuote:
    return CrossCurrencyFixedFloatQuote(
        quoted_side=QuotedSide.RECEIVE,
        term=Term(2, TermUnit.Y, ADJ_CLP),
        quote_date=VAL_DATE,
        adj_convention=ADJ_CLP,
        fixed_leg=_fixed_spec(CLP),
        floating_leg=_float_spec(index),
        **kw,
    )


def test_xccy_quotes_are_deliverable_by_default():
    swap = _fixed_float_quote().get_instrument()  # used to raise NameError (spec.spread)
    assert swap.is_deliverable and swap.settlement_currency is None
    assert isinstance(swap.pay_leg, XCCYFloatingLeg)
    assert swap.pay_leg.spread is SPREAD


def test_non_deliverable_xccy_quote_settles_in_payment_currency():
    quote = _fixed_float_quote(is_deliverable=False, payment_currency=CLP, fx_fixing_lag=2)
    swap = quote.get_instrument()
    assert not swap.is_deliverable and swap.settlement_currency == CLP
    calendar = ADJ_CLP.calendar
    for leg in (swap.receive_leg, swap.pay_leg):
        assert leg.fx_fixing_dates == [calendar.add_business_days(p, -2) for p in leg.payment_dates]


def test_float_float_quote_supports_non_deliverable_settlement():
    quote = CrossCurrencyFloatFloatQuote(
        quoted_side=QuotedSide.RECEIVE,
        term=Term(2, TermUnit.Y, ADJ_CLP),
        quote_date=VAL_DATE,
        adj_convention=ADJ_CLP,
        receive_leg=_float_spec(ICP),
        pay_leg=_float_spec(SOFR),
        is_deliverable=False,
        payment_currency=CLP,
    )
    swap = quote.get_instrument(build_market())
    assert swap.settlement_currency == CLP
    assert swap.receive_leg.residuals[0] == pytest.approx(swap.pay_leg.residuals[0] * 950.0)


@pytest.mark.parametrize(
    "kw", [dict(payment_currency=CLP), dict(is_deliverable=False)], ids=["deliv+ccy", "nondeliv-no-ccy"]
)
def test_inconsistent_xccy_settlement_is_rejected(kw):
    with pytest.raises(ValueError, match="payment_currency"):
        _fixed_float_quote(**kw)


def test_xccy_quote_on_a_term_index_builds_a_term_leg():
    swap = _fixed_float_quote(index=TERM_SOFR_3M).get_instrument()
    assert isinstance(swap.pay_leg, TermRateLeg)


# ── Curve-builder routing ───────────────────────────────────────────────────

def test_non_deliverable_quote_pins_usd_funding_up_to_its_last_fx_fixing():
    nd = _fixed_float_quote(is_deliverable=False, payment_currency=CLP, fx_fixing_lag=2)
    dl = _fixed_float_quote()
    assert _curves_needed(nd, RISKLESS) == _curves_needed(dl, RISKLESS) == frozenset({RISKLESS, USD})

    nd_swap, dl_swap = nd.get_instrument(), dl.get_instrument()
    usd_leg = nd_swap.pay_leg
    assert _get_maturity(USD, nd_swap, RISKLESS) == max(usd_leg.fx_fixing_dates)
    assert _get_maturity(USD, dl_swap, RISKLESS) == max(dl_swap.pay_leg.payment_dates)
    assert _get_maturity(RISKLESS, nd_swap, RISKLESS) == max(usd_leg.payment_dates)


def test_non_deliverable_quote_values_through_the_settlement_path():
    market = build_market()
    swap = _fixed_float_quote(is_deliverable=False, payment_currency=CLP, fx_fixing_lag=2).get_instrument()
    usd = swap.pay_leg
    expected = Calculator._leg_pv(swap.receive_leg, None, market) - Calculator._leg_pv_non_deliverable(
        usd, None, market, CLP
    )
    assert swap.valuate(market, CLP) == pytest.approx(expected, rel=1e-12)
