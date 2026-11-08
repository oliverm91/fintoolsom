import pytest
from fintoolsom.curve_builder.builder import _curves_needed, _get_maturity, _Proj
from fintoolsom.dates import TF_Actual360
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.market import Currency
from fintoolsom.market.conventions import FixedLegSpec, FloatingLegSpec, PaymentFrequency
from fintoolsom.market.quotes import (
    CrossCurrencyFixedFloatQuote,
    QuotedSide,
)
from tests.factories.swap_factory import (
    RISKLESS,
    SOFR,
    SPREAD,
    VAL_DATE,
    Ccy,
    _rate,
    adj_convention,
    build_market,
)



CLP, USD = Currency.CLP, Currency.USD


ADJ_CLP = adj_convention(Ccy.CLP)


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


def test_non_deliverable_quote_pins_usd_funding_up_to_its_last_fx_fixing():
    nd = _fixed_float_quote(is_deliverable=False, payment_currency=CLP, fx_fixing_lag=2)
    dl = _fixed_float_quote()
    # SOFR is not a collateral here: projection-only, pinned as its own forward curve.
    assert _curves_needed(nd, RISKLESS) == _curves_needed(dl, RISKLESS) == frozenset({RISKLESS, USD, _Proj(SOFR)})

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
