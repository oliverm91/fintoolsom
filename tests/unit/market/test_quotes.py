import pytest
from fintoolsom.dates import TF_Actual360
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.swaps.legs import TermRateLeg, XCCYFloatingLeg
from fintoolsom.market import Currency
from fintoolsom.market.conventions import (
    FixedLegSpec,
    FloatingLegSpec,
    PaymentFrequency,
)
from fintoolsom.market.quotes import (
    CrossCurrencyFixedFloatQuote,
    CrossCurrencyFloatFloatQuote,
    QuotedSide,
)
from tests.factories.swap_factory import (
    ICP,
    SOFR,
    SPREAD,
    TERM_SOFR_3M,
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


def test_xccy_quotes_are_deliverable_by_default():
    swap = (
        _fixed_float_quote().get_instrument()
    )  # used to raise NameError (spec.spread)
    assert swap.is_deliverable and swap.settlement_currency is None
    assert isinstance(swap.pay_leg, XCCYFloatingLeg)
    assert swap.pay_leg.spread is SPREAD


def test_non_deliverable_xccy_quote_settles_in_payment_currency():
    quote = _fixed_float_quote(
        is_deliverable=False, payment_currency=CLP, fx_fixing_lag=2
    )
    swap = quote.get_instrument()
    assert not swap.is_deliverable and swap.settlement_currency == CLP
    calendar = ADJ_CLP.calendar
    for leg in (swap.receive_leg, swap.pay_leg):
        assert leg.fx_fixing_dates == [
            calendar.add_business_days(p, -2) for p in leg.payment_dates
        ]


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
    assert swap.receive_leg.residuals[0] == pytest.approx(
        swap.pay_leg.residuals[0] * 950.0
    )


@pytest.mark.parametrize(
    "kw",
    [dict(payment_currency=CLP), dict(is_deliverable=False)],
    ids=["deliv+ccy", "nondeliv-no-ccy"],
)
def test_inconsistent_xccy_settlement_is_rejected(kw):
    with pytest.raises(ValueError, match="payment_currency"):
        _fixed_float_quote(**kw)


def test_xccy_quote_on_a_term_index_builds_a_term_leg():
    swap = _fixed_float_quote(index=TERM_SOFR_3M).get_instrument()
    assert isinstance(swap.pay_leg, TermRateLeg)


# ── Single-currency swap quotes ─────────────────────────────────────────────
from datetime import date  # noqa: E402

from fintoolsom.derivatives.forwards.forwards import NDF, Forward  # noqa: E402
from fintoolsom.derivatives.swaps.legs import FixedLeg, OvernightLeg  # noqa: E402
from fintoolsom.market import CurrencyPair  # noqa: E402
from fintoolsom.market.currencies import FX_Rate  # noqa: E402
from fintoolsom.market.quotes import (  # noqa: E402
    ForwardPointsQuote,
    ForwardPriceQuote,
    IRBasisQuote,
    IRSQuote,
)
from tests.factories.swap_factory import CAMARA, TAB_3M  # noqa: E402

TWO_Y = Term(2, TermUnit.Y, ADJ_CLP)
USDCLP = CurrencyPair(USD, CLP)


def _irs(side: QuotedSide, **kw) -> IRSQuote:
    return IRSQuote(
        quoted_side=side,
        term=TWO_Y,
        quote_date=VAL_DATE,
        adj_convention=ADJ_CLP,
        fixed_leg=_fixed_spec(CLP),
        floating_leg=_float_spec(ICP),
        **kw,
    )


@pytest.mark.parametrize("side", [QuotedSide.RECEIVE, QuotedSide.PAY])
def test_irs_quote_puts_the_fixed_leg_on_the_quoted_side(side):
    swap = _irs(side).get_instrument()
    fixed, floating = (
        (swap.receive_leg, swap.pay_leg)
        if side == QuotedSide.RECEIVE
        else (swap.pay_leg, swap.receive_leg)
    )
    assert isinstance(fixed, FixedLeg) and isinstance(floating, OvernightLeg)
    start = ADJ_CLP.calendar.add_business_days(VAL_DATE, 2)  # spot lag
    assert fixed.start_dates[0] == start and floating.start_dates[0] == start
    assert fixed.end_dates[-1] == TWO_Y.advance(start)


def test_swap_quote_honours_explicit_start_and_nearby_maturity():
    start, maturity = date(2024, 7, 15), date(2026, 7, 17)
    swap = _irs(
        QuotedSide.RECEIVE, start_date=start, maturity_date=maturity
    ).get_instrument()
    assert swap.receive_leg.start_dates[0] == start
    assert swap.receive_leg.end_dates[-1] == maturity


def test_swap_quote_rejects_a_maturity_far_from_its_term():
    with pytest.raises(ValueError, match="deviates"):
        _irs(QuotedSide.RECEIVE, maturity_date=date(2026, 9, 1)).get_instrument()


def test_irs_quote_needs_legs_in_one_currency():
    with pytest.raises(ValueError):
        IRSQuote(
            quoted_side=QuotedSide.RECEIVE,
            term=TWO_Y,
            quote_date=VAL_DATE,
            fixed_leg=_fixed_spec(USD),
            floating_leg=_float_spec(ICP),
        )


def test_ir_basis_quote_builds_two_floating_legs():
    quote = IRBasisQuote(
        quoted_side=QuotedSide.RECEIVE,
        term=TWO_Y,
        quote_date=VAL_DATE,
        adj_convention=ADJ_CLP,
        receive_leg=_float_spec(TAB_3M),
        pay_leg=_float_spec(CAMARA),
    )
    swap = quote.get_instrument()
    assert isinstance(swap.receive_leg, TermRateLeg) and isinstance(
        swap.pay_leg, OvernightLeg
    )
    assert swap.receive_leg.index is TAB_3M and swap.pay_leg.index is CAMARA


def test_ir_basis_quote_needs_legs_in_one_currency():
    with pytest.raises(ValueError):
        IRBasisQuote(
            quoted_side=QuotedSide.RECEIVE,
            term=TWO_Y,
            quote_date=VAL_DATE,
            receive_leg=_float_spec(ICP),
            pay_leg=_float_spec(SOFR),
        )


def test_cross_currency_float_float_quote_needs_two_currencies():
    with pytest.raises(ValueError, match="different currencies"):
        CrossCurrencyFloatFloatQuote(
            quoted_side=QuotedSide.RECEIVE,
            term=TWO_Y,
            quote_date=VAL_DATE,
            receive_leg=_float_spec(SOFR),
            pay_leg=_float_spec(SOFR),
        )


def test_cross_currency_fixed_float_quote_needs_two_currencies():
    with pytest.raises(ValueError, match="different currencies"):
        CrossCurrencyFixedFloatQuote(
            quoted_side=QuotedSide.RECEIVE,
            term=TWO_Y,
            quote_date=VAL_DATE,
            fixed_leg=_fixed_spec(USD),
            floating_leg=_float_spec(SOFR),
        )


# ── FX forward quotes ───────────────────────────────────────────────────────


def test_forward_price_quote_from_a_term_fixes_after_spot_and_pays_a_day_later():
    term = Term(1, TermUnit.M, ADJ_CLP)
    quote = ForwardPriceQuote(USDCLP, 950.0, True, term=term, quote_date=VAL_DATE)
    ndf = quote.get_instrument()
    spot = ADJ_CLP.calendar.add_business_days(VAL_DATE, 2)
    assert isinstance(ndf, NDF)
    assert ndf.fixing_date == term.advance(spot)
    assert ndf.payment_date == ADJ_CLP.calendar.add_business_days(ndf.fixing_date, 1)
    assert quote.to_outright(FX_Rate(USDCLP, 900.0)) == 950.0


def test_forward_price_quote_with_a_payment_date_is_deliverable():
    fwd = ForwardPriceQuote(
        USDCLP, 950.0, False, payment_date=date(2024, 10, 8)
    ).get_instrument()
    assert type(fwd) is Forward and fwd.strike == 950.0 and not fwd.is_buy


def test_forward_quote_without_dates_cannot_build():
    with pytest.raises(ValueError, match="payment_date"):
        ForwardPriceQuote(USDCLP, 950.0, True).get_instrument()


def test_forward_points_add_to_the_spot_in_either_quotation():
    quote = ForwardPointsQuote(
        currency_pair=USDCLP,
        value=250.0,
        is_buy=True,
        points_divisor=100,
        payment_date=date(2024, 10, 8),
    )
    assert quote.to_outright(FX_Rate(USDCLP, 900.0)) == pytest.approx(902.5)
    assert quote.to_outright(
        FX_Rate(CurrencyPair(CLP, USD), 1 / 900.0)
    ) == pytest.approx(902.5)
    with pytest.raises(ValueError, match="incompatible"):
        quote.to_outright(FX_Rate(CurrencyPair(Currency.EUR, CLP), 1000.0))
    assert quote.get_instrument().strike == 0.0  # structure only, no market


def test_forward_points_divisor_must_be_positive():
    with pytest.raises(ValueError):
        ForwardPointsQuote(
            currency_pair=USDCLP, value=1.0, is_buy=True, points_divisor=0
        )


def test_leg_spec_payment_lag_defaults_to_zero_and_is_keyword_only():
    assert _fixed_spec(CLP).payment_lag == 0
    assert _float_spec(SOFR).payment_lag == 0


def test_swap_quote_passes_each_leg_spec_payment_lag_to_its_leg():
    fixed = _fixed_spec(CLP)
    fixed.payment_lag = 1
    floating = _float_spec(SOFR)
    floating.payment_lag = 2
    swap = CrossCurrencyFixedFloatQuote(
        quoted_side=QuotedSide.RECEIVE,
        term=Term(2, TermUnit.Y, ADJ_CLP),
        quote_date=VAL_DATE,
        adj_convention=ADJ_CLP,
        fixed_leg=fixed,
        floating_leg=floating,
    ).get_instrument()
    calendar = ADJ_CLP.calendar
    for leg, lag in ((swap.receive_leg, 1), (swap.pay_leg, 2)):
        assert leg.payment_dates == [
            calendar.add_business_days(d, lag) for d in leg.end_dates
        ]
