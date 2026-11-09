from dataclasses import replace
from datetime import date
from typing import Any
import pytest
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair
from tests.factories.swap_factory import (
    FIXED_RATES,
    FX_TO_CLP,
    NOTIONALS,
    PAST_FX_FACTOR,
    PAYMENT_FREQUENCY,
    SOFR,
    SPREAD,
    START_DATES,
    SWAP_TERM_YEARS,
    VAL_DATE,
    Ccy,
    SwapStatus,
    _rate,
    adj_convention,
    build_market,
)


USD, CLP = Currency.USD, Currency.CLP


SPOT = FX_TO_CLP[Ccy.USD]


USD_CLP = CurrencyPair(USD, CLP)


@pytest.fixture(scope="module")
def market():
    return build_market()


def _common(ccy: Ccy, status: SwapStatus) -> dict:
    return dict(
        notional=NOTIONALS[ccy],
        start_date=START_DATES[status],
        term=Term(SWAP_TERM_YEARS, TermUnit.Y, adj_convention(ccy)),
        frequency=PAYMENT_FREQUENCY,
        adj_convention=adj_convention(ccy),
    )


def _xccy_usd(
    status: SwapStatus = SwapStatus.NOT_STARTED, fx_fixing_lag: int = 0
) -> Any:
    return builders.xccy_floating_leg(
        index=SOFR,
        spread=SPREAD,
        fx_fixing_lag=fx_fixing_lag,
        **_common(Ccy.USD, status),
    )


def _fixed(
    ccy: Ccy, status: SwapStatus = SwapStatus.NOT_STARTED, fx_fixing_lag: int = 0
) -> Any:
    return builders.fixed_leg(
        rate=_rate(FIXED_RATES[ccy]),
        currency=USD if ccy is Ccy.USD else CLP,
        fx_fixing_lag=fx_fixing_lag,
        **_common(ccy, status),
    )


def _df(market, currency: Currency, d: date) -> float:
    """Uncollateralised DF: the riskless (ICP) curve for CLP, the currency curve otherwise."""
    return market.get_discount_curve(currency).get_df(d)


def _forward_fx(market, d: date) -> float:
    return SPOT * _df(market, USD, d) / _df(market, CLP, d)


def _xccy_flows(market, leg) -> list[float]:
    """Unstarted SOFR coupons: projected interest + spread + final notional."""
    curve = market.get_index_curve(SOFR)
    return [
        c.residual * (curve.get_df(c.start_date) / curve.get_df(c.end_date) - 1)
        + c.residual * SPREAD.rate * (c.end_date - c.start_date).days / 360
        + c.amortization
        for c in leg.coupons
    ]


def test_deliverable_xccy_leg_discounts_its_own_flows_in_its_own_currency(market):
    leg = _xccy_usd()
    expected = sum(
        f * _df(market, USD, c.payment_date)
        for f, c in zip(_xccy_flows(market, leg), leg.coupons)
    )
    assert Calculator._leg_pv(leg, None, market) == pytest.approx(expected, rel=1e-12)


def test_deliverable_swap_converts_each_leg_pv_at_spot(market):
    usd, clp = _xccy_usd(), _fixed(Ccy.CLP)
    swap = Swap(receive_leg=usd, pay_leg=clp, is_deliverable=True)
    expected = SPOT * Calculator._leg_pv(usd, None, market) - Calculator._leg_pv(
        clp, None, market
    )
    assert swap.valuate(market, CLP) == pytest.approx(expected, rel=1e-12)
    assert swap.valuate(market, USD) == pytest.approx(expected / SPOT, rel=1e-12)


def test_non_deliverable_projects_fx_to_each_fixing_date(market):
    usd, clp = _xccy_usd(fx_fixing_lag=2), _fixed(Ccy.CLP)
    expected_usd_leg = sum(
        f * _forward_fx(market, fix) * _df(market, CLP, c.payment_date)
        for f, fix, c in zip(_xccy_flows(market, usd), usd.fx_fixing_dates, usd.coupons)
    )
    swap = Swap(receive_leg=usd, pay_leg=clp, payment_currency=CLP)
    expected = expected_usd_leg - Calculator._leg_pv(clp, None, market)
    assert swap.valuate(market, CLP) == pytest.approx(expected, rel=1e-12)


def test_non_deliverable_differs_from_deliverable(market):
    usd, clp = _xccy_usd(fx_fixing_lag=2), _fixed(Ccy.CLP)
    nd = Swap(receive_leg=usd, pay_leg=clp, payment_currency=CLP).valuate(market, CLP)
    dl = Swap(receive_leg=usd, pay_leg=clp, is_deliverable=True).valuate(market, CLP)
    assert abs(nd - dl) > 1.0  # fixing != payment date, and USD flows discount on CLP


def test_flow_with_fx_fixed_but_not_paid_uses_the_realised_fixing(market):
    # Started leg, running coupon paid 2024-07-10; 3 USD business days earlier is
    # 2024-07-05 (July 4th is a holiday): fixed before VAL_DATE, paid after it.
    leg = _fixed(Ccy.USD, SwapStatus.STARTED, fx_fixing_lag=3)
    running = next(
        i
        for i, c in enumerate(leg.coupons)
        if c.fx_fixing_date < VAL_DATE < c.payment_date
    )
    c = leg.coupons[running]
    realised = market.get_fx_rate(c.fx_fixing_date, USD_CLP).value
    assert realised == pytest.approx(SPOT * PAST_FX_FACTOR)

    only_running = replace(leg, coupons=[c])
    pv = Calculator._leg_pv_non_deliverable(only_running, None, market, CLP)
    assert pv == pytest.approx(
        c.flow * realised * _df(market, CLP, c.payment_date), rel=1e-12
    )


def test_fixing_on_valuation_date_uses_todays_fixing(market):
    leg = _fixed(
        Ccy.USD, SwapStatus.STARTED, fx_fixing_lag=2
    )  # fixes 2024-07-08 == VAL_DATE
    c = next(c for c in leg.coupons if c.fx_fixing_date == VAL_DATE)
    pv = Calculator._leg_pv_non_deliverable(
        replace(leg, coupons=[c]), None, market, CLP
    )
    assert pv == pytest.approx(
        c.flow * SPOT * _df(market, CLP, c.payment_date), rel=1e-12
    )


def test_paid_flow_is_dropped_even_though_its_fx_was_fixed(market):
    leg = _fixed(Ccy.USD, SwapStatus.STARTED, fx_fixing_lag=3)
    paid = [c for c in leg.coupons if c.payment_date <= VAL_DATE]
    assert paid
    assert (
        Calculator._leg_pv_non_deliverable(
            replace(leg, coupons=paid), None, market, CLP
        )
        == 0.0
    )


def test_started_non_deliverable_xccy_leg_mixes_realised_and_forward_fixings(market):
    leg = _xccy_usd(SwapStatus.STARTED, fx_fixing_lag=3)
    mask, flows = Calculator._leg_flows(
        leg, market
    )  # leg-currency flows (tested elsewhere)
    unpaid = [c for c, m in zip(leg.coupons, mask) if m]
    assert unpaid[0].fx_fixing_date < VAL_DATE < unpaid[0].payment_date
    fx = [
        market.get_fx_rate(c.fx_fixing_date, USD_CLP).value
        if c.fx_fixing_date <= VAL_DATE
        else _forward_fx(market, c.fx_fixing_date)
        for c in unpaid
    ]
    expected = sum(
        f * x * _df(market, CLP, c.payment_date) for f, x, c in zip(flows, fx, unpaid)
    )
    assert Calculator._leg_pv_non_deliverable(leg, None, market, CLP) == pytest.approx(
        expected, rel=1e-12
    )


def test_non_deliverable_leg_in_the_settlement_currency_is_plain(market):
    clp = _fixed(Ccy.CLP)
    assert Calculator._leg_pv_non_deliverable(
        clp, None, market, CLP
    ) == Calculator._leg_pv(clp, None, market)
