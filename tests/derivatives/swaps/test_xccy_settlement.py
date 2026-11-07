"""XCCYFloatingLeg and deliverable vs non-deliverable cross-currency settlement.

Expected values are built from the market inputs (curves, FX history) only:

* deliverable: each leg is discounted in its own currency, PV converted at spot;
* non-deliverable: each flow is converted at its FX fixing (realised when fixed on
  or before the valuation date, forward ``spot * df_base / df_quote`` otherwise) and
  discounted in the settlement currency;
* a flow whose FX is fixed but not yet paid uses the realised fixing;
* only the final notional exchange exists (no initial exchange).
"""
from dataclasses import replace
from datetime import date
from typing import Any

import pytest

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.coupons import FixedCoupon, XCCYCoupon
from fintoolsom.derivatives.swaps.legs import OvernightLeg, XCCYFloatingLeg
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair, FX_Rate

from .swap_factory import (
    FIXED_RATES,
    FX_TO_CLP,
    NOTIONALS,
    PAST_FX_FACTOR,
    PAYMENT_FREQUENCY,
    RISKLESS,
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


def _xccy_usd(status: SwapStatus = SwapStatus.NOT_STARTED, fx_fixing_lag: int = 0) -> Any:
    return builders.xccy_floating_leg(
        index=SOFR, spread=SPREAD, fx_fixing_lag=fx_fixing_lag, **_common(Ccy.USD, status)
    )


def _fixed(ccy: Ccy, status: SwapStatus = SwapStatus.NOT_STARTED, fx_fixing_lag: int = 0) -> Any:
    return builders.fixed_leg(
        rate=_rate(FIXED_RATES[ccy]),
        currency=USD if ccy is Ccy.USD else CLP,
        fx_fixing_lag=fx_fixing_lag,
        **_common(ccy, status),
    )


def _df(market, currency: Currency, d: date) -> float:
    """Uncollateralised DF: the riskless (ICP) curve for CLP, the currency curve otherwise."""
    return market.curves[RISKLESS if currency == CLP else currency].get_df(d)


def _forward_fx(market, d: date) -> float:
    return SPOT * _df(market, USD, d) / _df(market, CLP, d)


def _xccy_flows(market, leg) -> list[float]:
    """Unstarted SOFR coupons: projected interest + spread + final notional."""
    curve = market.curves[SOFR]
    return [
        c.residual * (curve.get_df(c.start_date) / curve.get_df(c.end_date) - 1)
        + c.residual * SPREAD.rate_value * (c.end_date - c.start_date).days / 360
        + c.amortization
        for c in leg.coupons
    ]


# ── XCCYFloatingLeg construction ────────────────────────────────────────────

def test_xccy_leg_is_a_daily_compounded_leg_of_xccy_coupons():
    leg = _xccy_usd()
    assert isinstance(leg, XCCYFloatingLeg) and isinstance(leg, OvernightLeg)
    assert all(isinstance(c, XCCYCoupon) for c in leg.coupons)
    assert leg.currency == USD
    assert leg.n_coupons == SWAP_TERM_YEARS * 4


def test_only_the_final_notional_is_exchanged():
    leg = _xccy_usd()
    assert list(leg.amortizations[:-1]) == [0.0] * (leg.n_coupons - 1)
    assert leg.amortizations[-1] == NOTIONALS[Ccy.USD]
    assert min(leg.payment_dates) > leg.start_dates[0]  # no flow at the start date


def test_fx_fixing_dates_default_to_payment_dates():
    for leg in (_xccy_usd(), _fixed(Ccy.USD)):
        assert leg.fx_fixing_dates == leg.payment_dates
        assert [c.fx_fixing_date for c in leg.coupons] == leg.payment_dates


def test_fx_fixing_lag_counts_business_days_back_from_payment():
    calendar = adj_convention(Ccy.USD).calendar
    leg = _xccy_usd(fx_fixing_lag=2)
    assert leg.fx_fixing_dates == [calendar.add_business_days(p, -2) for p in leg.payment_dates]
    assert all(f < p for f, p in zip(leg.fx_fixing_dates, leg.payment_dates))


def test_invalid_fx_fixing_dates_raise():
    with pytest.raises(ValueError):
        _xccy_usd(fx_fixing_lag=-1)
    with pytest.raises(ValueError, match="fx_fixing_date"):
        d = date(2025, 1, 10)
        XCCYCoupon(1.0, 0.0, date(2024, 10, 10), d, d, fx_fixing_date=date(2025, 1, 13), spread=SPREAD)


# ── Swap settlement definition ──────────────────────────────────────────────

def test_settlement_currency_validation():
    usd, clp = _xccy_usd(), _fixed(Ccy.CLP)
    with pytest.raises(ValueError, match="payment_currency"):
        Swap(receive_leg=usd, pay_leg=clp)  # non-deliverable XCCY needs one
    with pytest.raises(ValueError, match="payment_currency"):
        Swap(receive_leg=usd, pay_leg=clp, is_deliverable=True, payment_currency=CLP)
    assert Swap(receive_leg=usd, pay_leg=clp, is_deliverable=True).settlement_currency is None
    assert Swap(receive_leg=usd, pay_leg=clp, payment_currency=CLP).settlement_currency == CLP
    Swap(receive_leg=clp, pay_leg=_fixed(Ccy.CLP))  # single currency: nothing to settle


def test_market_fx_lookup_inverts_and_is_identity_on_same_currency(market):
    assert market.get_fx_rate(VAL_DATE, CurrencyPair(CLP, USD)).value == pytest.approx(1 / SPOT)
    assert market.get_fx_rate(VAL_DATE, CurrencyPair(CLP, CLP)).value == 1.0
    market.add_fx_rate(date(2000, 1, 3), FX_Rate(CurrencyPair(Currency.EUR, USD), 1.1))
    assert market.get_fx_rate(date(2000, 1, 3), CurrencyPair(USD, Currency.EUR)).value == pytest.approx(1 / 1.1)


# ── Deliverable ─────────────────────────────────────────────────────────────

def test_deliverable_xccy_leg_discounts_its_own_flows_in_its_own_currency(market):
    leg = _xccy_usd()
    expected = sum(f * _df(market, USD, c.payment_date) for f, c in zip(_xccy_flows(market, leg), leg.coupons))
    assert Calculator._leg_pv(leg, None, market) == pytest.approx(expected, rel=1e-12)


def test_deliverable_swap_converts_each_leg_pv_at_spot(market):
    usd, clp = _xccy_usd(), _fixed(Ccy.CLP)
    swap = Swap(receive_leg=usd, pay_leg=clp, is_deliverable=True)
    expected = SPOT * Calculator._leg_pv(usd, None, market) - Calculator._leg_pv(clp, None, market)
    assert swap.valuate(market, CLP) == pytest.approx(expected, rel=1e-12)
    assert swap.valuate(market, USD) == pytest.approx(expected / SPOT, rel=1e-12)


# ── Non-deliverable ─────────────────────────────────────────────────────────

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
    running = next(i for i, c in enumerate(leg.coupons) if c.fx_fixing_date < VAL_DATE < c.payment_date)
    c = leg.coupons[running]
    realised = market.get_fx_rate(c.fx_fixing_date, USD_CLP).value
    assert realised == pytest.approx(SPOT * PAST_FX_FACTOR)

    only_running = replace(leg, coupons=[c])
    pv = Calculator._leg_pv_non_deliverable(only_running, None, market, CLP)
    assert pv == pytest.approx(c.flow * realised * _df(market, CLP, c.payment_date), rel=1e-12)


def test_fixing_on_valuation_date_uses_todays_fixing(market):
    leg = _fixed(Ccy.USD, SwapStatus.STARTED, fx_fixing_lag=2)  # fixes 2024-07-08 == VAL_DATE
    c = next(c for c in leg.coupons if c.fx_fixing_date == VAL_DATE)
    pv = Calculator._leg_pv_non_deliverable(replace(leg, coupons=[c]), None, market, CLP)
    assert pv == pytest.approx(c.flow * SPOT * _df(market, CLP, c.payment_date), rel=1e-12)


def test_paid_flow_is_dropped_even_though_its_fx_was_fixed(market):
    leg = _fixed(Ccy.USD, SwapStatus.STARTED, fx_fixing_lag=3)
    paid = [c for c in leg.coupons if c.payment_date <= VAL_DATE]
    assert paid
    assert Calculator._leg_pv_non_deliverable(replace(leg, coupons=paid), None, market, CLP) == 0.0


def test_started_non_deliverable_xccy_leg_mixes_realised_and_forward_fixings(market):
    leg = _xccy_usd(SwapStatus.STARTED, fx_fixing_lag=3)
    mask, flows = Calculator._leg_flows(leg, market)  # leg-currency flows (tested elsewhere)
    unpaid = [c for c, m in zip(leg.coupons, mask) if m]
    assert unpaid[0].fx_fixing_date < VAL_DATE < unpaid[0].payment_date
    fx = [
        market.get_fx_rate(c.fx_fixing_date, USD_CLP).value if c.fx_fixing_date <= VAL_DATE
        else _forward_fx(market, c.fx_fixing_date)
        for c in unpaid
    ]
    expected = sum(f * x * _df(market, CLP, c.payment_date) for f, x, c in zip(flows, fx, unpaid))
    assert Calculator._leg_pv_non_deliverable(leg, None, market, CLP) == pytest.approx(expected, rel=1e-12)


def test_non_deliverable_leg_in_the_settlement_currency_is_plain(market):
    clp = _fixed(Ccy.CLP)
    assert Calculator._leg_pv_non_deliverable(clp, None, market, CLP) == Calculator._leg_pv(clp, None, market)


def test_fixed_coupon_carries_fx_fixing_date():
    d = date(2025, 1, 10)
    c = FixedCoupon(100.0, 0.0, date(2024, 10, 10), d, d, rate=_rate(0.05), fx_fixing_date=date(2025, 1, 8))
    assert c.fx_fixing_date == date(2025, 1, 8)
