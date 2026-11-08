"""Exact valuation of legs of a STARTED swap (valuation date inside the 2nd coupon).

Expected values are computed here from first principles, using the market only as
input data (curves, fixings, ICP levels), never the calculator's own helpers:

* the coupon already paid contributes nothing;
* the running floating coupon = realised fixings [start, t] compounded with the
  projection [t, end] (term rate: the whole coupon was fixed before t);
* the running fixed coupon pays its full-period flow, not the part accrued to t.

Legs carry an explicit spread: the builders' default spread is a known bug.
"""

from dataclasses import replace
from datetime import date
from typing import Any
import pytest
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps.legs import SwapLeg, TermRateLeg
from fintoolsom.market import Currency
from tests.factories.swap_factory import (
    FIXED_RATES,
    FLOAT_INDEXES,
    ICP,
    NOTIONALS,
    SPREAD,
    VAL_DATE,
    Ccy,
    LegCase,
    LegKind,
    SwapStatus,
    build_leg,
)



RUNNING = 1  # coupon index: the 1st coupon is paid, the 2nd is accruing at VAL_DATE


FIXED_CASES = [LegCase(Ccy.CLP, LegKind.FIXED), LegCase(Ccy.USD, LegKind.FIXED)]


FLOAT_CASES = [LegCase(ccy, kind) for (ccy, kind) in FLOAT_INDEXES]


ALL_CASES = FIXED_CASES + FLOAT_CASES


def _ids(cases: list[LegCase]) -> list[str]:
    return [str(c) for c in cases]


@pytest.fixture(scope="module")
def market():
    from tests.factories.swap_factory import build_market

    return build_market()


def _started_leg(case: LegCase) -> Any:
    return build_leg(case, SwapStatus.STARTED, with_spread=True)


def _sub_leg(leg: SwapLeg, start: int, stop: int | None = None) -> SwapLeg:
    """The same leg restricted to coupons[start:stop]."""
    sl = slice(start, stop)
    if isinstance(leg, TermRateLeg):
        return replace(leg, coupons=leg.coupons[sl], fixing_dates=leg.fixing_dates[sl])
    return replace(leg, coupons=leg.coupons[sl])


def _pv(leg: SwapLeg, market) -> float:
    return Calculator._leg_pv(leg, None, market)


def _discount_df(market, currency: Currency, d: date) -> float:
    """Uncollateralised DF: the riskless (ICP) curve for CLP, the currency curve otherwise."""
    return market.get_discount_curve(currency).get_df(d)


def _spread_flow(notional: float, start: date, end: date) -> float:
    return notional * SPREAD.rate * (end - start).days / 360  # linear act/360


def _realised_wf(market, index, start: date, end: date) -> float:
    """Realised wealth factor of an overnight index over [start, end] from its fixings."""
    history = market.get_index(index.name)
    if index == ICP:  # price index: ratio of published levels
        return history.get_value(end) / history.get_value(start)
    wf, d = 1.0, start
    while d < end:  # daily compounding on the index's own business days
        nxt = index.get_maturity(d)
        r = history.get_rate(d)
        wf *= 1 + r.rate * (min(nxt, end) - d).days / 360
        d = nxt
    return wf


def test_valuation_date_sits_inside_the_running_coupon():
    for case in ALL_CASES:
        leg = _started_leg(case)
        assert leg.payment_dates[RUNNING - 1] < VAL_DATE  # previous coupon paid
        assert leg.start_dates[RUNNING] < VAL_DATE < leg.end_dates[RUNNING]


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids(ALL_CASES))
def test_paid_coupon_is_dropped(case, market):
    leg = _started_leg(case)
    assert _pv(leg, market) == pytest.approx(_pv(_sub_leg(leg, RUNNING), market), rel=1e-14)


@pytest.mark.parametrize("case", FIXED_CASES, ids=_ids(FIXED_CASES))
def test_running_fixed_coupon_is_valued_in_full(case, market):
    leg = _started_leg(case)
    c = leg.coupons[RUNNING]
    notional, rate = NOTIONALS[case.ccy], FIXED_RATES[case.ccy]
    df = _discount_df(market, leg.currency, c.payment_date)

    full_flow = notional * rate * (c.end_date - c.start_date).days / 360
    accrued_to_t = notional * rate * (VAL_DATE - c.start_date).days / 360

    pv = _pv(_sub_leg(leg, RUNNING, RUNNING + 1), market)
    assert pv == pytest.approx(full_flow * df, rel=1e-12)
    assert pv > accrued_to_t * df * 1.01  # not just the part accrued so far


@pytest.mark.parametrize("case", FLOAT_CASES, ids=_ids(FLOAT_CASES))
def test_running_floating_coupon_combines_realised_and_projected(case, market):
    leg = _started_leg(case)
    c = leg.coupons[RUNNING]
    index = FLOAT_INDEXES[(case.ccy, case.kind)]
    notional = NOTIONALS[case.ccy]
    df = _discount_df(market, leg.currency, c.payment_date)

    if case.kind is LegKind.TERM:
        # The whole coupon was fixed before t: one rate over [start, end], nothing projected.
        fixing_date = leg.fixing_dates[RUNNING]
        assert fixing_date < VAL_DATE
        r = market.get_index(index.name).get_rate(fixing_date)
        interest = notional * r.rate * (c.end_date - c.start_date).days / 360
    else:
        realised_wf = _realised_wf(market, index, c.start_date, VAL_DATE)
        curve = market.get_index_curve(index)
        projected_wf = curve.get_df(VAL_DATE) / curve.get_df(c.end_date)
        interest = notional * (realised_wf * projected_wf - 1)

    expected = (interest + _spread_flow(notional, c.start_date, c.end_date)) * df
    assert _pv(_sub_leg(leg, RUNNING, RUNNING + 1), market) == pytest.approx(expected, rel=1e-12)
