"""Exact valuation of legs of a STARTED swap (valuation date inside the 2nd coupon).

Expected values are computed here from first principles, using the market only as
input data (curves, fixings, ICP levels), never the calculator's own helpers:

* the coupon already paid contributes nothing;
* the running floating coupon = realised fixings [start, t] compounded with the
  projection [t, end] (term rate: the whole coupon was fixed before t);
* the running fixed coupon pays its full-period flow, not the part accrued to t.

Legs carry an explicit spread, so the expected flows include its interest.
"""

from dataclasses import replace
from datetime import date
from typing import Any
import pytest
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.legs import SwapLeg, TermRateLeg
from fintoolsom.market import Currency
from tests.factories.swap_factory import (
    CAMARA,
    FIXED_RATES,
    FLOAT_INDEXES,
    ICP,
    NOTIONALS,
    SPREAD,
    TAB_3M,
    VAL_DATE,
    Ccy,
    LegCase,
    LegKind,
    SwapStatus,
    adj_convention,
    build_leg,
    build_market,
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
    assert _pv(leg, market) == pytest.approx(
        _pv(_sub_leg(leg, RUNNING), market), rel=1e-14
    )


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
    assert _pv(_sub_leg(leg, RUNNING, RUNNING + 1), market) == pytest.approx(
        expected, rel=1e-12
    )


# ── Regressions: unpaid coupons of started legs ─────────────────────────────

ADJ_CLP = adj_convention(Ccy.CLP)


def _clp_leg(builder, index, start=date(2024, 7, 15), years=2, **kw):
    return builder(
        notional=1_000.0,
        start_date=start,
        term=Term(years, TermUnit.Y, ADJ_CLP),
        frequency="QUARTERLY",
        adj_convention=ADJ_CLP,
        index=index,
        **kw,
    )


def test_fixed_term_coupon_keeps_its_notional_repayment():
    market = build_market()
    # 1Y leg: the last coupon (2024-04-12 → 2024-07-12) fixed before VAL_DATE, unpaid.
    leg = _clp_leg(
        builders.term_rate_leg, TAB_3M, start=date(2023, 7, 12), years=1, spread=SPREAD
    )
    last = leg.coupons[-1]
    assert (
        leg.fixing_dates[-1] < VAL_DATE < last.payment_date
        and last.amortization == 1_000.0
    )

    _, flows = Calculator._leg_flows(leg, market)
    rate = market.get_index(TAB_3M.name).get_rate(leg.fixing_dates[-1]).rate
    days = (last.end_date - last.start_date).days
    expected = 1_000.0 * (rate + SPREAD.rate) * days / 360 + 1_000.0
    assert flows[-1] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("index", [CAMARA, ICP])
def test_payment_lag_values_both_running_overnight_coupons(index):
    market = build_market()
    # Coupon 2024-04-05 → 2024-07-05 pays 2024-07-09 (lag 2): ended, still unpaid at
    # VAL_DATE, while the next one (2024-07-05 → 2024-10-07) is already accruing.
    leg = _clp_leg(
        builders.overnight_leg,
        index,
        start=date(2023, 7, 5),
        spread=SPREAD,
        payment_lag=2,
    )
    ended, accruing = leg.coupons[3], leg.coupons[4]
    assert ended.end_date < VAL_DATE < ended.payment_date
    assert accruing.start_date < VAL_DATE < accruing.end_date

    mask, flows = Calculator._leg_flows(leg, market)
    assert mask[3] and mask[4] and not mask[2]
    history, curve = market.get_index(index.name), market.get_index_curve(index)
    spread = SPREAD.get_accrued_interest

    expected_ended = history.get_accrued_interest(
        1_000.0, ended.start_date, ended.end_date
    ) + spread(1_000.0, ended.start_date, ended.end_date)
    realised = history.get_accrued_interest(1_000.0, accruing.start_date, VAL_DATE)
    expected_accruing = (
        realised
        + (1_000.0 + realised) * curve.get_accrual(VAL_DATE, accruing.end_date)
        + spread(1_000.0, accruing.start_date, accruing.end_date)
    )
    assert flows[0] == pytest.approx(expected_ended, rel=1e-12)
    assert flows[1] == pytest.approx(expected_accruing, rel=1e-12)
