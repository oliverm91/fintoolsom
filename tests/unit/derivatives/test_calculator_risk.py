"""Leg duration and DV01 estimates (Calculator.get_leg_duration / get_leg_dv01)."""

import numpy as np
import pytest

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps.swaps import LegSide, Swap
from fintoolsom.market import Currency
from tests.factories.swap_factory import (
    FLAT_RATES,
    ICP,
    VAL_DATE,
    Ccy,
    LegCase,
    LegKind,
    SwapStatus,
    build_leg,
    build_market,
)

CLP = Currency.CLP


@pytest.fixture(scope="module")
def market():
    return build_market()


def _swap(
    receive: LegKind, pay: LegKind, status=SwapStatus.NOT_STARTED, spread=False
) -> Swap:
    return Swap(
        receive_leg=build_leg(LegCase(Ccy.CLP, receive), status, spread),
        pay_leg=build_leg(LegCase(Ccy.CLP, pay), status, spread),
    )


def _tau(d):
    return (d - VAL_DATE).days / 365


def test_fixed_leg_duration_is_the_pv_weighted_time_of_its_flows(market):
    swap = _swap(LegKind.FIXED, LegKind.PRICE)
    leg = swap.receive_leg
    taus = np.array([_tau(d) for d in leg.payment_dates])
    pvs = leg.flows * np.exp(-FLAT_RATES[ICP] * taus)  # flat continuous ICP curve
    expected = float(np.sum(taus * pvs) / np.sum(pvs))
    assert Calculator.get_leg_duration(swap, LegSide.RECEIVE, market) == pytest.approx(
        expected, rel=1e-6
    )
    assert 1.8 < expected < 2.0  # a 2Y bullet-like leg


def test_unstarted_floating_leg_duration_is_its_start(market):
    # No spread: the floating-rate-note equivalent is the notional at the start date.
    swap = _swap(LegKind.FIXED, LegKind.PRICE)
    leg = swap.pay_leg
    assert Calculator.get_leg_duration(swap, LegSide.PAY, market) == pytest.approx(
        _tau(leg.start_dates[0]), rel=1e-12
    )


def test_spread_lengthens_a_floating_leg_duration(market):
    plain = _swap(LegKind.FIXED, LegKind.OVERNIGHT)
    spread = _swap(LegKind.FIXED, LegKind.OVERNIGHT, spread=True)
    assert Calculator.get_leg_duration(
        spread, LegSide.PAY, market
    ) > Calculator.get_leg_duration(plain, LegSide.PAY, market)


@pytest.mark.parametrize("kind", [LegKind.TERM, LegKind.OVERNIGHT, LegKind.PRICE])
def test_started_floating_leg_duration_runs_to_the_next_reset(market, kind):
    swap = _swap(LegKind.FIXED, kind, status=SwapStatus.STARTED)
    leg = swap.pay_leg
    duration = Calculator.get_leg_duration(swap, LegSide.PAY, market)
    # The next reset is the start of the first coupon whose rate is not set yet: a
    # term coupon is set once fixed (here the Jul-Oct one fixes today), an overnight
    # one once started.
    set_dates = leg.fixing_dates if kind is LegKind.TERM else leg.start_dates
    first_unset = next(i for i, d in enumerate(set_dates) if d > VAL_DATE)
    assert duration == pytest.approx(_tau(leg.start_dates[first_unset]), rel=0.05)
    assert duration < 0.3


def test_leg_dv01_is_minus_pv_times_duration(market):
    swap = _swap(LegKind.FIXED, LegKind.PRICE)
    pv = Calculator.get_leg_pv(swap, LegSide.RECEIVE, market, CLP)
    assert Calculator.get_leg_dv01(swap, LegSide.RECEIVE, market, CLP) == pytest.approx(
        -pv * Calculator.get_leg_duration(swap, LegSide.RECEIVE, market) / 10_000
    )


def test_fixed_leg_dv01_is_negative_and_matches_a_curve_bump(market):
    swap = _swap(LegKind.FIXED, LegKind.PRICE)
    dv01 = Calculator.get_leg_dv01(swap, LegSide.RECEIVE, market, CLP)
    assert dv01 < 0
    base = Calculator.get_leg_pv(swap, LegSide.RECEIVE, market, CLP)
    curve = market.get_index_curve(ICP)
    market.set_curve(curve.parallel_bump_bps(1.0))
    try:
        bumped = Calculator.get_leg_pv(swap, LegSide.RECEIVE, market, CLP)
    finally:
        market.set_curve(curve)
    # The bump shifts continuous zero rates by 1bp: first order, the DV01 itself.
    assert bumped - base == pytest.approx(dv01, rel=0.01)


@pytest.mark.parametrize("kind", [LegKind.TERM, LegKind.OVERNIGHT, LegKind.PRICE])
def test_floating_leg_dv01_raises(market, kind):
    swap = _swap(LegKind.FIXED, kind)
    with pytest.raises(NotImplementedError, match="fixed legs"):
        Calculator.get_leg_dv01(swap, LegSide.PAY, market, CLP)


def test_matured_leg_has_no_duration_nor_dv01(market):
    swap = _swap(LegKind.FIXED, LegKind.FIXED, status=SwapStatus.FINISHED)
    assert Calculator.get_leg_duration(swap, LegSide.RECEIVE, market) == 0.0
    assert Calculator.get_leg_dv01(swap, LegSide.PAY, market, CLP) == 0.0


def test_swap_leg_returns_the_leg_on_each_side():
    swap = _swap(LegKind.FIXED, LegKind.OVERNIGHT)
    assert swap.leg(LegSide.RECEIVE) is swap.receive_leg
    assert swap.leg(LegSide.PAY) is swap.pay_leg


@pytest.mark.parametrize("not_a_side", ["receive", "PAY", None])
def test_swap_leg_rejects_anything_but_a_side(not_a_side):
    swap = _swap(LegKind.FIXED, LegKind.OVERNIGHT)
    with pytest.raises(TypeError, match="LegSide"):
        swap.leg(not_a_side)


def test_passing_a_leg_instead_of_a_side_raises(market):
    swap = _swap(LegKind.FIXED, LegKind.OVERNIGHT)
    with pytest.raises(TypeError, match="LegSide"):
        Calculator.get_leg_pv(swap, swap.pay_leg, market, CLP)
