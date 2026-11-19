"""Term-rate coupons project the index forward over their fixing window.

A term-rate coupon pays the index fixing of its fixing date, accrued over its own
period. That fixing covers the index tenor from its value date, ``s = fixing +
spot_lag`` to ``e = index.get_maturity(s)``: before it fixes, the coupon projects the
index curve's forward over ``[s, e]`` and accrues it over ``[start, end]`` in the
index convention. A stub or a period adjusted off the tenor gets the index's rate,
not the curve's rate over its own length.
"""

from datetime import date

import numpy as np
import pytest

from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import Calculator, ValuationBatch
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.legs import TermRateLeg
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.market import Currency, Market, OvernightInterestPriceIndex
from fintoolsom.market.index import TermRateIndex
from fintoolsom.rates import (
    CompoundedInterestConvention,
    InterpolationMethod,
    PillarCurve,
    Rate,
    RateConvention,
)
from tests.factories.swap_factory import TAB_3M, Ccy, adj_convention

T = date(2024, 7, 8)
ADJ = adj_convention(Ccy.CLP)
ICP = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
TAB_COMP = TermRateIndex(  # a 3M term rate quoted compounded Act/365
    "TAB_COMP",
    currency=Currency.CLP,
    term=Term(3, TermUnit.M, ADJ),
    rate_convention=RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    ),
)
PC = InterpolationMethod.PiecewiseConstant


def _steep(t=T):
    # 4% for three months, then 6%: a window crossing 2024-10-08 is far from the
    # curve's rate over a shorter period.
    return PillarCurve(t, [t, date(2024, 10, 8), date(2030, 7, 8)], [0.04, 0.06], PC)


def _market(t=T, curve=None):
    curve = _steep(t) if curve is None else curve
    icp = PillarCurve(t, [t, date(2025, 7, 8), date(2035, 7, 8)], [0.05, 0.05], PC)
    return Market(
        t=t, risk_free_index=ICP, curves={ICP: icp, TAB_3M: curve, TAB_COMP: curve}
    )


def _leg(index=TAB_3M, months=13, start=date(2024, 7, 15)) -> TermRateLeg:
    # 13 months quarterly: a 1M first stub.
    return TermRateLeg.from_term(
        1e9, start, Term(months, TermUnit.M, ADJ), "QUARTERLY", ADJ, index
    )


def _window(leg, i):
    index = leg.index
    s = index.calendar.add_business_days(leg.fixing_dates[i], index.spot_lag)
    return s, index.get_maturity(s)


def _expected_interest(leg, curve):
    interest = []
    for i, c in enumerate(leg.coupons):
        s, e = _window(leg, i)
        forward = curve.get_forward_rate(s, e, leg.index.rate_convention)
        interest.append(
            forward.get_accrued_interest(c.residual, c.start_date, c.end_date)
        )
    return np.array(interest)


@pytest.mark.parametrize("index", [TAB_3M, TAB_COMP], ids=["linear", "compounded"])
def test_a_coupon_accrues_the_index_forward_of_its_fixing_window(index):
    leg, market = _leg(index), _market()
    _, flows = Calculator._leg_flows(leg, market)
    interest = flows - leg.fixed_flows
    np.testing.assert_allclose(interest, _expected_interest(leg, _steep()), rtol=1e-12)


def test_a_stub_gets_the_three_month_rate_not_a_one_month_rate():
    leg, market = _leg(), _market()
    stub = leg.coupons[0]
    assert (stub.end_date - stub.start_date).days == 32
    s, e = _window(leg, 0)
    assert (s, e) == (stub.start_date, date(2024, 10, 15))
    _, flows = Calculator._leg_flows(leg, market)
    three_month = _steep().get_forward_rate(s, e, TAB_3M.rate_convention).rate
    paid = (flows[0] - leg.fixed_flows[0]) / stub.residual * 360 / 32
    assert paid == pytest.approx(three_month, rel=1e-12)
    # The curve's own rate over the 32 days is ~16bp lower on this curve.
    own = _steep().get_forward_rate(
        stub.start_date, stub.end_date, TAB_3M.rate_convention
    )
    assert three_month - own.rate > 0.0015


def test_a_regular_leg_projects_over_its_periods_as_before():
    leg = _leg(months=12)
    assert leg._projects_on_accrual and leg._par_eligible
    _, flows = Calculator._leg_flows(leg, _market())
    curve = _steep()
    expected = [
        c.residual * (curve.get_wealth_factor(c.start_date, c.end_date) - 1)
        for c in leg.coupons
    ]
    np.testing.assert_allclose(flows - leg.fixed_flows, expected, rtol=1e-14)


def test_an_off_window_leg_is_not_valued_at_par():
    leg, market = _leg(), _market()
    assert not leg._projects_on_accrual and not leg._par_eligible
    # Discounted on its own index curve (collateral TAB): the flows, not the par shortcut.
    pv = Calculator._leg_pv(leg, TAB_3M, market)
    _, flows = Calculator._leg_flows(leg, market)
    dfs = _steep().get_dfs(leg.payment_dates)
    assert pv == pytest.approx(float(np.sum(flows * dfs)), rel=1e-14)


def _swap(index=TAB_3M, collateral=None):
    floating = _leg(index)
    fixed = builders.fixed_leg(
        rate=Rate(RateConvention(), 0.05),
        currency=Currency.CLP,
        notional=1e9,
        start_date=date(2024, 7, 15),
        term=Term(13, TermUnit.M, ADJ),
        frequency="QUARTERLY",
        adj_convention=ADJ,
    )
    return Swap(receive_leg=fixed, pay_leg=floating, collateral_index=collateral)


@pytest.mark.parametrize("index", [TAB_3M, TAB_COMP], ids=["linear", "compounded"])
@pytest.mark.parametrize("collateral", [None, ICP], ids=["uncoll", "coll_icp"])
def test_the_batch_values_off_window_legs_as_the_calculator_to_the_bit(
    index, collateral
):
    market = _market()
    swaps = [_swap(index, collateral), _swap(TAB_3M, collateral)]
    one_by_one = [Calculator.valuate(s, market, Currency.CLP) for s in swaps]
    batch = ValuationBatch(swaps, market, Currency.CLP)
    assert batch.differentiable
    assert batch.values().tolist() == one_by_one


@pytest.mark.parametrize("index", [TAB_3M, TAB_COMP], ids=["linear", "compounded"])
def test_the_batch_gradient_matches_a_curve_bump(index):
    market = _market()
    batch = ValuationBatch([_swap(index)], market, Currency.CLP)
    base_values, base_mtm = batch.source_values(), batch.values()[0]
    gradient = batch.gradient(base_values)[0]

    bumped = PillarCurve(
        T, [T, date(2024, 10, 8), date(2030, 7, 8)], [0.0401, 0.0602], PC
    )
    market.set_curve(index, bumped)
    try:
        moved = batch.source_values() - base_values
        change = batch.values()[0] - base_mtm
    finally:
        market.set_curve(index, _steep())
    assert gradient @ moved == pytest.approx(change, rel=1e-3)
