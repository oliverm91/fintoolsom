from datetime import date
import pytest
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps import builders
from tests.factories.swap_factory import (
    SPREAD,
    TAB_3M,
    VAL_DATE,
    Ccy,
    adj_convention,
    build_market,
)



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


def test_fixed_term_coupon_keeps_its_notional_repayment():
    market = build_market()
    # 1Y leg: the last coupon (2024-04-12 → 2024-07-12) fixed before VAL_DATE, unpaid.
    leg = _leg(builders.term_rate_leg, TAB_3M, start=date(2023, 7, 12), years=1, spread=SPREAD)
    last = leg.coupons[-1]
    assert leg.fixing_dates[-1] < VAL_DATE < last.payment_date and last.amortization == 1_000.0

    _, flows = Calculator._leg_flows(leg, market)
    rate = market.get_index(TAB_3M.name).get_rate(leg.fixing_dates[-1]).rate
    days = (last.end_date - last.start_date).days
    expected = 1_000.0 * (rate + SPREAD.rate) * days / 360 + 1_000.0
    assert flows[-1] == pytest.approx(expected, rel=1e-12)
