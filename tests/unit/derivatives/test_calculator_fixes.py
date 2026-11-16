from datetime import date
import pytest
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps import builders
from tests.factories.swap_factory import (
    CAMARA,
    ICP,
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
    leg = _leg(
        builders.term_rate_leg, TAB_3M, start=date(2023, 7, 12), years=1, spread=SPREAD
    )
    last = leg.get_coupons()[-1]
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
    leg = _leg(
        builders.overnight_leg,
        index,
        start=date(2023, 7, 5),
        spread=SPREAD,
        payment_lag=2,
    )
    ended, accruing = leg.get_coupons()[3], leg.get_coupons()[4]
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
