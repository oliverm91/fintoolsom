from datetime import date

import pytest
from fintoolsom.derivatives.swaps.coupons import FixedCoupon
from tests.factories.swap_factory import (
    _rate,
)


def test_fixed_coupon_carries_fx_fixing_date():
    d = date(2025, 1, 10)
    c = FixedCoupon(
        100.0,
        0.0,
        date(2024, 10, 10),
        d,
        d,
        rate=_rate(0.05),
        fx_fixing_date=date(2025, 1, 8),
    )
    assert c.fx_fixing_date == date(2025, 1, 8)


# ── 30E/360 ISDA: the termination date keeps a last-of-February end (B9) ─────


def _isda_rate(value: float):
    from fintoolsom.dates import Days30EISDADayCountConvention
    from fintoolsom.rates import LinearInterestConvention, Rate, RateConvention

    return Rate(
        RateConvention(LinearInterestConvention, Days30EISDADayCountConvention, 360),
        value,
    )


def test_fixed_coupon_ending_on_the_termination_date_keeps_the_last_of_february():
    start, end = date(2024, 1, 31), date(2024, 2, 29)
    rate = _isda_rate(0.36)  # 0.1% a day of 30/360
    intermediate = FixedCoupon(100.0, 0.0, start, end, end, rate=rate)
    final = FixedCoupon(100.0, 0.0, start, end, end, rate=rate, termination_date=end)
    # 29 Feb counts as the 30th, unless it is the termination date.
    assert intermediate.flow == pytest.approx(100.0 * 0.36 * 30 / 360, rel=1e-14)
    assert final.flow == pytest.approx(100.0 * 0.36 * 29 / 360, rel=1e-14)


def test_fixed_leg_from_term_passes_its_termination_date_to_every_coupon():
    from fintoolsom.dates import ModifiedFollowingConvention
    from fintoolsom.dates.term import Term, TermUnit
    from fintoolsom.derivatives.swaps.legs import FixedLeg
    from fintoolsom.market import Currency

    adj = ModifiedFollowingConvention()
    maturity = date(2028, 2, 29)
    rate = _isda_rate(0.05)
    leg = FixedLeg.from_term(
        1_000_000.0,
        date(2025, 8, 29),
        Term(2, TermUnit.Y, adj),
        "SEMIANNUAL",
        adj,
        rate,
        Currency.USD,
        maturity_date=maturity,
    )
    assert leg.end_dates[-1] == maturity
    assert {c.termination_date for c in leg.coupons} == {maturity}
    last = leg.coupons[-1]
    expected = rate.get_accrued_interest(
        last.residual, last.start_date, maturity, maturity
    )
    assert last.flow == float(expected) + last.amortization
    # Without the termination rule 29 Feb would count as the 30th: one day more.
    assert rate.get_accrued_interest(
        last.residual, last.start_date, maturity
    ) - expected == pytest.approx(1_000_000.0 * 0.05 / 360, rel=1e-9)
