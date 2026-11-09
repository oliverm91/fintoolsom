import warnings
from datetime import date

import pytest

from fintoolsom.dates import ModifiedFollowingConvention, Term, TermUnit
from fintoolsom.derivatives.swaps import TermRateCoupon, TermRateLeg
from fintoolsom.market import Currency, TermRateIndex
from fintoolsom.rates import Rate, RateConvention

INDEX = TermRateIndex(
    "TSOFR3M",
    currency=Currency.USD,
    term=Term(3, TermUnit.M, ModifiedFollowingConvention("US")),
)
SPREAD = Rate(RateConvention(), 0.0)
START = date(2026, 6, 1)


def _leg(payment_date: date) -> TermRateLeg:
    coupon = TermRateCoupon(1e6, 1e6, START, payment_date, payment_date, spread=SPREAD)
    return TermRateLeg(
        coupons=[coupon], index=INDEX, spread=SPREAD, fixing_dates=[START]
    )


@pytest.mark.parametrize(
    "payment_date",
    [
        date(2026, 9, 5),  # a Saturday
        date(2026, 12, 25),  # a weekday holiday of the US calendar
    ],
)
def test_warns_when_a_payment_date_is_an_index_holiday(payment_date):
    with pytest.warns(UserWarning, match=payment_date.isoformat()):
        _leg(payment_date)


def test_does_not_warn_on_a_business_day():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _leg(date(2026, 9, 4))
