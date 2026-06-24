from datetime import date

import pytest

from fintoolsom.rates import ZeroCouponCurve
from fintoolsom.rates.Rates import RateConvention, LinearInterestConvention
from fintoolsom.dates import Days30EDayCountConvention
from fintoolsom.fixedIncome import Bond, Coupon, Coupons


@pytest.fixture
def curve_date() -> date:
    return date(2024, 7, 8)


@pytest.fixture
def sample_zero_coupon_curve(curve_date) -> ZeroCouponCurve:
    curve_dfs = [
        (date(2024, 7, 11), 0.99982),
        (date(2024, 7, 17), 0.99875),
        (date(2024, 7, 24), 0.99746),
        (date(2024, 7, 31), 0.99618),
        (date(2024, 8, 10), 0.99437),
        (date(2024, 9, 10), 0.99534),
        (date(2024, 10, 10), 0.99339),
        (date(2025, 1, 10), 0.99397),
        (date(2025, 4, 10), 0.98939),
        (date(2025, 7, 10), 0.98432),
        (date(2026, 1, 10), 0.97194),
        (date(2026, 7, 10), 0.95984),
        (date(2027, 7, 10), 0.93721),
        (date(2028, 7, 10), 0.91496),
        (date(2029, 7, 10), 0.89386),
        (date(2031, 7, 10), 0.85234),
        (date(2032, 7, 10), 0.83147),
        (date(2034, 7, 10), 0.79863),
        (date(2036, 7, 10), 0.77565),
    ]
    return ZeroCouponCurve(curve_date, date_dfs=curve_dfs)


@pytest.fixture
def sample_bond() -> Bond:
    """A 1-year Bond with two semi-annual coupons, built by hand (no fixed_income.db)."""
    coupon_rate_convention = RateConvention(
        LinearInterestConvention, Days30EDayCountConvention, 360
    )
    coupons = Coupons(
        [
            Coupon(
                0, 2, 100, date(2024, 1, 1), date(2024, 7, 1), coupon_rate_convention
            ),
            Coupon(
                100, 2, 100, date(2024, 7, 1), date(2025, 1, 1), coupon_rate_convention
            ),
        ],
        check_residuals=True,
    )
    return Bond(coupons=coupons, currency="clp", notional=100_000_000)
