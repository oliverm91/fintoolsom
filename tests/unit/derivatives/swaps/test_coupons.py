from datetime import date
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
