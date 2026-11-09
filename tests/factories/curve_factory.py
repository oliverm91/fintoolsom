"""Zero-coupon curves used across the tests."""

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from fintoolsom.rates import InterpolationMethod, ZeroCouponCurve

SAMPLE_CURVE_DATE = date(2024, 7, 8)

# A realistic, slightly bumpy CLP discount curve as of SAMPLE_CURVE_DATE.
SAMPLE_CURVE_DFS = [
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


def sample_zero_coupon_curve(
    method: InterpolationMethod | None = None,
) -> ZeroCouponCurve:
    """The sample CLP curve (log-linear unless `method` is given)."""
    kwargs = {} if method is None else {"df_interpolation_method": method}
    return ZeroCouponCurve(SAMPLE_CURVE_DATE, date_dfs=list(SAMPLE_CURVE_DFS), **kwargs)


def flat_zero_coupon_curve(
    t: date, rate: float, years: int = 30, method: InterpolationMethod | None = None
) -> ZeroCouponCurve:
    """Curve at a flat continuously compounded act/365 `rate`, yearly pillars to `years`."""
    pillars = [t + relativedelta(days=1)] + [
        t + relativedelta(years=y) for y in range(1, years + 1)
    ]
    date_dfs = [(d, math.exp(-rate * (d - t).days / 365)) for d in pillars]
    kwargs = {} if method is None else {"df_interpolation_method": method}
    return ZeroCouponCurve(t, date_dfs=date_dfs, **kwargs)
