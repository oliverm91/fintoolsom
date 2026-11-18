"""Discount curves used across the tests."""

import math
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta

from fintoolsom.rates import InterpolationMethod, PillarCurve

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


def curve_from_dfs(
    curve_date,
    pillars,
    dfs,
    interpolation_method=InterpolationMethod.PiecewiseConstant,
    **kwargs,
) -> PillarCurve:
    """:meth:`PillarCurve.from_dfs`, also for a single pillar, which is split in two
    segments with the same forward."""
    if len(pillars) == 1:
        days = (pillars[0] - curve_date).days
        pillars = [curve_date + timedelta(days=days // 2), pillars[0]]
        dfs = [dfs[0] ** ((days // 2) / days), dfs[0]]
    return PillarCurve.from_dfs(
        curve_date, list(pillars), list(dfs), interpolation_method, **kwargs
    )


def sample_curve(
    method: InterpolationMethod = InterpolationMethod.PiecewiseConstant,
) -> PillarCurve:
    """The sample CLP curve on SAMPLE_CURVE_DFS (log-linear dfs by default)."""
    dates, dfs = zip(*SAMPLE_CURVE_DFS)
    return PillarCurve.from_dfs(SAMPLE_CURVE_DATE, list(dates), list(dfs), method)


def flat_curve(
    t: date,
    rate: float,
    years: int = 30,
    method: InterpolationMethod = InterpolationMethod.PiecewiseConstant,
) -> PillarCurve:
    """Curve at a flat continuously compounded act/365 `rate`, yearly pillars to `years`."""
    pillars = [t + relativedelta(days=1)] + [
        t + relativedelta(years=y) for y in range(1, years + 1)
    ]
    dfs = [math.exp(-rate * (d - t).days / 365) for d in pillars]
    return PillarCurve.from_dfs(t, pillars, dfs, method)
