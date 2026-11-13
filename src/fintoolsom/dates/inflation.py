"""Chilean CPI release and UF publication dates.

The CPI of month *m* is released on the business day before the 9th of month
*m+1* (the 8th, or the preceding business day), which fixes the UF from the 10th
of *m+1* to the 9th of *m+2*. Before the release the UF is known up to the 9th of
the current month; from it, up to the 9th of next month.
"""

from __future__ import annotations

from datetime import date

from .calendars import resolve_calendar

_CL = "CL"


def _ninth(year: int, month: int, months_ahead: int = 0) -> date:
    index = year * 12 + (month - 1) + months_ahead
    return date(index // 12, index % 12 + 1, 9)


def last_known_uf_date(t: date) -> date:
    """Last date whose UF is published as of `t`.

    The 9th of the current month before this month's CPI release, the 9th of next
    month from it (the release day included).

    Parameters
    ----------
    t : date
        As-of date.

    Returns
    -------
    date
    """
    calendar = resolve_calendar(_CL)
    this_ninth = _ninth(t.year, t.month)
    if t < calendar.add_business_days(this_ninth, -1):
        return this_ninth
    return _ninth(t.year, t.month, 1)


def next_cpi_publication_date(t: date) -> date:
    """Next CPI release after `t`: one business day before ``last_known_uf_date(t)``.

    On a release day that release is already out, so the next one is returned.

    Parameters
    ----------
    t : date
        As-of date.

    Returns
    -------
    date
    """
    calendar = resolve_calendar(_CL)
    return calendar.add_business_days(last_known_uf_date(t), -1)
