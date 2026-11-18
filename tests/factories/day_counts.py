"""Helpers and reference tables for day counts.

The 30/360 tables are shared by the two implementations (``dates.date_counts`` day
counts and ``dates.time_fractions`` year fractions), so both are checked on the same
rows. Values are Strata's (``DayCountTest.java``), copied by hand.
"""

from datetime import date

D = date


def calc360(y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> int:
    """Raw 30/360 day count with the day numbers given (Strata ``calc360``).

    Day numbers may be 30 in February: they are the already-adjusted D1/D2.
    """
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)


# (start, end, days)
THIRTY_E_360 = [
    (D(2011, 2, 28), D(2012, 2, 29), calc360(2011, 2, 28, 2012, 2, 29)),
    (D(2012, 2, 29), D(2016, 2, 29), calc360(2012, 2, 29, 2016, 2, 29)),
    (D(2012, 5, 30), D(2013, 8, 29), calc360(2012, 5, 30, 2013, 8, 29)),
    (D(2012, 5, 29), D(2013, 8, 31), calc360(2012, 5, 29, 2013, 8, 30)),
    (D(2012, 5, 30), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 30), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
]

# (start, end, is_termination, days): Strata data_30E360ISDA, its "maturity" column
# (the end is the termination date) and its other column. A start on the last of
# February always moves to 30; an end on it only when it is not the termination date.
THIRTY_E_360_ISDA = [
    # End is the termination date.
    (D(2011, 12, 28), D(2012, 2, 29), True, calc360(2011, 12, 28, 2012, 2, 29)),
    (D(2012, 2, 29), D(2012, 3, 28), True, calc360(2012, 2, 30, 2012, 3, 28)),
    (D(2011, 2, 28), D(2012, 2, 28), True, calc360(2011, 2, 30, 2012, 2, 28)),
    (D(2011, 2, 28), D(2012, 2, 29), True, calc360(2011, 2, 30, 2012, 2, 29)),
    (D(2012, 2, 29), D(2016, 2, 29), True, calc360(2012, 2, 30, 2016, 2, 29)),
    (D(2012, 5, 29), D(2013, 8, 31), True, calc360(2012, 5, 29, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), True, calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2024, 1, 31), D(2024, 2, 29), True, 29),
    # End is an intermediate date.
    (D(2011, 12, 28), D(2012, 2, 28), False, calc360(2011, 12, 28, 2012, 2, 28)),
    (D(2011, 12, 28), D(2012, 2, 29), False, calc360(2011, 12, 28, 2012, 2, 30)),
    (D(2012, 2, 29), D(2012, 3, 28), False, calc360(2012, 2, 30, 2012, 3, 28)),
    (D(2011, 2, 28), D(2012, 2, 28), False, calc360(2011, 2, 30, 2012, 2, 28)),
    (D(2011, 2, 28), D(2012, 2, 29), False, calc360(2011, 2, 30, 2012, 2, 30)),
    (D(2012, 2, 29), D(2016, 2, 29), False, calc360(2012, 2, 30, 2016, 2, 30)),
    (D(2012, 5, 29), D(2013, 8, 31), False, calc360(2012, 5, 29, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), False, calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2024, 1, 31), D(2024, 2, 29), False, 30),
]
