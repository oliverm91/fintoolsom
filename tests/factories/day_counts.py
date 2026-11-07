"""Helpers for day-count tables."""
from datetime import date

D = date


def calc360(y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> int:
    """Raw 30/360 day count with the day numbers given (Strata ``calc360``).

    Day numbers may be 30 in February: they are the already-adjusted D1/D2.
    """
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
