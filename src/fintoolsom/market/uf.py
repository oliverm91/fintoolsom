"""Chilean UF (Unidad de Fomento) / CPI mechanics.

The UF is the currency ``Currency.CLF``; its daily levels (CLP per UF) are the
CLF/CLP exchange rate and are loaded in the market like any FX rate (see
:func:`uf_rate_data`). This module only holds the UF <-> CPI arithmetic used to
produce or analyse those levels; valuation never needs it.

Within a reajuste period (day 10 of a month to day 9 of the next, anchored at the
9th) the UF grows geometrically by a monthly CPI, rounded to 2 decimals. The CPI
of month *m* is published around the 8th of month *m+1* and governs the period
from the 10th of *m+1* to the 9th of *m+2*, so levels are known ahead of time up
to the 9th of next month (see :func:`last_published_date`).

Levels are passed as ``dict[date, float]`` throughout.
"""

from __future__ import annotations

from datetime import date, timedelta

from ..dates import Calendar, PrecedingConvention, get_calendar
from .currencies import Currency, CurrencyPair, FX_Rate, FX_RateData
from .localities import Locality


def _shift_month(year: int, month: int, k: int) -> tuple[int, int]:
    """Add ``k`` calendar months to ``(year, month)``, returning ``(year, month)``."""
    index = year * 12 + (month - 1) + k
    return index // 12, index % 12 + 1


def _next_ninth(base_9: date) -> date:
    return date(*_shift_month(base_9.year, base_9.month, 1), 9)


class UFConvention:
    """Stateless UF reajuste formulas."""

    @staticmethod
    def reajuste(
        base_level: float, cpi: float, elapsed_days: int, period_days: int
    ) -> float:
        """UF level `elapsed_days` after the base 9th, rounded to 2 decimals.

        Parameters
        ----------
        base_level : float
            UF on the base 9th.
        cpi : float
            Monthly CPI variation, applied geometrically.
        elapsed_days : int
            Days since the base 9th.
        period_days : int
            Days in the reajuste period.

        Returns
        -------
        float
        """
        return round(base_level * (1.0 + cpi) ** (elapsed_days / period_days), 2)

    @staticmethod
    def implied_monthly_cpi(level_start: float, level_end: float) -> float:
        """Monthly CPI implied by the UF on two consecutive 9ths (a full period).

        Parameters
        ----------
        level_start, level_end : float
            UF levels on consecutive 9ths.

        Returns
        -------
        float
        """
        return level_end / level_start - 1.0

    @staticmethod
    def implied_monthly_cpi_between(
        base_9: date,
        base_level: float,
        end_date: date,
        end_level: float,
    ) -> float:
        """Monthly CPI implied by the UF on a base 9th and a later date in its period.

        The partial-period growth is scaled to a full month::

            cpi = (end_level / base_level) ** (period_days / elapsed_days) - 1

        Parameters
        ----------
        base_9 : date
            The 9th of a month (a mid-period base loses precision to rounding).
        base_level : float
            UF on `base_9`.
        end_date : date
            Later date, up to and including the next 9th.
        end_level : float
            UF on `end_date`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If `base_9` is not a 9th, equals `end_date`, or `end_date` is past the
            next 9th (the growth would compound two months' CPI).
        """
        if base_9.day != 9:
            raise ValueError(f"base_9 must be the 9th of a month; got {base_9}.")
        if base_9 == end_date:
            raise ValueError("base_9 and end_date must differ.")
        next_ninth = _next_ninth(base_9)
        if end_date > next_ninth:
            raise ValueError(
                f"{base_9} and {end_date} straddle the reajuste boundary "
                f"{next_ninth}; the implied figure would compound more than one "
                f"month's CPI."
            )
        period_days = (next_ninth - base_9).days
        elapsed_days = (end_date - base_9).days
        return (end_level / base_level) ** (period_days / elapsed_days) - 1.0


def last_published_date(today: date, calendar: Calendar | None = None) -> date:
    """Last date whose UF is published as of `today`.

    CPI is released on the 8th (or the preceding business day). Before the
    release the UF is known up to the 9th of the current month; from it, up to
    the 9th of next month. Load levels only up to this date so the market treats
    them as known and projects later ones from the curves.

    Parameters
    ----------
    today : date
        As-of date.
    calendar : Calendar, optional
        Release calendar; defaults to the Chilean calendar.

    Returns
    -------
    date
    """
    if calendar is None:
        calendar = get_calendar(Locality.CL.value)
    publication_date = PrecedingConvention(calendar).adjust(
        date(today.year, today.month, 8)
    )
    if today < publication_date:
        year, month = today.year, today.month
    else:
        year, month = _shift_month(today.year, today.month, 1)
    return date(year, month, 9)


def monthly_inflation(levels: dict[date, float], year: int, month: int) -> float:
    """Realised CPI of calendar month ``(year, month)`` implied by UF levels.

    Equals ``UF(9 of m+2) / UF(9 of m+1) - 1``, the reajuste window over which
    that month's CPI is applied.

    Parameters
    ----------
    levels : dict of date to float
        UF levels.
    year, month : int
        Calendar month.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If a boundary level is not loaded (the CPI is not published yet).
    """

    def boundary(k: int) -> float:
        d = date(*_shift_month(year, month, k), 9)
        if d not in levels:
            raise ValueError(
                f"UF level for reajuste boundary {d} is not loaded; the inflation "
                "spanning it has not been published / is not known yet."
            )
        return levels[d]

    return UFConvention.implied_monthly_cpi(boundary(1), boundary(2))


def extend_with_cpi(
    levels: dict[date, float], cpi: float, base_date: date | None = None
) -> dict[date, float]:
    """UF levels filled over one reajuste period from a monthly CPI.

    Each day `d` after the base 9th, up to the next 9th, is set to::

        UF_d = UF_base * (1 + cpi) ** (days(base -> d) / days(base -> next 9th))

    e.g. ``UF(13-Mar) = UF(9-Mar) * (1 + cpi) ** (4 / 31)``. Existing levels are
    kept; only missing days are filled.

    Parameters
    ----------
    levels : dict of date to float
        UF levels (not modified).
    cpi : float
        CPI governing the period (decimal, ``0.004`` for 0.4%): the variation of
        the month before the base month.
    base_date : date, optional
        The 9th the period grows from; defaults to the latest loaded 9th.

    Returns
    -------
    dict of date to float
        A copy of `levels` with the period filled.

    Raises
    ------
    ValueError
        If `base_date` is not a 9th, or no 9th is loaded.
    KeyError
        If the base level is not loaded.
    """
    if base_date is None:
        ninths = [d for d in levels if d.day == 9]
        if not ninths:
            raise ValueError("No 9th-day UF level loaded to use as the reajuste base.")
        base_date = max(ninths)
    elif base_date.day != 9:
        raise ValueError(f"base_date must fall on a 9th; got {base_date}.")
    if base_date not in levels:
        raise KeyError(f"UF base level for {base_date} is not loaded.")

    out = dict(levels)
    base_uf = levels[base_date]
    next_ninth = _next_ninth(base_date)
    period_days = (next_ninth - base_date).days
    d = base_date + timedelta(days=1)
    while d <= next_ninth:
        if d not in out:
            out[d] = UFConvention.reajuste(
                base_uf, cpi, (d - base_date).days, period_days
            )
        d += timedelta(days=1)
    return out


def uf_rate_data(levels: dict[date, float]) -> FX_RateData:
    """CLF/CLP FX data from UF levels, ready for ``Market.fx_history``.

    Parameters
    ----------
    levels : dict of date to float
        UF levels (CLP per UF), realised and already published.

    Returns
    -------
    FX_RateData
        Keyed by ``CurrencyPair(Currency.CLF, Currency.CLP)``; the market adds
        the inverse pair.
    """
    pair = CurrencyPair(Currency.CLF, Currency.CLP)
    return FX_RateData(pair, {d: FX_Rate(pair, v) for d, v in levels.items()})
