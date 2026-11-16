from abc import ABC, abstractmethod
from datetime import date
import calendar

import numpy as np

from .. import settings


def date_ordinals(dates) -> np.ndarray:
    """Proleptic Gregorian ordinals of `dates`, as integers (``date.toordinal()``).

    Parameters
    ----------
    dates : iterable of date, or numpy.ndarray of shape (n, 3)
        Dates, or the integer ``(year, month, day)`` rows of `n` dates.

    Raises
    ------
    ValueError
        For an ``(n, 3)`` array with a row that is not a valid date.
    """
    if type(dates) is np.ndarray and dates.ndim == 2:
        return _ymd_ordinals(dates)
    if type(dates) in _SEQUENCES and settings.is_cpp("dates"):
        return settings.native().date_ordinals(dates)
    return np.array([d.toordinal() for d in dates], dtype=np.int64)


_SEQUENCES = (list, tuple)
_DAYS_IN_MONTH = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
# date(1970, 1, 1).toordinal(): days_from_civil counts from 1970-01-01.
_EPOCH_ORDINAL = 719163


def _ymd_ordinals(ymd: np.ndarray) -> np.ndarray:
    """Ordinals of ``(year, month, day)`` rows, vectorised (Howard Hinnant's
    ``days_from_civil``); the same on both backends."""
    if ymd.shape[1] != 3 or not np.issubdtype(ymd.dtype, np.integer):
        raise ValueError(
            f"Expected integer (year, month, day) rows, shape (n, 3); got shape "
            f"{ymd.shape} of {ymd.dtype}."
        )
    y, m, d = (ymd[:, k].astype(np.int64) for k in range(3))
    leap = (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))
    month_ok = (m >= 1) & (m <= 12)
    days_in_month = _DAYS_IN_MONTH[np.where(month_ok, m, 0)] + (leap & (m == 2))
    bad = ~((y >= 1) & (y <= 9999) & month_ok & (d >= 1) & (d <= days_in_month))
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(f"Row {i} is not a date: {ymd[i].tolist()}.")
    y = y - (m <= 2)
    era = y // 400
    yoe = y - era * 400
    doy = (153 * ((m + 9) % 12) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468 + _EPOCH_ORDINAL


def _gdc_dd(cls, sd, ed):
    return cls._get_day_count(sd, ed)


def _gdc_dl(cls, sd, eds):
    return np.array([cls._get_day_count(sd, ed) for ed in eds])


def _gdc_ld(cls, sds, ed):
    return np.array([cls._get_day_count(sd, ed) for sd in sds])


def _gdc_ll(cls, sds, eds):
    if len(sds) == len(eds):
        return np.array([cls._get_day_count(sd, ed) for sd, ed in zip(sds, eds)])
    else:
        raise ValueError(
            f"Length of start_dates and end_dates must match. Got lengths {len(sds)} and {len(eds)}"
        )


def _actual_dl(cls, sd, eds):
    return date_ordinals(eds) - sd.toordinal()


def _actual_ld(cls, sds, ed):
    return ed.toordinal() - date_ordinals(sds)


class DayCountConventionBase(ABC):
    _method_mapper = {
        (date, date): _gdc_dd,
        (date, list): _gdc_dl,
        (list, date): _gdc_ld,
        (list, list): _gdc_ll,
    }

    @staticmethod
    @abstractmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        pass

    @classmethod
    def get_day_count(
        cls, start_date: date | list[date], end_date: date | list[date]
    ) -> int | np.ndarray:
        if settings.is_cpp("dates") and (kind := _KINDS.get(cls)) is not None:
            days = settings.native().day_counts(kind, start_date, end_date)
            if days is not None:
                return days
        try:
            return cls._method_mapper[(type(start_date), type(end_date))](
                cls, start_date, end_date
            )
        except KeyError as _:
            raise TypeError(
                f"Input types start_date: {type(start_date)} and end_date: {type(end_date)} are not mapped. Mapped values:\n{list(cls._method_mapper.keys())}"
            )

    @classmethod
    def get_time_fraction(
        cls,
        start_date: date | list[date],
        end_date: date | list[date],
        time_fraction_base: int,
    ) -> float | np.ndarray:
        if (
            settings.is_cpp("dates")
            and type(time_fraction_base) in _BASES
            and time_fraction_base != 0
            and (kind := _KINDS.get(cls)) is not None
        ):
            fraction = settings.native().day_counts(
                kind, start_date, end_date, time_fraction_base
            )
            if fraction is not None:
                return fraction
        days = cls.get_day_count(start_date, end_date)
        return days / time_fraction_base

    @abstractmethod
    def _strfy() -> str:
        pass

    @classmethod
    def __str__(cls) -> str:
        return cls._strfy()


class ActualDayCountConvention(DayCountConventionBase):
    _method_mapper = {
        **DayCountConventionBase._method_mapper,
        (date, list): _actual_dl,
        (list, date): _actual_ld,
    }

    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        return (end_date - start_date).days

    @staticmethod
    def _strfy():
        return "act"


class Days30ADayCountConvention(DayCountConventionBase):
    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        d1, d2 = start_date.day, end_date.day
        m1, m2 = start_date.month, end_date.month
        y1, y2 = start_date.year, end_date.year

        d1 = min(d1, 30)
        d2 = min(d2, 30) if d1 > 29 else d2
        count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return count

    @staticmethod
    def _strfy():
        return "30a"


class Days30UDayCountConvention(DayCountConventionBase):
    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        d1, d2 = start_date.day, end_date.day
        m1, m2 = start_date.month, end_date.month
        y1, y2 = start_date.year, end_date.year
        # 30U/360 with the end-of-month rule (Strata THIRTY_U_360_EOM, SIA):
        # 1. both dates last of February -> D2 = 30; 2. D1 last of February -> D1 = 30;
        # 3. D2 = 31 and D1 >= 30 -> D2 = 30; 4. D1 = 31 -> D1 = 30.
        start_last_of_february = (
            start_date.month == 2 and d1 == calendar.monthrange(y1, m1)[1]
        )
        end_last_of_february = (
            end_date.month == 2 and d2 == calendar.monthrange(y2, m2)[1]
        )
        if start_last_of_february and end_last_of_february:
            d2 = 30
        if start_last_of_february:
            d1 = 30
        if d2 == 31 and d1 >= 30:
            d2 = 30
        if d1 == 31:
            d1 = 30

        count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return count

    @staticmethod
    def _strfy():
        return "30u"


class Days30EDayCountConvention(DayCountConventionBase):
    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        d1, d2 = start_date.day, end_date.day
        m1, m2 = start_date.month, end_date.month
        y1, y2 = start_date.year, end_date.year

        d1 = min(d1, 30)
        d2 = min(d2, 30)

        count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return count

    @staticmethod
    def _strfy():
        return "30e"


class Days30EISDADayCountConvention(DayCountConventionBase):
    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        d1, d2 = start_date.day, end_date.day
        m1, m2 = start_date.month, end_date.month
        y1, y2 = start_date.year, end_date.year
        start_date_month_end_day = calendar.monthrange(y1, m1)[1]
        end_date_month_end_day = calendar.monthrange(y2, m2)[1]
        if d1 == start_date_month_end_day:
            d1 = 30
        if d2 == end_date_month_end_day and m2 != 2:
            d2 = 30
        count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return count

    @staticmethod
    def _strfy():
        return "30eISDA"


# The kinds of fintoolsom._native.day_counts, by class (subclasses stay in Python).
_KINDS: dict[type, int] = {
    ActualDayCountConvention: 0,
    Days30ADayCountConvention: 1,
    Days30UDayCountConvention: 2,
    Days30EDayCountConvention: 3,
    Days30EISDADayCountConvention: 4,
}
_BASES = (int, float)
