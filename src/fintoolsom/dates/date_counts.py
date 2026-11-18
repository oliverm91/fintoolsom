from abc import ABC, abstractmethod
from datetime import date
import calendar

import numpy as np


def date_ordinals(dates) -> np.ndarray:
    """Proleptic Gregorian ordinals of `dates`, as integers."""
    return np.array([d.toordinal() for d in dates], dtype=np.int64)


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
        cls,
        start_date: date | list[date],
        end_date: date | list[date],
        termination_date: date | None = None,
    ) -> int | np.ndarray:
        """Days from `start_date` to `end_date` (either may be a list).

        `termination_date`, the last date of the schedule, is read only by
        conventions with a rule for it (30E/360 ISDA); the others ignore it.
        """
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
        termination_date: date | None = None,
    ) -> float | np.ndarray:
        if termination_date is None:
            days = cls.get_day_count(start_date, end_date)
        else:
            days = cls.get_day_count(
                start_date, end_date, termination_date=termination_date
            )
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


def _is_last_of_february(d: date) -> bool:
    return d.month == 2 and d.day == calendar.monthrange(d.year, 2)[1]


def _d1_d2_30e_360_isda(
    start_date: date, end_date: date, is_termination: bool = False
) -> tuple[int, int]:
    """D1 and D2 of 30E/360 ISDA (2006 ISDA Definitions 4.16(h)).

    D1 is 30 when the start is the 31st or the last day of February. D2 is 30 when
    the end is the 31st, or the last day of February unless it is the termination
    date (`is_termination`).
    """
    d1, d2 = start_date.day, end_date.day
    if d1 == 31 or _is_last_of_february(start_date):
        d1 = 30
    if d2 == 31 or (not is_termination and _is_last_of_february(end_date)):
        d2 = 30
    return d1, d2


def _days_30e_360_isda(
    start_date: date, end_date: date, is_termination: bool = False
) -> int:
    """30E/360 ISDA day count; the one rule behind both 30E/360 ISDA classes."""
    d1, d2 = _d1_d2_30e_360_isda(start_date, end_date, is_termination)
    return (
        360 * (end_date.year - start_date.year)
        + 30 * (end_date.month - start_date.month)
        + (d2 - d1)
    )


class _ISDAToTermination:
    """Stands in for the class in the shape dispatchers, with the termination bound."""

    __slots__ = ("termination_date",)

    def __init__(self, termination_date: date):
        self.termination_date = termination_date

    def _get_day_count(self, start_date: date, end_date: date) -> int:
        return _days_30e_360_isda(
            start_date, end_date, end_date == self.termination_date
        )


class Days30EISDADayCountConvention(DayCountConventionBase):
    """30E/360 ISDA (2006 ISDA Definitions 4.16(h)).

    A start or an end on the 31st or on the last day of February counts as the 30th,
    except an end on the last day of February that is the termination date. Pass
    `termination_date` to `get_day_count` / `get_time_fraction` (legs pass their last
    end date): without it every end date is taken as an intermediate one.
    """

    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        return _days_30e_360_isda(start_date, end_date)

    @classmethod
    def get_day_count(
        cls,
        start_date: date | list[date],
        end_date: date | list[date],
        termination_date: date | None = None,
    ) -> int | np.ndarray:
        method = cls._method_mapper.get((type(start_date), type(end_date)))
        if termination_date is None or method is None:
            return super().get_day_count(start_date, end_date)
        return method(_ISDAToTermination(termination_date), start_date, end_date)

    @staticmethod
    def _strfy():
        return "30eISDA"
