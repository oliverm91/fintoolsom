from dataclasses import dataclass, field
from functools import partial
from datetime import date
from typing import Self

import numpy as np
from holidays import country_holidays, HolidayBase

from .. import settings

# Holiday categories added on top of public holidays, per country (ISO code).
# Bank holidays close the financial markets: CL (31 Dec) and JP (2-3 Jan, 31 Dec).
_EXTRA_CATEGORIES: dict[str, tuple[str, ...]] = {"CL": ("bank",), "JP": ("bank",)}

_DATE = date  # `date` is also a parameter name below


@dataclass(slots=True)
class Calendar:
    country: str = field(default=None)
    subdiv: str = field(default=None)
    custom_holidays: list[date] = field(default_factory=list)
    _calendar: HolidayBase = field(default=None)
    _extra_categories: tuple = field(default_factory=tuple)

    _weekend_weekdays: set[int] = field(init=False, default_factory=set)
    # fintoolsom._native.CalendarCore, built on first use under the "cpp" backend of
    # "dates": its days, with is_holiday, add_business_days and the conventions.
    _core: object = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        self._weekend_weekdays = {5, 6}

        if self._calendar is None:
            if self.custom_holidays is None:
                self.custom_holidays = []
            if self.country is None:
                self._calendar = HolidayBase()
            else:
                self._extra_categories = _EXTRA_CATEGORIES.get(self.country, ())
                self._calendar = country_holidays(self.country, subdiv=self.subdiv)
                if self._extra_categories:
                    # Rebuild once with the extra categories, keeping the subdivision.
                    categories = set(self._calendar.categories) | set(
                        self._extra_categories
                    )
                    self._calendar = country_holidays(
                        self.country, subdiv=self.subdiv, categories=categories
                    )

            self._calendar.update(self.custom_holidays)

    def add_custom_holiday(self, custom_holiday: date):
        self._calendar.append(custom_holiday)
        self._core = None  # its cached years may hold the date as a working day

    def add_custom_holidays(self, custom_holidays: list[date]):
        for ch in custom_holidays:
            self.add_custom_holiday(ch)

    def is_holiday(self, date: date) -> bool:
        if type(date) is _DATE and settings.is_cpp("dates"):
            return self._native_core().is_holiday(date)
        return date in self._calendar or date.weekday() in self._weekend_weekdays

    def add_business_days(self, t: date, business_days: int) -> date:
        if type(t) is _DATE and type(business_days) is int and settings.is_cpp("dates"):
            return self._native_core().add_business_days(t, business_days)
        return self._calendar.get_nth_working_day(t, business_days)

    def __getstate__(self) -> dict:
        # The C++ core does not pickle; a copy builds its own on first use.
        return {name: getattr(self, name) for name in self.__slots__ if name != "_core"}

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self._core = None

    def _native_core(self):
        """The C++ view of this calendar (``fintoolsom._native.CalendarCore``). It loads
        its days from :func:`_range_flags`, many years in one call: on first use, then
        only for a date outside the years it holds."""
        core = self._core
        if core is None:
            mask = sum(1 << d for d in self._weekend_weekdays)
            loader = partial(_range_flags, self._calendar)  # not self: no cycle
            core = self._core = settings.native().CalendarCore(loader, mask)
        return core

    def combine(self, other: Self) -> Self:
        combined_calendar = self._calendar + other._calendar
        return Calendar(_calendar=combined_calendar)

    def __add__(self, other: Self) -> Self:
        return self.combine(other)

    def __repr__(self) -> str:
        # The default dataclass repr would list every holiday of every loaded year.
        parts = [self.country if self.country is not None else "weekends only"]
        if self.subdiv:
            parts.append(f"subdiv={self.subdiv}")
        if self.custom_holidays:
            parts.append(f"{len(self.custom_holidays)} custom")
        return f"Calendar({', '.join(parts)})"


def _range_flags(holidays: HolidayBase, first_year: int, last_year: int) -> bytes:
    """Per day of the years `first_year`..`last_year` (inclusive): 1 if a holiday of
    `holidays`, 2 if one of its weekend days, 4 if one of its weekend working days
    (:meth:`HolidayBase.is_working_day`). One call loads every year a C++ core needs.

    The neighbouring years are populated too, so a holiday added by populating the
    next year (an observed 31 December) is already there.
    """
    for y in range(max(first_year - 1, 1), min(last_year + 1, 9999) + 1):
        date(y, 1, 1) in holidays  # noqa: B015 - populates the year (expand=True)
    first = date(first_year, 1, 1).toordinal()
    days = np.arange(first, date(last_year + 1, 1, 1).toordinal(), dtype=np.int64)
    flags = np.zeros(days.size, dtype=np.uint8)

    def mark(dates, bit: int) -> None:
        i = np.fromiter((d.toordinal() for d in dates), dtype=np.int64) - first
        flags[i[(i >= 0) & (i < flags.size)]] |= bit

    mark(holidays.keys(), 1)
    if type(holidays)._get_weekend is HolidayBase._get_weekend:
        weekend = np.fromiter(holidays.weekend, dtype=np.int64)
        flags[np.isin((days + 6) % 7, weekend)] |= 2
    else:  # a weekend that changes with the date
        flags[[holidays._is_weekend(date.fromordinal(int(o))) for o in days]] |= 2
    mark(holidays.weekend_workdays, 4)
    return flags.tobytes()


# One shared calendar per country (None: weekends only). Building a country calendar
# and filling its holiday years is costly, so every consumer that names a country
# gets the same instance; a holiday added to it is seen by all of them.
_CALENDARS: dict[str | None, Calendar] = {}


def get_calendar(country: str | None = None) -> Calendar:
    """The shared calendar of `country` (ISO code), built on first use."""
    calendar = _CALENDARS.get(country)
    if calendar is None:
        calendar = _CALENDARS[country] = Calendar(country=country)
    return calendar


def resolve_calendar(calendar: "Calendar | str | None") -> Calendar:
    """`calendar` itself, or the shared calendar of the country code it names."""
    return calendar if isinstance(calendar, Calendar) else get_calendar(calendar)
