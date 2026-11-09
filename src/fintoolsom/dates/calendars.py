from dataclasses import dataclass, field
from datetime import date
from typing import Self

from holidays import country_holidays, HolidayBase

# Holiday categories added on top of public holidays, per country (ISO code).
# Bank holidays close the financial markets: CL (31 Dec) and JP (2-3 Jan, 31 Dec).
_EXTRA_CATEGORIES: dict[str, tuple[str, ...]] = {"CL": ("bank",), "JP": ("bank",)}


@dataclass(slots=True)
class Calendar:
    country: str = field(default=None)
    subdiv: str = field(default=None)
    custom_holidays: list[date] = field(default_factory=list)
    _calendar: HolidayBase = field(default=None)
    _extra_categories: tuple = field(default_factory=tuple)

    _weekend_weekdays: set[int] = field(init=False, default_factory=set)

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

    def add_custom_holidays(self, custom_holidays: list[date]):
        for ch in custom_holidays:
            self.add_custom_holiday(ch)

    def is_holiday(self, date: date) -> bool:
        return date in self._calendar or date.weekday() in self._weekend_weekdays

    def add_business_days(self, t: date, business_days: int) -> date:
        return self._calendar.get_nth_working_day(t, business_days)

    def combine(self, other: Self) -> Self:
        combined_calendar = self._calendar + other._calendar
        return Calendar(_calendar=combined_calendar)

    def __add__(self, other: Self) -> Self:
        return self.combine(other)


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
