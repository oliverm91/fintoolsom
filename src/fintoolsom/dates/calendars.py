from dataclasses import dataclass, field
from datetime import date
from typing import Self

from holidays import country_holidays, HolidayBase

# Holiday categories added on top of public holidays, per country (ISO code).
# Bank holidays close the financial markets: CL (31 Dec) and JP (2-3 Jan, 31 Dec).
_EXTRA_CATEGORIES: dict[str, tuple[str, ...]] = {"CL": ("bank",), "JP": ("bank",)}


@dataclass(slots=True)
class Calendar:
    """Business-day calendar: weekends plus the holidays of a country.

    A day is a holiday if it falls on a Saturday or Sunday, is a public holiday of
    `country` (plus its bank holidays for CL and JP), or is one of the custom
    holidays. Without a country only weekends and custom holidays close it.
    Calendars of several countries can be merged with `combine` or ``+``.

    Parameters
    ----------
    country : str, optional
        ISO 3166-1 country code (``"CL"``, ``"US"``). None for weekends only.
    subdiv : str, optional
        Subdivision of `country` (``"NY"`` for ``"US"``).
    custom_holidays : list of date, optional
        Extra holidays on top of the country's.

    Notes
    -----
    Use `get_calendar` or `resolve_calendar` to get the shared instance of a
    country: building one is costly, and a holiday added to a shared calendar is
    seen by every consumer.

    Examples
    --------
    >>> from datetime import date
    >>> cal = Calendar("CL")
    >>> cal.is_holiday(date(2026, 12, 8))
    True
    >>> cal.add_business_days(date(2026, 12, 9), -1)
    datetime.date(2026, 12, 7)
    """

    country: str = field(default=None)
    subdiv: str = field(default=None)
    custom_holidays: list[date] = field(default_factory=list)
    _calendar: HolidayBase = field(default=None)
    _extra_categories: tuple = field(default_factory=tuple)
    # The calendars it holds, as ISO 3166-2 labels ("CL", "US-NY"); a combined calendar
    # holds every input's, in order. Only for the repr.
    _countries: list[str] = field(default_factory=list)

    _weekend_weekdays: set[int] = field(init=False, default_factory=set)

    def __post_init__(self):
        self._weekend_weekdays = {5, 6}
        if not self._countries and self.country is not None:
            self._countries = [
                self.country if not self.subdiv else f"{self.country}-{self.subdiv}"
            ]

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
        """Make `custom_holiday` a holiday of this calendar."""
        self._calendar.append(custom_holiday)

    def add_custom_holidays(self, custom_holidays: list[date]):
        """Make each date of `custom_holidays` a holiday of this calendar."""
        for ch in custom_holidays:
            self.add_custom_holiday(ch)

    def is_holiday(self, date: date) -> bool:
        """True if `date` is a weekend day or a holiday (i.e. not a business day)."""
        return date in self._calendar or date.weekday() in self._weekend_weekdays

    def add_business_days(self, t: date, business_days: int) -> date:
        """The date `business_days` business days after `t` (before if negative)."""
        return self._calendar.get_nth_working_day(t, business_days)

    def combine(self, other: Self) -> Self:
        """New calendar that is closed whenever this one or `other` is."""
        combined_calendar = self._calendar + other._calendar
        return Calendar(
            custom_holidays=[
                *(self.custom_holidays or ()),
                *(other.custom_holidays or ()),
            ],
            _calendar=combined_calendar,
            _countries=list(dict.fromkeys([*self._countries, *other._countries])),
        )

    def __add__(self, other: Self) -> Self:
        """Same as `combine`."""
        return self.combine(other)

    def __repr__(self) -> str:
        # The default dataclass repr would list every holiday of every loaded year.
        parts = [" + ".join(self._countries) or "weekends only"]
        if self.custom_holidays:
            parts.append(f"{len(self.custom_holidays)} custom")
        return f"Calendar({', '.join(parts)})"


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
