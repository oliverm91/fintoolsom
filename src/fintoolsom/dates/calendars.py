from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date
from typing import Self

from holidays import (
    HolidayBase,
    country_holidays,
    financial_holidays,
    list_supported_financial,
)

# Codes of the financial-market calendars of the holidays library ("XECB" is TARGET2,
# "XNYS" the NYSE); any other code is a country.
_FINANCIAL_MARKETS = frozenset(list_supported_financial())

# Holiday categories added on top of public holidays, per country (ISO code).
# Bank holidays close the financial markets: CL (31 Dec) and JP (2-3 Jan, 31 Dec).
_EXTRA_CATEGORIES: dict[str, tuple[str, ...]] = {"CL": ("bank",), "JP": ("bank",)}

# Bumped by every change to the custom holidays of any calendar. A calendar keeps the
# year tables it built at one epoch and rebuilds them on first use at a later one, so a
# holiday added to a calendar also reaches the combined calendars made from it.
_epoch = 0


@dataclass(slots=True)
class Calendar:
    """Business-day calendar: weekends plus the holidays of a country or a market.

    A day is a holiday if it falls on a Saturday or Sunday, is a public holiday of
    `country` (plus its bank holidays for CL and JP), or is one of the custom
    holidays. Without a country only weekends and custom holidays close it.
    Calendars of several countries can be merged with `combine` or ``+``.

    Parameters
    ----------
    country : str, optional
        ISO 3166-1 country code (``"CL"``, ``"US"``), or the code of a financial
        market of the holidays library (``"XECB"``: TARGET2, the EUR calendar;
        ``"XNYS"``). None for weekends only.
    subdiv : str, optional
        Subdivision of `country` (``"NY"`` for ``"US"``).
    custom_holidays : list of date, optional
        Extra holidays on top of the country's.

    Notes
    -----
    Use `get_calendar` or `resolve_calendar` to get the shared instance of a
    country: building one is costly, and a holiday added to a shared calendar is
    seen by every consumer, combined calendars made from it included.

    Each year is read once into a table of business days, on first use: `is_holiday`
    is then an array lookup and `add_business_days` a binary search.

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
    # The calendars a combined calendar is closed by (never combined ones themselves);
    # empty for a calendar of its own.
    _parts: tuple = field(default=(), repr=False)

    _weekend_weekdays: set[int] = field(init=False, default_factory=set)
    # year -> (ordinal of 1 January, 1 per business day of the year, business-day ordinals)
    _years: dict = field(init=False, default_factory=dict, repr=False, compare=False)
    _years_epoch: int = field(init=False, default=-1, repr=False, compare=False)

    def __post_init__(self):
        self._weekend_weekdays = {5, 6}
        if not self._countries and self.country is not None:
            self._countries = [
                self.country if not self.subdiv else f"{self.country}-{self.subdiv}"
            ]
        if self.custom_holidays is None:
            self.custom_holidays = []
        if self._calendar is None:
            self._build_holidays()

    def _build_holidays(self) -> None:
        """The holidays of `country` (none without one) plus the custom holidays."""
        if self.country is None:
            self._calendar = HolidayBase()
        elif self.country in _FINANCIAL_MARKETS:
            self._calendar = financial_holidays(self.country, subdiv=self.subdiv)
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
        global _epoch
        self._calendar.append(custom_holiday)
        if custom_holiday not in self.custom_holidays:
            self.custom_holidays.append(custom_holiday)
        _epoch += 1

    def add_custom_holidays(self, custom_holidays: list[date]):
        """Make each date of `custom_holidays` a holiday of this calendar."""
        for ch in custom_holidays:
            self.add_custom_holiday(ch)

    def remove_custom_holiday(self, custom_holiday: date):
        """Undo `add_custom_holiday`: `custom_holiday` is a holiday again only if it is
        one of the country's (or of a calendar this one combines)."""
        global _epoch
        if custom_holiday in self.custom_holidays:
            self.custom_holidays.remove(custom_holiday)
            self._build_holidays()
            _epoch += 1

    def is_holiday(self, date: date) -> bool:
        """True if `date` is a weekend day or a holiday (i.e. not a business day)."""
        table = self._year_table(date.year)
        return not table[1][date.toordinal() - table[0]]

    def add_business_days(self, t: date, business_days: int) -> date:
        """The date `business_days` business days after `t` (before if negative).

        With 0, `t` itself if it is a business day, else the next one.
        """
        year, ordinal = t.year, t.toordinal()
        days = self._year_table(year)[2]
        if business_days >= 0:
            i = (
                bisect_right(days, ordinal)
                if business_days
                else bisect_left(days, ordinal)
            )
            left = max(business_days, 1)
            while left > len(days) - i:
                left -= len(days) - i
                year += 1
                days, i = self._year_table(year)[2], 0
            return date.fromordinal(days[i + left - 1])
        i, left = bisect_left(days, ordinal), -business_days
        while left > i:
            left -= i
            year -= 1
            days = self._year_table(year)[2]
            i = len(days)
        return date.fromordinal(days[i - left])

    def _year_table(self, year: int) -> tuple:
        if self._years_epoch != _epoch:
            self._years.clear()
            self._years_epoch = _epoch
        table = self._years.get(year)
        if table is None:
            table = self._years[year] = self._build_year(year)
        return table

    def _build_year(self, year: int) -> tuple:
        first = date(year, 1, 1).toordinal()
        n = date(year + 1, 1, 1).toordinal() - first
        if self._parts:
            # Open only when every calendar it combines is.
            open_days = int.from_bytes(b"\x01" * n, "big")
            for part in self._parts:
                open_days &= int.from_bytes(part._year_table(year)[1], "big")
            flags = bytearray(open_days.to_bytes(n, "big"))
            for custom in self.custom_holidays:
                if custom.year == year:
                    flags[custom.toordinal() - first] = 0
        else:
            holidays, weekend = self._calendar, self._weekend_weekdays
            start = date(year, 1, 1).weekday()
            flags = bytearray(n)
            for k in range(n):
                if (start + k) % 7 not in weekend and date.fromordinal(
                    first + k
                ) not in holidays:
                    flags[k] = 1
        return first, flags, [first + k for k in range(n) if flags[k]]

    def combine(self, other: Self) -> Self:
        """New calendar that is closed whenever this one or `other` is."""
        parts: list[Calendar] = []
        for part in (*(self._parts or (self,)), *(other._parts or (other,))):
            if not any(part is p for p in parts):
                parts.append(part)
        return Calendar(
            custom_holidays=[
                *(self.custom_holidays or ()),
                *(other.custom_holidays or ()),
            ],
            _countries=list(dict.fromkeys([*self._countries, *other._countries])),
            _parts=tuple(parts),
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


_JOINT_CALENDARS: dict[tuple, Calendar] = {}


def get_joint_calendar(*countries: str | None) -> Calendar:
    """The shared calendar closed on the holidays of every one of `countries`.

    One code gives `get_calendar` of it. A joint calendar reads its parts, the
    shared country calendars, so a holiday added to one of them reaches it too.
    """
    codes = tuple(dict.fromkeys(countries))
    if len(codes) <= 1:
        return get_calendar(codes[0] if codes else None)
    calendar = _JOINT_CALENDARS.get(codes)
    if calendar is None:
        calendar = get_calendar(codes[0])
        for code in codes[1:]:
            calendar = calendar + get_calendar(code)
        _JOINT_CALENDARS[codes] = calendar
    return calendar


def resolve_calendar(calendar: "Calendar | str | None") -> Calendar:
    """`calendar` itself, or the shared calendar of the country code it names."""
    return calendar if isinstance(calendar, Calendar) else get_calendar(calendar)
