import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, KW_ONLY
from datetime import date

from ..rates import Rate
from ..dates import Calendar
from .currencies import Currency


@dataclass
class Index:
    name: str
    _: KW_ONLY
    currency: Currency = field(default=None)


@dataclass
class InterestIndex(Index, ABC):
    @abstractmethod
    def get_accrued_interest(self, notional: float, start_date: date, end_date: date) -> float:
        pass


def _make_calendar(currency: Currency, provided: Calendar) -> Calendar:
    locality_calendar = (
        Calendar(country=currency.locality.value) if currency is not None else Calendar()
    )
    if provided is None:
        return locality_calendar
    return provided.combine(locality_calendar)


@dataclass
class OvernightRateIndex(InterestIndex):
    """Overnight index defined by daily rate fixings (e.g. SOFR, ESTR).
    Maintains a private index starting at 100 on the earliest rate date.
    Accrual is index_end/index_start - 1. Gaps are filled by repeating the last known rate."""
    overnight_rates: dict[date, Rate] = field(default_factory=dict)
    calendar: Calendar = field(default=None)

    _index_values: dict[date, float] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        self.calendar = _make_calendar(self.currency, self.calendar)
        if self.overnight_rates:
            self._check_for_gaps()
            self._rebuild_index()

    def _check_for_gaps(self, verbose: bool = False):
        """Fill any missing business-day entries in overnight_rates by repeating
        the last known rate. Walk forward from min to max rate date one BD at a time."""
        if not self.overnight_rates:
            return
        end_t = max(self.overnight_rates)
        t = min(self.overnight_rates)
        while t < end_t:
            next_t = self.calendar.add_business_days(t, 1)
            if next_t not in self.overnight_rates:
                self.overnight_rates[next_t] = self.overnight_rates[t]
                if verbose:
                    warnings.warn(
                        f"{self.name}: no rate for {next_t}, filled with rate from {t}."
                    )
            t = next_t

    def _rebuild_index(self):
        if not self.overnight_rates:
            self._index_values = {}
            return
        start_t = min(self.overnight_rates)
        # A rate on day t covers t → next_business_day(t), so extend one BD past the last rate.
        end_t = self.calendar.add_business_days(max(self.overnight_rates), 1)
        self._index_values = {start_t: 100.0}
        r = None
        t = start_t
        while t < end_t:
            next_t = self.calendar.add_business_days(t, 1)
            if t in self.overnight_rates:
                r = self.overnight_rates[t]
            self._index_values[next_t] = self._index_values[t] * r.get_wealth_factor(t, next_t)
            t = next_t

    def add_rate(self, t: date, rate: Rate):
        self.overnight_rates[t] = rate
        self._check_for_gaps()
        self._rebuild_index()

    def get_accrued_interest(self, notional: float, start_date: date, end_date: date) -> float:
        return notional * (self._index_values[end_date] / self._index_values[start_date] - 1.0)


@dataclass
class OvernightPriceIndex(InterestIndex):
    """Overnight index defined by published index levels (e.g. ICP).
    Accrual is index_end/index_start - 1. Missing business-day entries are filled by
    extrapolating the implied growth rate from the preceding consecutive pair."""
    index_values: dict[date, float] = field(default_factory=dict)
    calendar: Calendar = field(default=None)

    def __post_init__(self):
        self.calendar = _make_calendar(self.currency, self.calendar)
        if not self.index_values:
            raise ValueError("index_values must be non-empty.")
        self._check_for_gaps()

    def _check_for_gaps(self, verbose: bool = False):
        """Fill missing business-day entries in index_values by geometric interpolation
        between each pair of consecutive known values.
        price[t] = price[t_start] * (price[t_end]/price[t_start])^(days(t_start→t)/days(t_start→t_end))
        This is equivalent to applying the implied daily rate uniformly across the gap."""
        sorted_known = sorted(self.index_values)
        for i in range(len(sorted_known) - 1):
            t_start = sorted_known[i]
            t_end = sorted_known[i + 1]
            days_total = (t_end - t_start).days
            ratio = self.index_values[t_end] / self.index_values[t_start]
            t = self.calendar.add_business_days(t_start, 1)
            while t < t_end:
                if t not in self.index_values:
                    days_elapsed = (t - t_start).days
                    self.index_values[t] = self.index_values[t_start] * ratio ** (days_elapsed / days_total)
                    if verbose:
                        warnings.warn(f"{self.name}: no value for {t}, filled by interpolation.")
                t = self.calendar.add_business_days(t, 1)

    def add_index_value(self, t: date, value: float):
        self.index_values[t] = value
        self._check_for_gaps()

    def get_accrued_interest(self, notional: float, start_date: date, end_date: date) -> float:
        return notional * (self.index_values[end_date] / self.index_values[start_date] - 1.0)


@dataclass
class TermRateIndex(InterestIndex):
    """Term rate index (e.g. LIBOR 3M, SOFR Term). Maps fixing dates to Rates."""
    historic_data: dict[date, Rate] = field(default_factory=dict)

    def add_rate(self, t: date, rate: Rate):
        self.historic_data[t] = rate

    def get_rate(self, t: date) -> Rate:
        if t in self.historic_data:
            return self.historic_data[t]
        raise KeyError(f"Date {t} not found in {self.name} data.")

    def get_accrued_interest(
        self,
        notional: float,
        start_date: date,
        end_date: date,
        fixing_date: date = None,
    ) -> float:
        if fixing_date is None:
            fixing_date = start_date
        return self.get_rate(fixing_date).get_accrued_interest(notional, start_date, end_date)
