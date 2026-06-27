import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta

from ..rates import Rate
from ..dates import Calendar, PrecedingConvention
from .currencies import Currency
from .index import (
    Index,
    RateIndex,
    PriceIndex,
    InterestPriceIndex,
    UFIndex,
)


def _shift_month(year: int, month: int, k: int) -> tuple[int, int]:
    """Add ``k`` calendar months to ``(year, month)``, returning ``(year, month)``."""
    index = year * 12 + (month - 1) + k
    return index // 12, index % 12 + 1


def _resolve_overnight_calendar(provided: Calendar, currency: Currency) -> Calendar:
    """Combine a user-provided calendar with the currency's locality calendar.
    With no currency, fall back to a bare (weekends-only) calendar."""
    locality_calendar = (
        Calendar(country=currency.locality.value) if currency is not None else Calendar()
    )
    if provided is None:
        return locality_calendar
    return provided.combine(locality_calendar)


@dataclass
class IndexHistory(ABC):
    """Time series of fixings for an :class:`Index`, plus the calculations that
    consume them. The definition (name, currency) is held on ``index`` so it is
    the single source of truth — histories never duplicate it."""
    index: Index

    @property
    def name(self) -> str:
        return self.index.name

    @property
    def currency(self) -> Currency:
        return self.index.currency


# ── Capability mixins (ABCs) ───────────────────────────────────────────────

class InterestHistory(IndexHistory, ABC):
    """An index history that accrues interest. The abstract method forces every
    concrete interest-bearing history to provide an accrual; price-only histories
    (see :class:`PriceHistory`) deliberately do not inherit this, so they never
    expose accrual."""

    @abstractmethod
    def get_accrued_interest(self, notional: float, start_date: date, end_date: date) -> float:
        ...


@dataclass
class OvernightHistory(IndexHistory, ABC):
    """Base for overnight histories. Owns the calendar field and resolves it from
    any provided calendar combined with the currency's locality calendar."""
    calendar: Calendar = field(default=None)

    def __post_init__(self):
        self.calendar = _resolve_overnight_calendar(self.calendar, self.index.currency)


# ── Price-only history (no accrual) ────────────────────────────────────────

@dataclass
class PriceHistory(IndexHistory):
    """History whose fixings are published price levels (floats). Provides level
    lookup only — used for price-only indexes such as the UF, which are read for
    their price in some calculations and never accrue interest."""
    index: PriceIndex
    values: dict[date, float] = field(default_factory=dict)

    def add_value(self, t: date, value: float):
        self.values[t] = value

    def get_value(self, t: date) -> float:
        if t in self.values:
            return self.values[t]
        raise KeyError(f"Date {t} not found in {self.name} price history.")


@dataclass
class UFIndexHistory(PriceHistory):
    """History of the Chilean UF (Unidad de Fomento) — daily price levels (CLP per
    UF), lookup only, no interest accrual — that additionally derives monthly
    inflation and projects the UF forward from a published CPI.

    The UF reajusts daily applying the prior calendar month's CPI: the CPI of
    month *m* is spread geometrically over the UF from day 10 of month *m+1* to
    day 9 of month *m+2*. Consequently the levels are published *ahead of time*,
    so ``values`` legitimately holds future dates — UF levels are known up to the
    9th of next month once that month's CPI has been released (around the 8th),
    and only up to the 9th of the current month before that release. See
    :meth:`get_last_known_date`.

    The publication calendar is built from the index currency's locality (CLP → CL),
    i.e. ``Calendar(country=index.currency.locality.value)``."""
    index: UFIndex

    _calendar: Calendar = field(init=False, repr=False, default=None)

    def __post_init__(self):
        currency = self.index.currency
        self._calendar = (
            Calendar(country=currency.locality.value)
            if currency is not None
            else Calendar()
        )

    def get_inflation(self, year: int, month: int) -> float:
        """Realised monthly inflation of calendar month ``(year, month)``.

        It equals the UF reajuste accumulated from day 9 of the following month to
        day 9 of the month after (the window over which that month's CPI is
        applied): ``level(9 of m+2) / level(9 of m+1) - 1``. Both boundary levels
        must be loaded — the later one is a future date until the month's CPI is
        published."""
        start = self._boundary_level(*_shift_month(year, month, 1))
        end = self._boundary_level(*_shift_month(year, month, 2))
        return end / start - 1.0

    def _boundary_level(self, year: int, month: int) -> float:
        d = date(year, month, 9)
        try:
            return self.get_value(d)
        except KeyError:
            raise ValueError(
                f"{self.name}: level for reajuste boundary {d} is not loaded; the "
                f"inflation spanning it has not been published / is not known yet."
            )

    def get_last_known_date(self, today: date, calendar: Calendar = None) -> date:
        """Last date for which a level is known as of ``today``.

        CPI is released around the 8th (taken as the preceding business day of the
        8th). Before that release only levels up to the 9th of the current month
        are known; on/after it, up to the 9th of next month. ``calendar`` defaults
        to the index locality calendar (CLP → CL)."""
        if calendar is None:
            calendar = self._calendar
        publication_date = PrecedingConvention(calendar).adjust(
            date(today.year, today.month, 8)
        )
        if today < publication_date:
            year, month = today.year, today.month
        else:
            year, month = _shift_month(today.year, today.month, 1)
        return date(year, month, 9)

    def extend_with_cpi(self, cpi: float, base_date: date = None) -> None:
        """Project the UF levels forward across one reajuste period from a monthly
        CPI variation ``cpi`` (decimal, e.g. ``0.004`` for 0.4%).

        The base is the 9th-day level the period grows from — by default the latest
        9th already loaded (the base UF is always the last known UF on a 9th). Each
        day ``d`` in the period (base 9th exclusive, next 9th inclusive) is set to::

            UF_d = UF_base * (1 + cpi) ** (days(base -> d) / days(base -> next 9th))

        where the denominator is the number of days from the base 9th to the 9th of
        the following month — equivalently the number of days in the base month.
        For example ``UF(13-Mar) = UF(9-Mar) * (1 + cpi) ** (4 / 31)``.

        ``cpi`` is the CPI that governs this period, i.e. the variation of the month
        before the base month (it is fully reflected once the next 9th is reached,
        so ``get_inflation`` of that month then returns ``cpi``). Existing levels are
        left untouched; only missing days are filled."""
        if base_date is None:
            base_date = self._last_known_ninth()
        elif base_date.day != 9:
            raise ValueError(f"base_date must fall on a 9th; got {base_date}.")
        if base_date not in self.values:
            raise KeyError(f"{self.name}: base level for {base_date} is not loaded.")

        base_uf = self.values[base_date]
        next_ninth = date(*_shift_month(base_date.year, base_date.month, 1), 9)
        period_days = (next_ninth - base_date).days

        d = base_date + timedelta(days=1)
        while d <= next_ninth:
            if d not in self.values:
                elapsed = (d - base_date).days
                # The UF is published rounded to 2 decimals (centavos).
                self.values[d] = round(base_uf * (1.0 + cpi) ** (elapsed / period_days), 2)
            d += timedelta(days=1)

    def _last_known_ninth(self) -> date:
        ninths = [d for d in self.values if d.day == 9]
        if not ninths:
            raise ValueError(
                f"{self.name}: no 9th-day level loaded to use as the reajuste base."
            )
        return max(ninths)


# ── Rate-valued interest histories ─────────────────────────────────────────

@dataclass
class RateHistory(InterestHistory, ABC):
    """Interest history whose fixings are :class:`Rate` values. Provides rate
    lookup; the accrual formula differs per index kind (overnight vs term) and is
    left abstract for subclasses to implement."""
    index: RateIndex
    rates: dict[date, Rate] = field(default_factory=dict)

    def add_rate(self, t: date, rate: Rate):
        self.rates[t] = rate

    def get_rate(self, t: date) -> Rate:
        if t in self.rates:
            return self.rates[t]
        raise KeyError(f"Date {t} not found in {self.name} rate history.")


@dataclass
class OvernightRateHistory(OvernightHistory, RateHistory):
    """Overnight index defined by daily rate fixings (e.g. SOFR, ESTR).
    Maintains a private index starting at 100 on the earliest rate date;
    accrual is index_end/index_start - 1. Gaps are filled by repeating the last
    known rate."""

    _index_values: dict[date, float] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        super().__post_init__()  # resolves calendar via OvernightHistory
        if self.rates:
            self._check_for_gaps()
            self._rebuild_index()

    def _check_for_gaps(self, verbose: bool = False):
        """Fill any missing business-day entries in ``rates`` by repeating the
        last known rate. Walk forward from min to max rate date one BD at a time."""
        if not self.rates:
            return
        end_t = max(self.rates)
        t = min(self.rates)
        while t < end_t:
            next_t = self.calendar.add_business_days(t, 1)
            if next_t not in self.rates:
                self.rates[next_t] = self.rates[t]
                if verbose:
                    warnings.warn(
                        f"{self.name}: no rate for {next_t}, filled with rate from {t}."
                    )
            t = next_t

    def _rebuild_index(self):
        if not self.rates:
            self._index_values = {}
            return
        start_t = min(self.rates)
        # A rate on day t covers t → next_business_day(t), so extend one BD past the last rate.
        end_t = self.calendar.add_business_days(max(self.rates), 1)
        self._index_values = {start_t: 100.0}
        r = None
        t = start_t
        while t < end_t:
            next_t = self.calendar.add_business_days(t, 1)
            if t in self.rates:
                r = self.rates[t]
            self._index_values[next_t] = self._index_values[t] * r.get_wealth_factor(t, next_t)
            t = next_t

    def add_rate(self, t: date, rate: Rate):
        self.rates[t] = rate
        self._check_for_gaps()
        self._rebuild_index()

    def get_accrued_interest(self, notional: float, start_date: date, end_date: date) -> float:
        return notional * (self._index_values[end_date] / self._index_values[start_date] - 1.0)


@dataclass
class TermRateHistory(RateHistory):
    """Term rate index (e.g. LIBOR 3M, SOFR Term). Accrual uses the single rate
    fixed at ``fixing_date`` (defaults to ``start_date``)."""

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


# ── Price-valued interest histories ────────────────────────────────────────

@dataclass
class InterestPriceHistory(PriceHistory, InterestHistory, ABC):
    """Interest history published as price levels (e.g. ICP). Reuses the level
    lookup of :class:`PriceHistory` and additionally accrues via the level ratio.
    The concrete accrual (and any gap handling) is provided by subclasses."""
    index: InterestPriceIndex


@dataclass
class OvernightInterestPriceHistory(OvernightHistory, InterestPriceHistory):
    """Overnight index defined by published index levels (e.g. ICP).
    Accrual is level_end/level_start - 1. Missing business-day entries are filled
    by geometric interpolation between surrounding known values."""

    def __post_init__(self):
        super().__post_init__()  # resolves calendar via OvernightHistory
        if not self.values:
            raise ValueError("values must be non-empty.")
        self._check_for_gaps()

    def _check_for_gaps(self, verbose: bool = False):
        """Fill missing business-day entries by geometric interpolation between
        each pair of consecutive known values:
        level[t] = level[t0] * (level[t1]/level[t0])^(days(t0→t)/days(t0→t1)).
        This applies the implied daily rate uniformly across the gap."""
        sorted_known = sorted(self.values)
        for i in range(len(sorted_known) - 1):
            t_start = sorted_known[i]
            t_end = sorted_known[i + 1]
            days_total = (t_end - t_start).days
            ratio = self.values[t_end] / self.values[t_start]
            t = self.calendar.add_business_days(t_start, 1)
            while t < t_end:
                if t not in self.values:
                    days_elapsed = (t - t_start).days
                    self.values[t] = self.values[t_start] * ratio ** (days_elapsed / days_total)
                    if verbose:
                        warnings.warn(f"{self.name}: no value for {t}, filled by interpolation.")
                t = self.calendar.add_business_days(t, 1)

    def add_value(self, t: date, value: float):
        self.values[t] = value
        self._check_for_gaps()

    def get_accrued_interest(self, notional: float, start_date: date, end_date: date) -> float:
        return notional * (self.values[end_date] / self.values[start_date] - 1.0)
