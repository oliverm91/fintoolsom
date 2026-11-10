import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

from ..rates import Rate
from ..dates import Calendar
from .currencies import Currency
from .index import (
    Index,
    RateIndex,
    OvernightRateIndex,
    PriceIndex,
    InterestPriceIndex,
    OvernightInterestPriceIndex,
)


@dataclass
class IndexHistory(ABC):
    """Time series of fixings of an :class:`Index`, and the calculations using them.

    The definition (name, currency, calendar) lives only on `index`.

    Parameters
    ----------
    index : Index
        Definition of the index.
    """

    index: Index

    @property
    def name(self) -> str:
        """Index name."""
        return self.index.name

    @property
    def currency(self) -> Currency:
        """Index currency."""
        return self.index.currency


# ── Capability mixins (ABCs) ───────────────────────────────────────────────


class InterestHistory(IndexHistory, ABC):
    """Index history that accrues interest.

    Every concrete interest-bearing history must implement the accrual.
    """

    @abstractmethod
    def get_accrued_interest(
        self, notional: float, start_date: date, end_date: date
    ) -> float:
        """Realised interest on `notional` over ``[start_date, end_date]``.

        Parameters
        ----------
        notional : float
            Amount accruing.
        start_date, end_date : date
            Accrual period.

        Returns
        -------
        float
        """
        ...


@dataclass
class OvernightHistory(IndexHistory, ABC):
    """Base for overnight histories, accruing one business day at a time.

    The next accrual date and the calendar belong to the index
    (:meth:`Index.get_maturity`, :attr:`Index.calendar`).
    """

    @property
    def calendar(self) -> Calendar:
        """Fixing calendar of the index."""
        return self.index.calendar

    @abstractmethod
    def spot_overnight_rate(self, t: date) -> Rate:
        """Overnight rate anchoring a curve's short end ``[t, t + 1bd]`` at `t`.

        A rate index returns today's fixing, which applies to ``[t, t + 1bd]``. A
        price index, which does not publish tomorrow's level, returns the latest
        realised one-day rate as an estimate.

        Parameters
        ----------
        t : date
            Valuation date.

        Returns
        -------
        Rate

        Raises
        ------
        KeyError
            If it cannot be determined (no fixing at `t`, too few levels), so
            callers can fall back.
        """
        ...


# ── Rate-valued interest histories ─────────────────────────────────────────


@dataclass
class RateHistory(InterestHistory, ABC):
    """Interest history whose fixings are :class:`Rate` values.

    The accrual differs per index kind (overnight vs term) and is left to
    subclasses.

    Parameters
    ----------
    index : RateIndex
        Definition of the index.
    rates : dict of date to Rate, optional
        Fixings.
    """

    index: RateIndex
    rates: dict[date, Rate] = field(default_factory=dict)

    def add_rate(self, t: date, rate: Rate):
        """Set the fixing at `t`."""
        self.rates[t] = rate

    def get_rate(self, t: date) -> Rate:
        """Fixing at `t`.

        Raises
        ------
        KeyError
            If there is no fixing at `t`.
        """
        if t in self.rates:
            return self.rates[t]
        raise KeyError(f"Date {t} not found in {self.name} rate history.")


@dataclass
class OvernightRateHistory(OvernightHistory, RateHistory):
    """Overnight index defined by daily rate fixings (e.g. SOFR, ESTR, Cámara).

    Keeps a private level series starting at 100 on the first fixing date and
    compounding each fixing over its business day; accrual is the level ratio
    minus 1. Missing business days repeat the last known fixing, so a fixing added
    inside such a gap applies until the next fixing that was actually set.

    Parameters
    ----------
    index : OvernightRateIndex
        Definition of the index.
    rates : dict of date to Rate, optional
        Daily fixings.
    """

    index: OvernightRateIndex

    _index_values: dict[date, float] = field(
        init=False, repr=False, default_factory=dict
    )
    _filled_dates: set[date] = field(init=False, repr=False, default_factory=set)

    def __post_init__(self):
        if self.rates:
            self._check_for_gaps()
            self._rebuild_index()

    def _check_for_gaps(self, verbose: bool = False):
        """Fill missing business days in `rates` with the last known fixing."""
        if self.rates:
            self._fill_gaps(min(self.rates), max(self.rates), verbose)

    def _fill_gaps(self, t: date, end_t: date, verbose: bool = False) -> date:
        """Fill the business days from `t` to `end_t`; returns the date the walk ends on."""
        while t < end_t:
            next_t = self.index.get_maturity(
                t
            )  # next business day (overnight accrual step)
            if next_t not in self.rates:
                self.rates[next_t] = self.rates[t]
                self._filled_dates.add(next_t)
                if verbose:
                    warnings.warn(
                        f"{self.name}: no rate for {next_t}, filled with rate from {t}."
                    )
            t = next_t
        return t

    def _rebuild_index(self):
        if not self.rates:
            self._index_values = {}
            return
        start_t = min(self.rates)
        self._index_values = {start_t: 100.0}
        # A rate on day t covers t → next_business_day(t), so extend one BD past the last rate.
        self._compound_levels(start_t, self.index.get_maturity(max(self.rates)))

    def _compound_levels(self, t: date, end_t: date):
        """Recompute the levels after `t` up to `end_t`, from the level at `t`."""
        levels = self._index_values
        r = None
        while t < end_t:
            next_t = self.index.get_maturity(t)
            if t in self.rates:
                r = self.rates[t]
            levels[next_t] = levels[t] * r.get_wealth_factor(t, next_t)
            t = next_t

    def _replace_fixing(self, t: date, rate: Rate):
        """Apply `rate` from `t` until the next set fixing; later levels scale by the same ratio."""
        levels = self._index_values
        next_fixing = self.index.get_maturity(t)
        while next_fixing in self._filled_dates:
            self.rates[next_fixing] = rate
            next_fixing = self.index.get_maturity(next_fixing)
        previous = levels[next_fixing]
        self._compound_levels(t, next_fixing)
        ratio = levels[next_fixing] / previous
        for d in reversed(levels):
            if d <= next_fixing:
                break
            levels[d] *= ratio

    def _prepend_levels(self, t: date, first: date):
        """Start the levels at `t`, scaling the existing ones to keep them consistent."""
        existing = self._index_values
        self._index_values = {t: 100.0}
        self._compound_levels(t, first)
        ratio = self._index_values[first] / existing[first]
        for d, level in existing.items():
            self._index_values[d] = level * ratio

    def add_rate(self, t: date, rate: Rate):
        """Set the fixing at `t`, recomputing only what it changes.

        A date past the last fixing adds the new levels. A date already in the history
        takes `rate` until the next set fixing, replacing any repeated fixings in
        between, and scales the later levels, as does a date before the first fixing.
        A date that is not a business day of the index rebuilds the whole history.
        """
        levels = self._index_values
        self.rates[t] = rate
        self._filled_dates.discard(t)
        newest = reversed(levels)
        end, last_fixing = next(newest, None), next(newest, None)
        if end is not None:
            first = next(iter(levels))
            if t == end or (t > end and self._fill_gaps(last_fixing, t) == t):
                self._compound_levels(end, self.index.get_maturity(t))
                return
            if t in levels:
                self._replace_fixing(t, rate)
                return
            if t < first and self._fill_gaps(t, first) == first:
                self._prepend_levels(t, first)
                return
        self._check_for_gaps()
        self._rebuild_index()

    def get_accrued_interest(
        self, notional: float, start_date: date, end_date: date
    ) -> float:
        return notional * (
            self._index_values[end_date] / self._index_values[start_date] - 1.0
        )

    def spot_overnight_rate(self, t: date) -> Rate:
        # Today's published overnight fixing applies forward to [t, t+1bd].
        return self.get_rate(t)


@dataclass
class TermRateHistory(RateHistory):
    """Term rate history (e.g. LIBOR 3M, Term SOFR, TAB).

    Accrual applies the single rate fixed at the fixing date over the period.
    """

    def get_accrued_interest(
        self,
        notional: float,
        start_date: date,
        end_date: date,
        fixing_date: date = None,
    ) -> float:
        if fixing_date is None:
            fixing_date = start_date
        return self.get_rate(fixing_date).get_accrued_interest(
            notional, start_date, end_date
        )


# ── Price-level history ────────────────────────────────────────────────────


@dataclass
class PriceHistory(IndexHistory):
    """History of published price levels, with level lookup.

    Base of :class:`InterestPriceHistory` (e.g. the ICP), which adds accrual
    through level ratios.

    Parameters
    ----------
    index : PriceIndex
        Definition of the index.
    values : dict of date to float, optional
        Published levels.
    """

    index: PriceIndex
    values: dict[date, float] = field(default_factory=dict)

    def add_value(self, t: date, value: float):
        """Set the level at `t`."""
        self.values[t] = value

    def get_value(self, t: date) -> float:
        """Level at `t`.

        Raises
        ------
        KeyError
            If no level is loaded at `t`.
        """
        if t in self.values:
            return self.values[t]
        raise KeyError(f"Date {t} not found in {self.name} price history.")


# ── Price-valued interest histories ────────────────────────────────────────


@dataclass
class InterestPriceHistory(PriceHistory, InterestHistory, ABC):
    """Interest history published as price levels (e.g. ICP).

    Reuses the level lookup of :class:`PriceHistory` and accrues through the level
    ratio; the concrete accrual and gap handling are left to subclasses.

    Parameters
    ----------
    index : InterestPriceIndex
        Definition of the index.
    """

    index: InterestPriceIndex


@dataclass
class OvernightInterestPriceHistory(OvernightHistory, InterestPriceHistory):
    """Overnight index defined by published levels (e.g. ICP).

    Accrual is ``level_end / level_start - 1``. Missing business days are filled
    by geometric interpolation between the surrounding known levels.

    Parameters
    ----------
    index : OvernightInterestPriceIndex
        Definition of the index.
    values : dict of date to float
        Published levels.

    Raises
    ------
    ValueError
        If `values` is empty.
    """

    index: OvernightInterestPriceIndex

    def __post_init__(self):
        if not self.values:
            raise ValueError("values must be non-empty.")
        self._check_for_gaps()

    def _check_for_gaps(self, verbose: bool = False):
        """Fill missing business days by geometric interpolation.

        ``level[t] = level[t0] * (level[t1] / level[t0]) ** (days(t0, t) / days(t0, t1))``
        between consecutive known levels, applying the implied daily rate
        uniformly across the gap.
        """
        sorted_known = sorted(self.values)
        for i in range(len(sorted_known) - 1):
            t_start = sorted_known[i]
            t_end = sorted_known[i + 1]
            days_total = (t_end - t_start).days
            ratio = self.values[t_end] / self.values[t_start]
            t = self.index.get_maturity(
                t_start
            )  # next business day (overnight accrual step)
            while t < t_end:
                if t not in self.values:
                    days_elapsed = (t - t_start).days
                    self.values[t] = self.values[t_start] * ratio ** (
                        days_elapsed / days_total
                    )
                    if verbose:
                        warnings.warn(
                            f"{self.name}: no value for {t}, filled by interpolation."
                        )
                t = self.index.get_maturity(t)

    def add_value(self, t: date, value: float):
        self.values[t] = value
        self._check_for_gaps()

    def get_accrued_interest(
        self, notional: float, start_date: date, end_date: date
    ) -> float:
        return notional * (self.values[end_date] / self.values[start_date] - 1.0)

    def spot_overnight_rate(self, t: date) -> Rate:
        # No forward level is published, so estimate the overnight rate from the most
        # recent realised one-business-day level ratio, in the index's rate convention.
        known = sorted(d for d in self.values if d <= t)
        if len(known) < 2:
            raise KeyError(
                f"{self.name}: need at least two levels on/before {t} to imply an overnight rate."
            )
        prev, last = known[-2], known[-1]
        wf = self.values[last] / self.values[prev]
        rc = self.index.rate_convention
        year_fraction = rc.day_count_convention.get_time_fraction(
            prev, last, rc.time_fraction_base
        )
        rate_value = float(rc.interest_convention.get_rate_from_wf(wf, year_fraction))
        return Rate(rc.copy(), rate_value)
