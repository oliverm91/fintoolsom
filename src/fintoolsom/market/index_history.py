import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

from ..rates import Rate, RateConvention, LinearInterestConvention
from ..dates import Calendar, ActualDayCountConvention
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
    def get_accrued_interest(self, notional: float, start_date: date, end_date: date) -> float:
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
    minus 1. Missing business days repeat the last known fixing.

    Parameters
    ----------
    index : OvernightRateIndex
        Definition of the index.
    rates : dict of date to Rate, optional
        Daily fixings.
    """
    index: OvernightRateIndex

    _index_values: dict[date, float] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        if self.rates:
            self._check_for_gaps()
            self._rebuild_index()

    def _check_for_gaps(self, verbose: bool = False):
        """Fill missing business days in `rates` with the last known fixing."""
        if not self.rates:
            return
        end_t = max(self.rates)
        t = min(self.rates)
        while t < end_t:
            next_t = self.index.get_maturity(t)  # next business day (overnight accrual step)
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
        end_t = self.index.get_maturity(max(self.rates))
        self._index_values = {start_t: 100.0}
        r = None
        t = start_t
        while t < end_t:
            next_t = self.index.get_maturity(t)
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
        return self.get_rate(fixing_date).get_accrued_interest(notional, start_date, end_date)


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
            t = self.index.get_maturity(t_start)  # next business day (overnight accrual step)
            while t < t_end:
                if t not in self.values:
                    days_elapsed = (t - t_start).days
                    self.values[t] = self.values[t_start] * ratio ** (days_elapsed / days_total)
                    if verbose:
                        warnings.warn(f"{self.name}: no value for {t}, filled by interpolation.")
                t = self.index.get_maturity(t)

    def add_value(self, t: date, value: float):
        self.values[t] = value
        self._check_for_gaps()

    def get_accrued_interest(self, notional: float, start_date: date, end_date: date) -> float:
        return notional * (self.values[end_date] / self.values[start_date] - 1.0)

    def spot_overnight_rate(self, t: date) -> Rate:
        # No forward level is published, so estimate the overnight rate from the most
        # recent realised one-business-day level ratio (act/360 linear, as elsewhere).
        known = sorted(d for d in self.values if d <= t)
        if len(known) < 2:
            raise KeyError(
                f"{self.name}: need at least two levels on/before {t} to imply an overnight rate."
            )
        prev, last = known[-2], known[-1]
        wf = self.values[last] / self.values[prev]
        year_fraction = (last - prev).days / 360
        rate_value = float(LinearInterestConvention.get_rate_from_wf(wf, year_fraction))
        return Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 360), rate_value)
