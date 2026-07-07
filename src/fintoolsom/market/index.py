from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, KW_ONLY
from datetime import date
from typing import TYPE_CHECKING

from .currencies import Currency
from ..dates import Calendar

if TYPE_CHECKING:
    from ..dates.term import Term


@dataclass(eq=False)
class Index(ABC):
    """Definition (identity) of a market index: its name and currency.

    A definition holds no time series — the historical fixings and the
    calculations that consume them live in a separate ``IndexHistory``
    (see ``index_history.py``). Definitions are the lightweight objects that
    travel inside quotes and leg specs; histories are the market data the
    calculator reads."""
    name: str
    _: KW_ONLY
    currency: Currency = field(default=None)
    # Fixing / business-day calendar, resolved from the currency's locality in __post_init__.
    calendar: Calendar = field(init=False, default=None)

    def __post_init__(self):
        self.calendar = (
            Calendar(country=self.currency.locality.value)
            if self.currency is not None
            else Calendar()
        )

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Index) and self.name == other.name

    @abstractmethod
    def get_maturity(self, start_date: date) -> date:
        """End date of one accrual period of this index starting on ``start_date``
        (overnight → the next business day; term rate → advanced by the tenor)."""
        ...


class InterestIndex(ABC):
    """Marker for index definitions that bear interest (their history can accrue).

    Used to type interest-bearing references such as a swap's floating-leg index
    or collateral index, so that price-only indexes (e.g. the UF) are excluded by
    the type rather than by a runtime check."""


@dataclass(eq=False)
class RateIndex(Index, InterestIndex):
    """Index whose fixings are :class:`Rate` values (e.g. SOFR, ESTR, LIBOR 3M). Always
    interest-bearing. Abstract by convention — use a concrete kind:
    :class:`OvernightRateIndex` (daily compounding, no tenor) or :class:`TermRateIndex`
    (a single fixing over an explicit tenor).

    ``spot_lag`` is the business-day offset from the fixing date to the accrual (value)
    start: 0 for an overnight index (the fixing applies from ``t``), non-zero for a term
    rate (typically 2 for Term SOFR / LIBOR)."""
    spot_lag: int = field(default=0, kw_only=True)


class OvernightIndex(ABC):
    """Marker for indexes whose fixing accrues by DAILY compounding over the accrual
    period (OIS-style: SOFR, ESTR, ICP), rather than a single term-rate fixing.

    It is orthogonal to rate-vs-price — it combines with :class:`RateIndex` or
    :class:`InterestPriceIndex`, mirroring ``OvernightHistory`` on the history side — and
    it is what lets a swap builder pick an ``OvernightLeg`` (daily compounding) over a
    ``TermRateLeg`` for legs on this index. Concrete overnight indexes advance one
    business day (see e.g. :meth:`OvernightRateIndex.get_maturity`)."""


@dataclass(eq=False)
class OvernightRateIndex(OvernightIndex, RateIndex):
    """Overnight rate index (e.g. SOFR, ESTR, ICP-as-rate). It has no tenor — it accrues
    by daily compounding over a single business day of its fixing calendar."""

    def get_maturity(self, start_date: date) -> date:
        # Overnight accrual spans one business day on the index's fixing calendar.
        return self.calendar.add_business_days(start_date, 1)


@dataclass(eq=False)
class TermRateIndex(RateIndex):
    """Term rate index (e.g. LIBOR 3M, Term SOFR). ``term`` is the rate's tenor — the
    period a single fixing covers; ``spot_lag`` (default 2) is the fixing→value offset."""
    term: Term
    spot_lag: int = field(default=2, kw_only=True)

    def get_maturity(self, start_date: date) -> date:
        # A term rate covers a full tenor: advance start_date by the term.
        return self.term.advance(start_date)


@dataclass(eq=False)
class PriceIndex(Index):
    """Index published as price levels / floats and read only for its level
    (e.g. the UF). Price-only: it does not accrue interest, hence it is *not* an
    :class:`InterestIndex`. Abstract by convention — use a concrete kind
    (:class:`UFIndex`, :class:`OvernightInterestPriceIndex`)."""


@dataclass(eq=False)
class InterestPriceIndex(PriceIndex, InterestIndex):
    """Index published as price levels whose level ratio *does* accrue interest
    (e.g. the Chilean ICP). Both a price index and interest-bearing. Abstract by
    convention — use a concrete kind (:class:`OvernightInterestPriceIndex`)."""


@dataclass(eq=False)
class OvernightInterestPriceIndex(OvernightIndex, InterestPriceIndex):
    """Overnight interest-bearing price index (e.g. the Chilean ICP): published as
    daily levels whose ratio accrues by daily compounding over a single business
    day. Mirrors :class:`OvernightRateIndex` on the price side (paired with
    :class:`OvernightInterestPriceHistory`)."""

    def get_maturity(self, start_date: date) -> date:
        # Overnight accrual spans one business day on the index's fixing calendar.
        return self.calendar.add_business_days(start_date, 1)


@dataclass(eq=False)
class UFIndex(PriceIndex):
    """The Chilean UF (Unidad de Fomento): an inflation-linked unit published as
    daily price levels (CLP per UF) and read only for its level. Price-only — it
    does not accrue interest — but its daily levels encode realised CPI, from which
    monthly inflation can be derived. Always denominated in CLP."""

    def get_maturity(self, start_date: date) -> date:
        # The UF's natural period is the reajuste month: the 9th to the 9th of the
        # next month. Return the 9th that CLOSES the period containing start_date (a
        # date on the 9th starts its own period). Unadjusted calendar 9th, matching
        # the reajuste boundaries used by UFIndexHistory / UFConvention.
        if start_date.day < 9:
            return date(start_date.year, start_date.month, 9)
        year = start_date.year + start_date.month // 12
        month = start_date.month % 12 + 1
        return date(year, month, 9)
