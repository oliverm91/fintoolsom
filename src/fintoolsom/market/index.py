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
    """Definition (identity) of a market index.

    A definition holds no time series: fixings and the calculations using them
    live in an :class:`~fintoolsom.market.index_history.IndexHistory`. Definitions
    travel inside quotes and legs; histories are market data. Two indexes are
    equal when their names are.

    Parameters
    ----------
    name : str
        Index name, the identity of the index.
    currency : Currency, optional
        Currency of the index (keyword-only).

    Attributes
    ----------
    calendar : Calendar
        Fixing and business-day calendar of the currency's locality (a plain
        weekend calendar when there is no currency).
    """
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
        """End of one accrual period of the index starting on `start_date`.

        Parameters
        ----------
        start_date : date
            Period start.

        Returns
        -------
        date
            The next business day for an overnight index, `start_date` advanced
            by the tenor for a term rate.
        """
        ...


class InterestIndex(ABC):
    """Marker for interest-bearing index definitions (their history accrues).

    Types interest-bearing references such as a floating-leg or collateral index,
    so price-only indexes are excluded by type.
    """


@dataclass(eq=False)
class RateIndex(Index, InterestIndex):
    """Index whose fixings are :class:`Rate` values (e.g. SOFR, ESTR, LIBOR 3M).

    Abstract by convention: use :class:`OvernightRateIndex` (daily compounding)
    or :class:`TermRateIndex` (one fixing over a tenor).

    Parameters
    ----------
    spot_lag : int, default 0
        Business days from the fixing date to the accrual start (keyword-only):
        0 for an overnight index, typically 2 for a term rate.
    """
    spot_lag: int = field(default=0, kw_only=True)


class OvernightIndex(ABC):
    """Marker for indexes accrued by daily compounding (OIS-style: SOFR, ESTR, ICP).

    Orthogonal to rate vs price: it combines with :class:`RateIndex` or
    :class:`InterestPriceIndex`. It selects an ``OvernightLeg`` rather than a
    ``TermRateLeg`` for legs on the index. Concrete overnight indexes advance
    one business day per accrual period.
    """


@dataclass(eq=False)
class OvernightRateIndex(OvernightIndex, RateIndex):
    """Overnight rate index (e.g. SOFR, ESTR, Cámara).

    No tenor: each fixing covers one business day of the index calendar and
    fixings compound daily.
    """

    def get_maturity(self, start_date: date) -> date:
        # Overnight accrual spans one business day on the index's fixing calendar.
        return self.calendar.add_business_days(start_date, 1)


@dataclass(eq=False)
class TermRateIndex(RateIndex):
    """Term rate index (e.g. LIBOR 3M, Term SOFR, TAB).

    Parameters
    ----------
    term : Term
        Tenor covered by a single fixing.
    spot_lag : int, default 2
        Business days from the fixing date to the accrual start (keyword-only).
    """
    term: Term
    spot_lag: int = field(default=2, kw_only=True)

    def get_maturity(self, start_date: date) -> date:
        # A term rate covers a full tenor: advance start_date by the term.
        return self.term.advance(start_date)


@dataclass(eq=False)
class PriceIndex(Index):
    """Index published as price levels.

    Not an :class:`InterestIndex` by itself. Abstract by convention: use
    :class:`OvernightInterestPriceIndex`. (The UF is not an index: it is the
    currency ``Currency.CLF``, see :mod:`fintoolsom.market.uf`.)
    """


@dataclass(eq=False)
class InterestPriceIndex(PriceIndex, InterestIndex):
    """Price index whose level ratio accrues interest (e.g. the Chilean ICP).

    Both a price index and interest-bearing. Abstract by convention: use
    :class:`OvernightInterestPriceIndex`.
    """


@dataclass(eq=False)
class OvernightInterestPriceIndex(OvernightIndex, InterestPriceIndex):
    """Overnight interest-bearing price index (e.g. the Chilean ICP).

    Published as daily levels; the ratio of two levels is the daily-compounded
    accrual between them. Paired with ``OvernightInterestPriceHistory``.
    """

    def get_maturity(self, start_date: date) -> date:
        # Overnight accrual spans one business day on the index's fixing calendar.
        return self.calendar.add_business_days(start_date, 1)
