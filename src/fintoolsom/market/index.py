from abc import ABC
from dataclasses import dataclass, field, KW_ONLY

from .currencies import Currency


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

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Index) and self.name == other.name


class InterestIndex(ABC):
    """Marker for index definitions that bear interest (their history can accrue).

    Used to type interest-bearing references such as a swap's floating-leg index
    or collateral index, so that price-only indexes (e.g. the UF) are excluded by
    the type rather than by a runtime check."""


@dataclass(eq=False)
class RateIndex(Index, InterestIndex):
    """Index whose fixings are :class:`Rate` values (e.g. SOFR, ESTR, LIBOR 3M).
    Always interest-bearing."""


@dataclass(eq=False)
class PriceIndex(Index):
    """Index published as price levels / floats and read only for its level
    (e.g. the UF). Price-only: it does not accrue interest, hence it is *not* an
    :class:`InterestIndex`."""


@dataclass(eq=False)
class InterestPriceIndex(PriceIndex, InterestIndex):
    """Index published as price levels whose level ratio *does* accrue interest
    (e.g. the Chilean ICP). Both a price index and interest-bearing."""


@dataclass(eq=False)
class UFIndex(PriceIndex):
    """The Chilean UF (Unidad de Fomento): an inflation-linked unit published as
    daily price levels (CLP per UF) and read only for its level. Price-only — it
    does not accrue interest — but its daily levels encode realised CPI, from which
    monthly inflation can be derived. Always denominated in CLP."""
