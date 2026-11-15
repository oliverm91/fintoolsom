from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from fintoolsom import settings

if TYPE_CHECKING:
    from ...market.currencies import CurrencyPair
    from ...market.index import Index


@dataclass
class Option(ABC):
    notional: float
    strike: float
    maturity: date
    currency_pair: CurrencyPair
    # Indices identifying the market.curves entries for each side of currency_pair.
    domestic_index: Index = None
    foreign_index: Index = None

    _sign: int = field(init=False)
    # Backend of "derivatives" when created: valued in C++ when True (settings).
    _native: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._native = settings.get_backend("derivatives") == "cpp"


@dataclass
class Call(Option):
    def __post_init__(self):
        super().__post_init__()
        self._sign = 1


@dataclass
class Put(Option):
    def __post_init__(self):
        super().__post_init__()
        self._sign = -1
