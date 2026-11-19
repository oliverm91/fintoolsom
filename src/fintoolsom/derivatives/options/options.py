from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...market.currencies import CurrencyPair


@dataclass
class Option(ABC):
    notional: float
    strike: float
    maturity: date
    currency_pair: CurrencyPair

    _sign: int = field(init=False)


@dataclass
class Call(Option):
    def __post_init__(self):
        self._sign = 1


@dataclass
class Put(Option):
    def __post_init__(self):
        self._sign = -1
