from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...market.currencies import CurrencyPair


@dataclass
class Forward:
    notional: float
    strike: float
    payment_date: date
    is_buy: bool
    # None only allowed for UF-indexed NDFs (see NDF.is_uf_indexed), since UF is not a Currency.
    currency_pair: CurrencyPair = None

    sign: int = field(init=False)

    def __post_init__(self):
        self.sign = 1 if self.is_buy else -1
        if self.currency_pair is None:
            raise ValueError("currency_pair is required for a Forward.")


@dataclass
class NDF(Forward):
    fixing_date: date = None
    is_uf_indexed: bool = False

    def __post_init__(self):
        self.sign = 1 if self.is_buy else -1
        if self.fixing_date is None:
            raise ValueError("fixing_date is required for an NDF.")
        if not self.is_uf_indexed and self.currency_pair is None:
            raise ValueError("currency_pair is required for a non UF-indexed NDF.")
