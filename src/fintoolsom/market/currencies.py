from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Self

from ..rates import ZeroCouponCurve


class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CLP = "CLP"
    BRL = "BRL"
    RUB = "RUB"
    CNY = "CNY"
    INR = "INR"
    AUD = "AUD"
    NZD = "NZD"
    HKD = "HKD"
    SGD = "SGD"
    KRW = "KRW"
    THB = "THB"
    COP = "COP"
    MXN = "MXN"
    BND = "BND"
    SEK = "SEK"
    NOK = "NOK"


@dataclass(slots=True)
class CurrencyPair:
    cp_date: date
    base_currency: Currency
    quote_currency: Currency
    value: float

    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"{self.base_currency.value}/{self.quote_currency.value}"

    def accrue(self, base_currency_curve: ZeroCouponCurve, quote_currency_curve: ZeroCouponCurve, t: date) -> Self:
        new_value = self.value * base_currency_curve.get_dfs(t) / quote_currency_curve.get_dfs(t)
        return CurrencyPair(t, self.base_currency, self.quote_currency, new_value, name=self.name)
    
    def invert(self) -> Self:
        return CurrencyPair(self.cp_date, self.quote_currency, self.base_currency, 1/self.value, name=self.name)
