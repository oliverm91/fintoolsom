from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Self

import numpy as np

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
    base_currency: Currency
    quote_currency: Currency

    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"{self.base_currency.value}/{self.quote_currency.value}"
    
    def invert(self) -> Self:
        return CurrencyPair(self.quote_currency, self.base_currency)

    def copy(self) -> Self:
        return CurrencyPair(self.base_currency, self.quote_currency)
    
    def __copy__(self) -> Self:
        return self.copy()
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: Self) -> bool:
        return self.name==other.name


@dataclass(slots=True)
class FX_Rate:
    currency_pair: CurrencyPair
    value: float

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError(f'FX_Rate value must be greater than 0. Got {self.value}.')
    
    def invert(self) -> Self:
        return FX_Rate(self.currency_pair.copy().invert(), 1 / self.value)
    
    def get_accrued_value_date(self, t: date, base_currency_curve: ZeroCouponCurve, quote_currency_curve: ZeroCouponCurve) -> float:
        base_df = base_currency_curve.get_df(t)
        quote_df = quote_currency_curve.get_df(t)
        return self.value * base_df / quote_df
    
    def get_accrued_values_dates(self, ts: list[date], base_currency_curve: ZeroCouponCurve, quote_currency_curve: ZeroCouponCurve) -> np.ndarray:
        base_df = base_currency_curve.get_dfs(ts)
        quote_df = quote_currency_curve.get_dfs(ts)
        return self.value * base_df / quote_df
    
    def copy(self) -> Self:
        return FX_Rate(self.currency_pair.copy(), self.value)
    
    def __copy__(self) -> Self:
        return self.copy()
    
    def __str__(self) -> str:
        return f'{self.currency_pair}@{self.value}'
    
    def __hash__(self) -> int:
        return hash(self.currency_pair.name)


@dataclass(slots=True)
class FX_RateData:
    currency_pair: CurrencyPair
    historic_data: dict[date, FX_Rate]

    def __post_init__(self):
        for fxr in self.historic_data.values():
            if fxr.currency_pair!=self.currency_pair:
                raise ValueError(f'All values in historic_data of FX_RateData object must match currency_pair. currency_pair: {self.currency_pair}, found {fxr.currency_pair} in historic_data')
            
    def copy(self) -> Self:
        return FX_RateData(self.currency_pair.copy(), self.historic_data.copy())
    
    def __copy__(self) -> Self:
        return self.copy()

    def invert(self) -> Self:
        return FX_RateData(self.currency_pair.copy().invert(), {t: fxr.copy().invert() for t, fxr in self.historic_data.items()})
    
    def get_fx_rate(self, t: date) -> FX_Rate:
        if t in self.historic_data:
            return self.historic_data[t]
        else:
            raise KeyError(f'Date {t} not found in {self.currency_pair} data.')
    
    def add_date(self, t: date, fx_rate: FX_Rate):
        self.historic_data[t] = fx_rate