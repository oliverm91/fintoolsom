from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Self

import numpy as np

from ..rates import ZeroCouponCurve
from .localities import Locality


class CurrencyName(Enum):
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
    PEN = "PEN"


_CURRENCY_LOCALITY_MAP: dict[CurrencyName, Locality] = {
    CurrencyName.USD: Locality.US,
    CurrencyName.EUR: Locality.EU,
    CurrencyName.GBP: Locality.GB,
    CurrencyName.JPY: Locality.JP,
    CurrencyName.CHF: Locality.CH,
    CurrencyName.CLP: Locality.CL,
    CurrencyName.BRL: Locality.BR,
    CurrencyName.RUB: Locality.RU,
    CurrencyName.CNY: Locality.CN,
    CurrencyName.INR: Locality.IN,
    CurrencyName.AUD: Locality.AU,
    CurrencyName.NZD: Locality.NZ,
    CurrencyName.HKD: Locality.HK,
    CurrencyName.SGD: Locality.SG,
    CurrencyName.KRW: Locality.KR,
    CurrencyName.THB: Locality.TH,
    CurrencyName.COP: Locality.CO,
    CurrencyName.MXN: Locality.MX,
    CurrencyName.BND: Locality.BN,
    CurrencyName.SEK: Locality.SE,
    CurrencyName.NOK: Locality.NO,
    CurrencyName.PEN: Locality.PE,
}


class Currency:
    def __init__(self, name: CurrencyName):
        self.name = name
        self.locality: Locality = _CURRENCY_LOCALITY_MAP[name]

    @property
    def value(self) -> str:
        return self.name.value

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, Currency) and self.name == other.name

    def __repr__(self) -> str:
        return f"Currency({self.name.value})"

    def __str__(self) -> str:
        return self.name.value


Currency.USD = Currency(CurrencyName.USD)
Currency.EUR = Currency(CurrencyName.EUR)
Currency.GBP = Currency(CurrencyName.GBP)
Currency.JPY = Currency(CurrencyName.JPY)
Currency.CHF = Currency(CurrencyName.CHF)
Currency.CLP = Currency(CurrencyName.CLP)
Currency.BRL = Currency(CurrencyName.BRL)
Currency.RUB = Currency(CurrencyName.RUB)
Currency.CNY = Currency(CurrencyName.CNY)
Currency.INR = Currency(CurrencyName.INR)
Currency.AUD = Currency(CurrencyName.AUD)
Currency.NZD = Currency(CurrencyName.NZD)
Currency.HKD = Currency(CurrencyName.HKD)
Currency.SGD = Currency(CurrencyName.SGD)
Currency.KRW = Currency(CurrencyName.KRW)
Currency.THB = Currency(CurrencyName.THB)
Currency.COP = Currency(CurrencyName.COP)
Currency.MXN = Currency(CurrencyName.MXN)
Currency.BND = Currency(CurrencyName.BND)
Currency.SEK = Currency(CurrencyName.SEK)
Currency.NOK = Currency(CurrencyName.NOK)
Currency.PEN = Currency(CurrencyName.PEN)


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
        return self.name == other.name


@dataclass(slots=True)
class FX_Rate:
    currency_pair: CurrencyPair
    value: float

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError(f"FX_Rate value must be greater than 0. Got {self.value}.")

    def invert(self) -> Self:
        return FX_Rate(self.currency_pair.copy().invert(), 1 / self.value)

    def get_accrued_value_date(
        self,
        t: date,
        base_currency_curve: ZeroCouponCurve,
        quote_currency_curve: ZeroCouponCurve,
    ) -> float:
        base_df = base_currency_curve.get_df(t)
        quote_df = quote_currency_curve.get_df(t)
        return self.value * base_df / quote_df

    def get_accrued_values_dates(
        self,
        ts: list[date],
        base_currency_curve: ZeroCouponCurve,
        quote_currency_curve: ZeroCouponCurve,
    ) -> np.ndarray:
        base_df = base_currency_curve.get_dfs(ts)
        quote_df = quote_currency_curve.get_dfs(ts)
        return self.value * base_df / quote_df

    def copy(self) -> Self:
        return FX_Rate(self.currency_pair.copy(), self.value)

    def __copy__(self) -> Self:
        return self.copy()

    def __str__(self) -> str:
        return f"{self.currency_pair}@{self.value}"

    def __hash__(self) -> int:
        return hash(self.currency_pair.name)


@dataclass(slots=True)
class FX_RateData:
    currency_pair: CurrencyPair
    historic_data: dict[date, FX_Rate]

    def __post_init__(self):
        for fxr in self.historic_data.values():
            if fxr.currency_pair != self.currency_pair:
                raise ValueError(
                    f"All values in historic_data of FX_RateData object must match currency_pair. currency_pair: {self.currency_pair}, found {fxr.currency_pair} in historic_data"
                )

    def copy(self) -> Self:
        return FX_RateData(self.currency_pair.copy(), self.historic_data.copy())

    def __copy__(self) -> Self:
        return self.copy()

    def invert(self) -> Self:
        return FX_RateData(
            self.currency_pair.copy().invert(),
            {t: fxr.copy().invert() for t, fxr in self.historic_data.items()},
        )

    def get_fx_rate(self, t: date) -> FX_Rate:
        if t in self.historic_data:
            return self.historic_data[t]
        else:
            raise KeyError(f"Date {t} not found in {self.currency_pair} data.")

    def add_date(self, t: date, fx_rate: FX_Rate):
        self.historic_data[t] = fx_rate
