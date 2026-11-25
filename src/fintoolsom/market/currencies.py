from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import ClassVar, Self

import numpy as np

from ..rates import Curve
from .localities import Locality


class CurrencyName(Enum):
    USD = "USD"
    CAD = "CAD"
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
    CLF = "CLF"  # Unidad de Fomento (UF)


_CURRENCY_LOCALITY_MAP: dict[CurrencyName, Locality] = {
    CurrencyName.USD: Locality.US,
    CurrencyName.CAD: Locality.CA,
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
    CurrencyName.CLF: Locality.CL,
}


class Currency:
    """A currency, identified by its ISO name.

    Parameters
    ----------
    name : CurrencyName
        ISO name.
    settles_in : Currency, optional
        For a unit of account that cannot be delivered (the UF, ``CLF``), the
        currency its flows are paid in, converted at the rate of their fixing
        date. None for a deliverable currency.

    Attributes
    ----------
    locality : Locality
        Locality whose calendar the currency uses.
    """

    # One constant per CurrencyName, assigned after the class (they are instances
    # of it). Declared here so type checkers know them.
    USD: ClassVar["Currency"]
    CAD: ClassVar["Currency"]
    EUR: ClassVar["Currency"]
    GBP: ClassVar["Currency"]
    JPY: ClassVar["Currency"]
    CHF: ClassVar["Currency"]
    CLP: ClassVar["Currency"]
    BRL: ClassVar["Currency"]
    RUB: ClassVar["Currency"]
    CNY: ClassVar["Currency"]
    INR: ClassVar["Currency"]
    AUD: ClassVar["Currency"]
    NZD: ClassVar["Currency"]
    HKD: ClassVar["Currency"]
    SGD: ClassVar["Currency"]
    KRW: ClassVar["Currency"]
    THB: ClassVar["Currency"]
    COP: ClassVar["Currency"]
    MXN: ClassVar["Currency"]
    BND: ClassVar["Currency"]
    SEK: ClassVar["Currency"]
    NOK: ClassVar["Currency"]
    PEN: ClassVar["Currency"]
    CLF: ClassVar["Currency"]

    def __init__(self, name: CurrencyName, settles_in: "Currency | None" = None):
        self.name = name
        self.locality: Locality = _CURRENCY_LOCALITY_MAP[name]
        self.settles_in = (
            _CURRENCY_SETTLEMENT_MAP.get(name) if settles_in is None else settles_in
        )

    @property
    def value(self) -> str:
        """ISO code."""
        return self.name.value

    @property
    def is_deliverable(self) -> bool:
        """False for a unit of account settled in another currency (the UF)."""
        return self.settles_in is None

    @property
    def settlement_currency(self) -> "Currency":
        """Currency flows in this currency are paid in (itself when deliverable)."""
        return self if self.settles_in is None else self.settles_in

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, Currency) and self.name == other.name

    def __repr__(self) -> str:
        return f"Currency({self.name.value})"

    def __str__(self) -> str:
        return self.name.value


_CURRENCY_SETTLEMENT_MAP: dict[CurrencyName, Currency] = {}

Currency.USD = Currency(CurrencyName.USD)
Currency.CAD = Currency(CurrencyName.CAD)
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
# The UF is a CLP unit of account: UF flows are paid in CLP at the UF of their fixing.
_CURRENCY_SETTLEMENT_MAP[CurrencyName.CLF] = Currency.CLP
Currency.CLF = Currency(CurrencyName.CLF)

# Spot lag of an FX pair: business days, on the joint calendar of both currencies,
# from the trade date to the value date of a spot deal. Keyed by the two currencies
# in either order; a pair not listed spots at DEFAULT_FX_SPOT_LAG. Add or change an
# entry to set another market convention.
DEFAULT_FX_SPOT_LAG = 2
FX_SPOT_LAGS: dict[frozenset[Currency], int] = {
    frozenset((Currency.USD, Currency.CLP)): 1,
    frozenset((Currency.USD, Currency.CAD)): 1,
    frozenset((Currency.EUR, Currency.USD)): 2,
    frozenset((Currency.GBP, Currency.USD)): 2,
    frozenset((Currency.USD, Currency.JPY)): 2,
}


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

    @property
    def spot_lag(self) -> int:
        """Business days from the trade date to the spot value date: the pair's
        entry in :data:`FX_SPOT_LAGS` (either order), else
        :data:`DEFAULT_FX_SPOT_LAG`. USD/CLP and USD/CAD spot at T+1."""
        key = frozenset((self.base_currency, self.quote_currency))
        return FX_SPOT_LAGS.get(key, DEFAULT_FX_SPOT_LAG)

    def get_spot_date(self, trade_date: date) -> date:
        """Spot value date of a deal traded on `trade_date`: `spot_lag` business
        days later on the joint calendar of both currencies' localities.

        Parameters
        ----------
        trade_date : date

        Returns
        -------
        date
        """
        from ..dates.calendars import get_joint_calendar

        calendar = get_joint_calendar(
            self.base_currency.locality.value, self.quote_currency.locality.value
        )
        return calendar.add_business_days(trade_date, self.spot_lag)

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
        base_currency_curve: Curve,
        quote_currency_curve: Curve,
    ) -> float:
        base_df = base_currency_curve.get_df(t)
        quote_df = quote_currency_curve.get_df(t)
        return self.value * base_df / quote_df

    def get_accrued_values_dates(
        self,
        ts: list[date],
        base_currency_curve: Curve,
        quote_currency_curve: Curve,
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
    # (number of dates, last date inserted, latest date) when `latest_date` last ran.
    _latest: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def latest_date(self) -> date | None:
        """The latest date of `historic_data`, None when it is empty.

        Cached: it is recomputed when the number of dates or the date inserted last
        changes, which any addition or removal through `add_date` or directly on
        `historic_data` does (a dict keeps insertion order).
        """
        data = self.historic_data
        if not data:
            return None
        fingerprint = (len(data), next(reversed(data)))
        cached = self._latest
        if cached is None or cached[:2] != fingerprint:
            cached = self._latest = (*fingerprint, max(data))
        return cached[2]

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
