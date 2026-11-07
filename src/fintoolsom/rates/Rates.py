import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Self
import numpy as np

from ..dates import DayCountConventionBase, ActualDayCountConvention


class InterestConventionBase(ABC):
    @staticmethod
    @abstractmethod
    def get_wf_from_rate(
        rate_value: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        pass

    @abstractmethod
    def _strfy() -> str:
        pass

    @classmethod
    def __str__(cls) -> str:
        return cls._strfy()

    @staticmethod
    @abstractmethod
    def get_rate_from_wf(
        wf: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        pass

    @classmethod
    def get_df_from_rate(
        cls, rate_value: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        return 1 / cls.get_wf_from_rate(rate_value, time_fraction)

    @classmethod
    def get_rate_from_df(
        cls, df: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        return cls.get_rate_from_wf(1 / df, time_fraction)


class LinearInterestConvention(InterestConventionBase):
    @staticmethod
    def get_wf_from_rate(
        rate_value: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        return 1 + rate_value * time_fraction

    @staticmethod
    def get_rate_from_wf(
        wf: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        return (wf - 1) / time_fraction

    @staticmethod
    def _strfy():
        return "l"


class CompoundedInterestConvention(InterestConventionBase):
    @staticmethod
    def get_wf_from_rate(
        rate_value: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        return (1 + rate_value) ** time_fraction

    @staticmethod
    def get_rate_from_wf(
        wf: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        return wf ** (1 / time_fraction) - 1

    @staticmethod
    def _strfy():
        return "c"


class ExponentialInterestConvention(InterestConventionBase):
    @staticmethod
    def get_wf_from_rate(
        rate_value: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        return np.exp(rate_value * time_fraction)

    @staticmethod
    def get_rate_from_wf(
        wf: float | np.ndarray, time_fraction: float | np.ndarray
    ) -> float | np.ndarray:
        return np.log(wf) / time_fraction

    @staticmethod
    def _strfy():
        return "e"


@dataclass(slots=True)
class RateConvention:
    interest_convention: InterestConventionBase = field(
        default=CompoundedInterestConvention
    )
    day_count_convention: DayCountConventionBase = field(
        default=ActualDayCountConvention
    )  # Replaced by time_fraction_convention: TimeFractionBase
    time_fraction_base: int = field(default=365)  # Dissapears in next Update

    def __copy__(self) -> Self:
        return self.copy()

    def copy(self) -> Self:
        return RateConvention(
            interest_convention=self.interest_convention,
            day_count_convention=self.day_count_convention,
            time_fraction_base=self.time_fraction_base,
        )

    def __str__(self) -> str:
        # The conventions are classes: str() of a class ignores its __str__ classmethod.
        return (
            self.interest_convention.__str__()
            + self.day_count_convention.__str__()
            + str(self.time_fraction_base)
        )


_RATE_VALUE_DEPRECATION = "Rate.rate_value is deprecated; use Rate.rate instead."


def _warn_rate_value() -> None:
    warnings.warn(_RATE_VALUE_DEPRECATION, DeprecationWarning, stacklevel=3)


@dataclass(slots=True, init=False)
class Rate:
    """An interest rate value quoted in a rate convention.

    Parameters
    ----------
    rate_convention : RateConvention
        Interest convention, day count and time-fraction base of the rate.
    rate : float
        Rate value as a decimal (``0.05`` for 5%).
    rate_value : float, optional
        Deprecated keyword alias of `rate`; emits a ``DeprecationWarning``.

    Raises
    ------
    TypeError
        If neither or both of `rate` and `rate_value` are given.
    """

    rate_convention: RateConvention
    rate: float

    def __init__(
        self,
        rate_convention: RateConvention,
        rate: float | None = None,
        *,
        rate_value: float | None = None,
    ):
        if rate_value is not None:
            _warn_rate_value()
            if rate is not None:
                raise TypeError("Pass either rate or rate_value (deprecated), not both.")
            rate = rate_value
        if rate is None:
            raise TypeError("Rate() missing required argument: 'rate'.")
        self.rate_convention = rate_convention
        self.rate = rate

    @property
    def rate_value(self) -> float:
        """Deprecated alias of :attr:`rate`; emits a ``DeprecationWarning``."""
        _warn_rate_value()
        return self.rate

    @rate_value.setter
    def rate_value(self, value: float) -> None:
        _warn_rate_value()
        self.rate = value

    def __copy__(self) -> Self:
        return self.copy()

    def copy(self) -> Self:
        return Rate(self.rate_convention.copy(), self.rate)

    def get_wealth_factor(
        self, start_date: date | list[date], end_date: date | list[date]
    ) -> float | np.ndarray:
        tfb = self.rate_convention.time_fraction_base
        return self.rate_convention.interest_convention.get_wf_from_rate(
            self.rate,
            self.rate_convention.day_count_convention.get_time_fraction(
                start_date, end_date, tfb
            ),
        )

    def get_discount_factor(
        self, start_date: date | list[date], end_date: date | list[date]
    ) -> float | np.ndarray:
        wf = self.get_wealth_factor(start_date, end_date)
        df = 1 / wf
        return df

    def get_accrued_interest(
        self,
        n: float | np.ndarray,
        start_date: date | list[date],
        end_date: date | list[date],
    ) -> float | np.ndarray:
        wf = self.get_wealth_factor(start_date, end_date)
        interest = n * (wf - 1)
        return interest

    def convert_rate_convention(
        self, rate_convention: RateConvention, start_date: date, end_date: date
    ):
        current_wf = self.get_wealth_factor(start_date, end_date)
        self.rate_convention = rate_convention
        new_tf = self.rate_convention.day_count_convention.get_time_fraction(
            start_date, end_date, self.rate_convention.time_fraction_base
        )
        self.rate = self.rate_convention.interest_convention.get_rate_from_wf(
            current_wf, new_tf
        )
