from enum import Enum
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Self
import numpy as np

from .. import dates


class InterestConvention(Enum):
    Linear = 1
    Compounded = 2
    Exponential = 3


@dataclass(slots=True)
class RateConvention:
    interest_convention: InterestConvention = field(default=InterestConvention.Compounded)
    day_count_convention: dates.DayCountConvention = field(default=dates.DayCountConvention.Actual)
    time_fraction_base: int = field(default=365)

    def __copy__(self):
        return self.copy()
    
    def copy(self):
        return RateConvention(interest_convention=self.interest_convention, day_count_convention=self.day_count_convention, time_fraction_base=self.time_fraction_base)

@dataclass
class Rate:
    rate_convention: RateConvention
    rate_value: float
    _wf_router: dict[InterestConvention, Callable[[float, Iterable[date] | date, Iterable[date] | date], np.ndarray | float]] = field(init=False)
    def __post_init__(self):
        self._wf_router = {
            InterestConvention.Linear: self._get_wf_from_linear_rate,
            InterestConvention.Compounded: self._get_wf_from_compounded_rate,
            InterestConvention.Exponential: self._get_wf_from_exponential_rate
        }
        
    def __copy__(self):
        return self.copy()
    
    def copy(self):
        return Rate(self.rate_convention.copy(), self.rate_value)

    def _get_wf_from_linear_rate(self, rate_value: float, start_date: Iterable[date] | date, end_date: Iterable[date] | date) -> np.ndarray | float:
        time_fraction = self._get_time_fraction(start_date, end_date)
        wf = (1 + rate_value * time_fraction)
        return wf

    def _get_wf_from_compounded_rate(self, rate_value: float, start_date: Iterable[date] | date, end_date: Iterable[date] | date) -> np.ndarray | float:
        time_fraction = self._get_time_fraction(start_date, end_date)
        wf = (1 + rate_value) ** time_fraction
        return wf

    def _get_wf_from_exponential_rate(self, rate_value: float, start_date: Iterable[date] | date, end_date: Iterable[date] | date) -> np.ndarray | float:
        time_fraction = self._get_time_fraction(start_date, end_date)
        wf = np.exp(rate_value * time_fraction)
        return wf

    def get_wealth_factor(self, start_date: date, end_date: date) -> np.ndarray | float:
        wf = self._wf_router[self.rate_convention.interest_convention](self.rate_value, start_date, end_date)
        return wf

    def get_discount_factor(self, start_date: date, end_date: date) ->  np.ndarray | float:
        wf = self.get_wealth_factor(start_date, end_date)
        df = 1 / wf
        return df
    
    def _get_time_fraction(self, start_date: Iterable[date] | date, end_date: Iterable[date] | date) -> np.ndarray | float:
        days_count = dates.get_day_count(start_date, end_date, self.rate_convention.day_count_convention)
        time_fraction = days_count / self.rate_convention.time_fraction_base
        return time_fraction
        
    def get_accrued_interest(self, n: float, start_date: date, end_date: date) -> np.ndarray | float:
        wf = self.get_wealth_factor(start_date, end_date)
        interest = n * (wf - 1)
        return interest
        
    def convert_rate_conventions(self, rate_convention: RateConvention, start_date: date, end_date: date):
        current_wf = self.get_wealth_factor(start_date, end_date)        
        self.rate_convention = rate_convention
        new_rate: Rate = self.get_rate_from_wf(current_wf, start_date, end_date, self.rate_convention)
        self.rate_value = new_rate.rate_value

    @staticmethod
    def _get_linear_rate_values_from_wf(wf: float | np.ndarray, start_date: date | list[date] | np.ndarray, end_date: date | list[date] | np.ndarray, rate_convention: RateConvention) -> float | np.ndarray:
        time_fraction = dates.get_time_fraction(start_date, end_date, rate_convention.day_count_convention, rate_convention.time_fraction_base)
        rate_value = (wf - 1) / time_fraction
        return rate_value

    @staticmethod
    def _get_compounded_rate_values_from_wf(wf: float | np.ndarray, start_date: date | list[date] | np.ndarray, end_date: date | list[date] | np.ndarray, rate_convention: RateConvention) -> float | np.ndarray:
        time_fraction = dates.get_time_fraction(start_date, end_date, rate_convention.day_count_convention, rate_convention.time_fraction_base)
        rate_value = np.power(wf, 1/time_fraction) - 1
        return rate_value

    @staticmethod
    def _get_exponential_rate_values_from_wf(wf: float | np.ndarray, start_date: date | list[date] | np.ndarray, end_date: date | list[date] | np.ndarray, rate_convention: RateConvention) -> float | np.ndarray:
        time_fraction = dates.get_time_fraction(start_date, end_date, rate_convention.day_count_convention, rate_convention.time_fraction_base)
        rate_value = np.log(wf) / time_fraction
        return rate_value

    @staticmethod
    def _rate_values_to_rate_fl(rate_values: float, rate_convention: RateConvention) -> Self:
        return Rate(rate_convention, rate_values)

    @staticmethod  
    def _rate_values_to_rate_npfl(rate_values: np.float64, rate_convention: RateConvention) -> Self:
        return Rate(rate_convention, rate_values)

    @staticmethod
    def _rate_values_to_rate_npar(rate_values: np.ndarray, rate_convention: RateConvention) -> list[Self]:
        return [Rate(rate_convention, rv) for rv in rate_values]

    
    _rate_values_to_rate_router: dict[type, Callable[[float | np.float64 | np.ndarray, RateConvention], Self | list[Self]]] = {
        float: _rate_values_to_rate_fl.__func__,
        np.float64: _rate_values_to_rate_npfl.__func__,
        np.ndarray: _rate_values_to_rate_npar.__func__
    }
    _rate_router: dict[InterestConvention, Callable[[float | np.ndarray, date | list[date] | np.ndarray, date | list[date] | np.ndarray, RateConvention], float | np.ndarray]] = {
        InterestConvention.Linear: _get_linear_rate_values_from_wf.__func__,
        InterestConvention.Compounded: _get_compounded_rate_values_from_wf.__func__,
        InterestConvention.Exponential: _get_exponential_rate_values_from_wf.__func__
    }    
    @staticmethod
    def get_rate_from_wf(wf: list[float] | np.ndarray | float, start_date: list[date] | np.ndarray | float, end_date: list[date] | np.ndarray | float, rate_convention: RateConvention) -> Self | list[Self]:
        func = Rate._rate_router[rate_convention.interest_convention]
        rate_values = func(wf, start_date, end_date, rate_convention)
        func_rate = Rate._rate_values_to_rate_router[type(rate_values)]
        rate_objs = func_rate(rate_values, rate_convention)
        return rate_objs

    @staticmethod
    def get_rate_from_df(df: list[float] | np.ndarray | float, start_date: list[date] | np.ndarray | float, end_date: list[date] | np.ndarray | float, rate_convention: RateConvention) -> Self | list[Self]:
        wf = 1/df
        return Rate.get_rate_from_wf(wf, start_date, end_date, rate_convention)