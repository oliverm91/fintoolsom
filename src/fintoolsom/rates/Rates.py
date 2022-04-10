from enum import Enum
from datetime import date
import numpy as np
from collections.abc import Sequence
from typing import Union, get_args

from .. import dates


class InterestConvention(Enum):
    Linear = 1
    Compounded = 2
    Exponential = 3


class RateConvention:
    def __init__(self, interest_convention=InterestConvention.Compounded, day_count_convention=dates.DayCountConvention.Actual, time_fraction_base=365):
        self.interest_convention = interest_convention
        self.day_count_convention = day_count_convention
        self.time_fraction_base = time_fraction_base


class Rate:
    def __init__(self, rate_convention: RateConvention, rate_value: float):
        self.rate_value = rate_value
        self.rate_convention = rate_convention
        self._wf_router = {
            InterestConvention.Linear: self._get_wf_from_linear_rate,
            InterestConvention.Compounded: self._get_wf_from_compounded_rate,
            InterestConvention.Exponential: self._get_wf_from_exponential_rate
        }
        
    def copy(self):
        return Rate(self.rate_convention, self.rate_value)

    def _get_wf_from_linear_rate(self, rate_value, start_date, end_date):
        time_fraction = self.get_time_fraction(start_date, end_date)
        wf = (1 + rate_value * time_fraction)
        return wf

    def _get_wf_from_compounded_rate(self, rate_value, start_date, end_date):
        time_fraction = self.get_time_fraction(start_date, end_date)
        wf = (1 + rate_value) ** time_fraction
        return wf

    def _get_wf_from_exponential_rate(self, rate_value, start_date, end_date):
        time_fraction = self.get_time_fraction(start_date, end_date)
        wf = np.exp(rate_value * time_fraction)
        return wf
        
    def get_wealth_factor(self, start_date, end_date) -> Union[np.ndarray, float]:
        func = self._wf_router[self.rate_convention.interest_convention]
        wf = func(self.rate_value, start_date, end_date)
        return wf

    def get_discount_factor(self, start_date, end_date) ->  Union[np.ndarray, float]:
        wf = self.get_wealth_factor(start_date, end_date)
        df = 1 / wf
        return df
    
    def get_days_count(self, start_date, end_date) -> Union[np.ndarray, int]:
        days_count = dates.get_day_count(start_date, end_date, self.rate_convention.day_count_convention)
        return days_count
    
    def get_time_fraction(self, start_date, end_date) -> Union[np.ndarray, float]:
        days_count = self.get_days_count(start_date, end_date)
        time_fraction = days_count / self.rate_convention.time_fraction_base
        return time_fraction
        
    def get_accrued_interest(self, n: float, start_date: date, end_date: date) -> Union[np.ndarray, float]:
        wf = self.get_wealth_factor(start_date, end_date)
        interest = n * (wf - 1)
        return interest
        
    def convert_rate_conventions(self, rate_convention: RateConvention, start_date: date, end_date: date):
        current_wf = self.get_wealth_factor(start_date, end_date)        
        self.rate_convention = rate_convention
        new_rate = get_rate_from_wf(current_wf, self.rate_convention)
        self.rate_value = new_rate.rate_value
        

def get_rate_from_df(df: Union[Sequence, np.ndarray, float], start_date, end_date, rate_convention: RateConvention) -> Union[Sequence, Rate]:
    wf = 1/df
    return get_rate_from_wf(wf, start_date, end_date, rate_convention)

def get_rate_from_wf(wf: Union[Sequence, np.ndarray, float], start_date, end_date, rate_convention: RateConvention) -> Union[Sequence, Rate]:
    time_fraction = dates.get_time_fraction(start_date, end_date, rate_convention.day_count_convention, rate_convention.time_fraction_base)
    
    ic = rate_convention.interest_convention
    if ic == InterestConvention.Linear:
        rate_value = (wf - 1) / time_fraction
    elif ic == InterestConvention.Compounded:
        rate_value = np.power(wf, 1 / time_fraction) - 1 
    elif ic == InterestConvention.Exponential:
        rate_value = np.log(wf) / time_fraction
        
    if isinstance(time_fraction,  np.ndarray):
        return [Rate(rate_value, rv) for rv in rate_value]    
    else:
        return Rate(rate_convention, rate_value)