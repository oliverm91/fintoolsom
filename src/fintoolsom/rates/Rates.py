from enum import Enum
from datetime import date
import numpy as np

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
        
    def copy(self):
        return Rate(self.rate_convention, self.rate_value)
        
    def get_wealth_factor(self, start_date, end_date) -> float:
        time_fraction = dates.get_time_fraction(start_date, end_date, self.rate_convention.day_count_convention, self.rate_convention.time_fraction_base)
        wf = None
        if self.rate_convention.interest_convention == InterestConvention.Linear:
            wf = (1 + self.rate_value * time_fraction)
        elif self.rate_convention.interest_convention == InterestConvention.Compounded:
            wf = (1 + self.rate_value) ** time_fraction
        elif self.rate_convention.interest_convention == InterestConvention.Exponential:
            wf = np.exp(self.rate_value * time_fraction)

        return wf
    
    def get_days_count(self, start_date, end_date) -> int:
        days_count = dates.get_day_count(start_date, end_date, self.rate_convention.day_count_convention)
        return days_count
    
    def get_time_fraction(self, start_date, end_date) -> float:
        days_count = self.get_days_count(start_date, end_date)
        time_fraction = days_count / self.rate_convention.time_fraction_base
        return time_fraction
        
    def get_accrued_interest(self, n: float, start_date: date, end_date: date) -> float:
        wf = self.get_wealth_factor(start_date, end_date)
        interest = n * (wf - 1)
        return interest
    
    def get_rate_value_from_wf(self, wf, start_date, end_date):
        time_fraction = self.get_time_fraction(start_date, end_date)
        
        rate_value = self.rate_value # variable initialization

        if self.rate_convention.interest_convention == InterestConvention.Linear:
            rate_value = (wf - 1) / time_fraction
        elif self.rate_convention.interest_convention == InterestConvention.Compounded:
            rate_value = np.power(wf, 1 / time_fraction) - 1 
        elif self.rate_convention.interest_convention == InterestConvention.Exponential:
            rate_value = np.log(wf) / time_fraction
           
        return rate_value 
        
    def convert_rate_conventions(self, rate_convention: RateConvention, start_date: date, end_date: date):
        current_wf = self.get_wealth_factor(start_date, end_date)
        
        self.rate_convention = rate_convention
        self.rate_value = self.get_rate_value_from_wf(current_wf, start_date, end_date)
    
    