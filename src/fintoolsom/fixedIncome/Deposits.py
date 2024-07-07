from typing import Self
from datetime import date

from ..rates import Rate, RateConvention, LinearInterestConvention
from ..dates import ActualDayCountConvention


class Deposit:
    def __init__(self, nemo: str, currency: str, payment_date: date, payment: float):
        self.currency = currency.lower()
        self.payment_date = payment_date
        self.payment = payment  
        self.nemo = nemo

    def get_value(self, t: date, rate_value: float, fx: int=1) -> float:
        base = 30 if self.currency=='clp' else 360
        rc = RateConvention(LinearInterestConvention, ActualDayCountConvention, base)
        r = Rate(rc, rate_value)
        
        df = r.get_discount_factor(t, self.payment_date)
        value = self.payment * df * fx
        
        return value
    
    def get_dv01(self, t: date, rate_value: float) -> float:
        dur = self.get_duration(t)
        pv = self.get_value(t, rate_value)
        dv01 = - pv * dur / 10_000
        return dv01
    
    def get_duration(self, t: date, base_year_fraction: int=365) -> float:
        dur = (self.payment_date - t).days / base_year_fraction
        return dur
    
    def copy(self) -> Self:
        return Deposit(self.nemo, self.currency, self.payment_date, self.payment)
    
    def __copy__(self) -> Self:
        return self.copy()
