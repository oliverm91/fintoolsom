from .. import rates
from .. import dates
from datetime import date

class Deposit:
    def __init__(self, nemo: str, currency: str, payment_date: date, payment: float):
        self.currency = currency.lower()
        self.payment_date = payment_date
        self.payment = payment  
        self.nemo = nemo

    def get_value(self, t: date, rate_value: float, fx: int=1) -> float:
        ic = rates.InterestConvention.Linear
        dcc = dates.DayCountConvention.Actual
        base = 30 if self.currency=='clp' else 360
        rc = rates.RateConvention(ic, dcc, base)
        r = rates.Rate(rc, rate_value)
        
        wf = r.get_wealth_factor(t, self.payment_date)
        df = 1 / wf
        value = self.payment * df * fx
        
        return value
    
    def get_dv01(self, t: date, rate_value: float) -> float:
        dur = self.get_duration(t)
        pv = self.get_value(t, rate_value) / 100
        dv01 = - pv * dur / 10_000
        dv01 *= self.payment
        return dv01
    
    def get_duration(self, t: date, base_year_fraction: int=365) -> float:
        dur = (self.payment_date - t).days / base_year_fraction        
        return dur

    def __copy__(self):
        return Deposit(self.nemo, self.currency, self.payment_date, self.payment)