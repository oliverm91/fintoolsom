from .Bonds import Bond
from datetime import date
from .. import rates
from .. import dates

class CLBond(Bond):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tera = kwargs.get('tera', None)
        self.tera = tera if tera is not None else self.calculate_tera()
        
    def calculate_tera(self) -> rates.Rate:
        tera_rate_convention = rates.RateConvention(rates.InterestConvention.Compounded, dates.DayCountConvention.Actual, 365)
        tera = self.get_irr(self.start_date, 100.0, tera_rate_convention)
        tera.rate_value = round(tera.rate_value, 6)
        self.tera = tera
        return tera
        
    def get_amount_value(self, date: date, irr: rates.Rate, fx: float=1.0) -> float:
        price, par_value = self.get_price(date, irr, price_decimals=4)
        amount = self.notional * price * par_value / 10_000.0
        if fx != 1.0:
            amount = round(amount, 8)
        amount = round(amount * fx, 0)
        return amount
    
    def get_par_value(self, date: date, decimals: int=8) -> float:
        current_coupon = self.coupons.get_current_coupon(date)
        par_value = current_coupon.residual + current_coupon.get_accrued_interest(date, self.tera)
        return round(par_value, decimals)