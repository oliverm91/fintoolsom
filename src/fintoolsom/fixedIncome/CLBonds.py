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
        '''
        Calculates the TERA of the Chilean bond.
        --------
        Returns:
        ----
            tera (Rate): The TERA of the Chilean bond.
        '''
        tera_rate_convention = rates.RateConvention(rates.InterestConvention.Compounded, dates.DayCountConvention.Actual, 365)
        tera = self.get_irr(self.start_date, 100.0, tera_rate_convention)
        tera.rate_value = round(tera.rate_value, 6)
        self.tera = tera
        return tera
        
    def get_amount_value(self, date: date, irr: rates.Rate, fx: float=1.0) -> float:
        '''
        Calculates the amount to pay of the Chilean bond based on the given IRR.
        --------
        Parameters:
        ----
            date (date): The date to calculate the amount to pay.
            irr (Rate): The IRR to calculate the amount to pay.
            fx (float): Optional. The foreign exchange rate to calculate the amount to pay. Default is 1.0.
        ----
        Returns:
        ----
            float: The amount to pay.
        '''
        price, par_value = self.get_price(date, irr, price_decimals=4)
        amount = self.notional * price * par_value / 10_000.0
        if fx != 1.0:
            amount = round(amount, 8)
        amount = round(amount * fx, 0)
        return amount
    
    def get_par_value(self, date: date, decimals: int=8) -> float:
        '''
        Calculates the par value of the Chilean bond as of Notional + accruead interest of current coupon at TERA rate.
        --------
        Parameters:
        ----
            date (date): The date to calculate the par value.
            decimals (int): Optional. The number of decimals to round the par value. Default is 8.
        ----
        Returns:
        ----
            par_value (float): The par value.'''
        current_coupon = self.coupons.get_current_coupon(date)
        par_value = current_coupon.residual + current_coupon.get_accrued_interest(date, self.tera)
        return round(par_value, decimals)