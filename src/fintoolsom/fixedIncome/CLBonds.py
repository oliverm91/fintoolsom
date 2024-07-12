from copy import copy
from datetime import date
from typing import Self
from scipy.optimize import newton

from .Bonds import Bond
from ..rates.Rates import Rate, RateConvention, CompoundedInterestConvention
from ..dates import ActualDayCountConvention


class CLBond(Bond):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tera = kwargs.get('tera', None)
        self.tera = tera if tera is not None else self.calculate_tera()
        self.irr_default_convention = RateConvention(interest_convention=CompoundedInterestConvention, 
                                                          day_count_convention=ActualDayCountConvention,
                                                          time_fraction_base=365)

    def __copy__(self) -> Self:
        return CLBond(**{'coupons': copy(self.coupons), 'currency': self.currency, 'notional': self.notional, 'tera': self.tera})
        
    def calculate_tera(self) -> Rate:
        '''
        Calculates the TERA of the Chilean bond.
        --------
        Returns:
        ----
            tera (Rate): The TERA of the Chilean bond.
        '''
        tera_rate_convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
        tera = self.get_irr_from_present_value(self.start_date, 100.0, tera_rate_convention)
        tera.rate_value = round(tera.rate_value, 6)
        self.tera = tera
        return tera
        
    def get_amount_value(self, date: date, irr: Rate, fx: float=1.0) -> float:
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

    def _get_amount_value_rate_value(self, date: date, rate_value: float, rate_convention: RateConvention, fx: float=1.0) -> float:
        '''
        Calculates the amount to pay of the Chilean bond based on the given rate value.
        --------
        Parameters:
        ----
            date (date): The date to calculate the amount to pay.
            rate_value (float): The rate value to calculate the amount to pay.
            rate_convention (RateConvention): The rate convention to calculate the amount to pay.
            fx (float): Optional. The foreign exchange rate to calculate the amount to pay. Default is 1.0.
        ----
        Returns:
        ----
            float: The amount to pay.
        '''
        rate = Rate(rate_convention, rate_value)
        return self.get_amount_value(date, rate, fx)

    def get_irr_from_amount(self, date: date, amount: float, irr_rate_convention: RateConvention=None, fx: float=1.0, tol: float=1e-6) -> Rate:
        '''
        Calculates the IRR of the Chilean bond based on the given amount.
        --------
        Parameters:
        ----
            date (date): The date to calculate the IRR.
            amount (float): The amount to calculate the IRR.
            fx (float): Optional. The foreign exchange rate to calculate the IRR. Default is 1.0.
        ----
        Returns:
        ----
            irr (Rate): The IRR of the bond.
        '''
        if irr_rate_convention is None:
            irr_rate_convention = self.irr_default_convention
        tera_value = self.get_amount_value(date, self.tera, fx=fx)
        dv01 = self.get_dv01(date, self.tera) * fx
        initial_guess = self.tera.rate_value + ((amount - tera_value) / dv01) / 10_000
        div_value = min(self.notional / 1000, 10)
        def objective_function(irr_value: float) -> float:
            return (self._get_amount_value_rate_value(date, irr_value, irr_rate_convention, fx=fx) - amount)/div_value
        irr_value = newton(objective_function, x0=initial_guess, tol=tol, maxiter=100)
        rate_value = round(irr_value, 6)
        newton_result = self._get_amount_value_rate_value(date, rate_value, irr_rate_convention, fx=fx)
        counter = 0
        if newton_result != amount:
            direction = 1 if newton_result > amount else -1
            while True:
                rate_value += 0.000001 * direction
                newton_result_i = self._get_amount_value_rate_value(date, rate_value, irr_rate_convention, fx=fx)
                direction_i = 1 if newton_result_i > amount else -1
                if direction_i != direction:
                    if abs(newton_result_i-amount)>abs(newton_result-amount):
                        rate_value -= 0.000001 * direction
                    break
                else:
                    newton_result = newton_result_i
                counter += 1
                if counter == 30:
                    ig_value = self._get_amount_value_rate_value(date, initial_guess, irr_rate_convention, fx=fx)
                    raise ValueError(f'Failed to adjust to amount.\nInitial guess {initial_guess}, IG value {ig_value}, Amount {amount}, Newton_result {newton_result}, rate_value {rate_value}.')

        irr = Rate(irr_rate_convention, rate_value)
        return irr
    
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