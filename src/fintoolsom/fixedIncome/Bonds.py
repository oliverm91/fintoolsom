import numpy as np
from datetime import date
from mathsom import numerics, solvers
from .. import dates
from ..rates import Rate, RateConvention, ZeroCouponCurve
from .PaymentStructures import FixedRateCoupons

class Bond:
    def __init__(self, **kwargs):
        self.coupons: FixedRateCoupons = kwargs['coupons']
        self.currency: str = kwargs['currency']
        self.notional: int = kwargs['notional']
        self.start_date: date = self.coupons.first_start_date
        self.end_dates: np.ndarray = self.coupons.end_dates
        self.coupon_rate: Rate = self.coupons.get_coupon_rate()
        self.flows_amount: np.ndarray = self.coupons.get_flows()

    def copy(self):
        return Bond({'coupons': self.coupons.copy(), 'currency': self.currency, 'notional': self.notional})

    def get_maturity_date(self) -> date:
        return max(self.end_dates)

    def get_accrued_interest(self, date: date) -> float:
        accrued_interest = self.coupons.get_accrued_interest(date)
        return accrued_interest
    
    # Check if w method can be implemented in CLBond
    def get_flows_pv(self, date: date, irr: Rate) -> np.ndarray:
        future_flows_mask = self.end_dates > date
        wealth_factors = irr.get_wealth_factor(date, self.end_dates)
        pvs = self.flows_amount  * future_flows_mask / wealth_factors
        return pvs
    
    def get_present_value(self, date: date, irr: Rate) -> float:
        '''
        Returns the present value of the bond at a given date.
        ----------
            date (date): date at which the present value is calculated.
            irr_value (float): irr value of the bond.
            rate_convention (RateConvention): rate convention of the bond.
        ----
        Returns:
        ----
            present_value (float): present value of the bond at the given date.
        '''
        pvs = self.get_flows_pv(date, irr)
        total_pv = sum(pvs)
        return total_pv

    def _get_present_value_rate_value(self, date: date, irr_value: float, rate_convention: RateConvention) -> float:
        '''
        Returns the present value of the bond at a given date.
        ----------
            date (date): date at which the present value is calculated.
            irr_value (float): irr value of the bond.
            rate_convention (RateConvention): rate convention of the bond.
        ----
        Returns:
        ----
            present_value (float): present value of the bond at the given date.
        '''
        pvs = self.get_flows_pv(date, Rate(rate_convention, irr_value))
        total_pv = sum(pvs)
        return total_pv
    
    def get_present_value_zc(self, date: date, zc_curve: ZeroCouponCurve) -> float:
        end_dates = self.coupons.get_end_dates()
        future_flows_mask = end_dates > date
        flows_dfs = zc_curve.get_dfs(end_dates) * future_flows_mask
        pvs = self.flows_amount * flows_dfs 
        pv = sum(pvs)
        return pv
    
    def get_irr_from_present_value(self, date: date, present_value: float, irr_rate_convention: RateConvention) -> Rate:
        '''
        Calculates the internal rate of return of a bond for a given present value.
        --------
        Parameters:
        ----
            date (date): The date for which the IRR is calculated.
            present_value (float): The present value of the bond.
            irr_rate_convention (RateConvention): The convention for the IRR.
        ----
        Returns:
        ----
            irr (float): The internal rate of return of the bond.'''
        irr_initial_guess = self.coupon_rate.rate_value
        objective_value = present_value
        args = [date, irr_initial_guess, irr_rate_convention]
        args_irr_index = 1
        irr_value = solvers.newton_raphson_solver(objective_value, self._get_present_value_rate_value, irr_initial_guess, args, args_irr_index)
        irr = Rate(irr_rate_convention, irr_value)
        return irr
    
    def get_par_value(self, date: date, decimals: int=8) -> float:
        '''
        Calculates the par value (as of notional + accrued interest at current coupon rate) of the bond.
        ----------
        Parameters:
        ----
            date(date): The date for which the par value is calculated.
            decimals(int): The number of round the result.
        ----
        Returns:
        ----
            par_value(float): The par value of the bond.
        '''
        current_coupon = self.coupons.get_current_coupon(date)
        par_value = current_coupon.residual + current_coupon.get_accrued_interest(date)
        return round(par_value, decimals)
    
    def get_price(self, date: date, irr: Rate, price_decimals: int=4, par_value_decimals: int=8) -> float:
        pv = self.get_present_value(date, irr)
        par_value = self.get_par_value(date, decimals=par_value_decimals)
        price = round(100.0 * pv/par_value, price_decimals)
        return price, par_value
    
    def get_duration(self, date: date, irr: Rate, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual, time_fraction_base: int=365) -> float:
        '''
        Calculates the bond duration.
        ----------
        Parameters:
        ----
            date (date): The date for which the duration is calculated.
            irr (Rate): The interest rate.
            day_count_convention (DayCountConvention): Optional. The day count convention. Default is Actual.
            time_fraction_base (int): The time fraction base. Default is 365.
        ----
        Returns:
        ----
            duration (float): The bond duration.
        '''
        end_dates = self.coupons.get_end_dates()
        tenors = dates.get_time_fraction(date, end_dates, day_count_convention, time_fraction_base)
        pvs = self.get_flows_pv(date, irr)
        total_pv = sum(pvs)
        duration = sum(pvs * tenors) / total_pv
        return duration
    
    def get_dv01_approx(self, date: date, irr: Rate, fx=1.0) -> float:
        '''
        Calculate dv01 of the bond with approximation formula: - present_value * duration / 10.000
        ----------
        Parameters:
        ----
            date (date): The date for which the dv01 is calculated.
            irr (Rate): The interest rate.
            fx (float): Optional. The foreign exchange rate. Default is 1.
        ----
        Returns:
        ----
            dv01 (float): The dv01 of the bond.
        '''
        dur = self.get_duration(date, irr)
        pv = self.get_present_value(date, irr) / 100
        dv01 = - pv * dur * fx / 10_000
        dv01 *= self.notional 
        return dv01
    
    def get_dv01(self, date: date, irr: Rate, fx=1.0) -> float:
        '''
        Calculate dv01 of the bond with numeric differentiation.
        ----------
        Parameters:
        ----
            date (date): The date for which the dv01 is calculated.
            irr (Rate): The interest rate.
            fx (float): Optional. The foreign exchange rate. Default is 1.
        ----
        Returns:
        ----
            dv01 (float): The dv01 of the bond.
        '''
        irr_value = irr.rate_value
        irr_rate_convention = irr.rate_convention
        valuation_parameters = [date, irr_value, irr_rate_convention]
        valuation_function = self._get_present_value_rate_value
        variable_derivation_index = 1
        slope = numerics.differentiate(valuation_function, irr_value, valuation_parameters, variable_derivation_index)
        slope /= 100 # Makes flows Notional = 1
        dv01 = slope / 10_000 # DV01 of Notional = 1
        dv01 *= self.notional # Adjust to Notional = self.notional
        dv01 *= fx # Adjust for fx
        return dv01
