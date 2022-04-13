import numpy as np
from datetime import date
from collections.abc import Sequence
from mathsom import numerics, solvers
from .. import rates
from .. import dates
from ..rates import Rate, RateConvention


class Coupon:
    def __init__(self, amortization: float, interest: float, residual: float, start_date: date, end_date: date, accrue_rate_convention: rates.RateConvention):
        self.amortization = float(amortization)
        self.interest = float(interest)
        self.flow = self.amortization + self.interest
        self.residual = float(residual)
        if type(start_date)!=date or type(end_date)!=date:
            try:
                start_date = start_date.date()
            except:
                raise Exception(f'Could not cast start_date format to datetime.date:\n{type(start_date)} type received.')
            try:
                end_date = end_date.date()
            except:
                raise Exception(f'Could not cast end_date format to datetime.date:\n{type(end_date)} type received.')
        self.start_date = start_date
        self.end_date = end_date
        self.wf = (self.residual + self.interest) / self.residual
        self.accrue_rate = Rate.get_rate_from_wf(self.wf, self.start_date, self.end_date, accrue_rate_convention)
        
    def get_accrued_interest(self, date: date, accrue_rate: Rate=None) -> float:
        accrue_rate = self.accrue_rate if accrue_rate is None else accrue_rate
        if date >= self.end_date or date <= self.start_date:
            return 0
        accrued_interest = accrue_rate.get_accrued_interest(self.residual, self.start_date, date)
        return accrued_interest
    
    
class Coupons:
    def __init__(self, coupons: Sequence):
        self.coupons = coupons
        self.sort()

    def copy(self):
        return Coupons(self.coupons)

    def sort(self):
        self.coupons = sorted(self.coupons, key=lambda c: c.start_date)
        self.first_start_date = self.coupons[0].start_date
        self.flows = self.get_flows()
        self.end_dates = self.get_end_dates()
    
    def get_accrue_rate(self) -> float:
        return self.coupons[0].accrue_rate
    
    def get_flows(self) -> np.ndarray:
        return np.array([c.flow for c in self.coupons])
    
    def get_flows_between_dates(self, t1, t2) -> np.ndarray:
        return np.array([c.flow for c in self.coupons if t1 < c.end_date <= t2])
    
    def get_end_dates(self) -> np.ndarray:
        return np.array([c.end_date for c in self.coupons])
    
    def get_flows_maturities(self, date: date):
        return dates.get_day_count(date, self.end_dates, dates.DayCountConvention.Actual)
    
    def get_current_coupon(self, date: date) -> Coupon:
        for c in self.coupons:
            if c.start_date <= date and c.end_date > date:
                return c
        return None
    
    def get_residual_amount(self, date: date) -> float:
        current_coupon = self.get_current_coupon(date)
        residual = current_coupon.residual
        return residual

    def get_accrued_interest(self, date: date, rate: Rate=None) -> float:
        cc = self.get_current_coupon(date)
        return cc.get_accrued_interest(date, rate)


class Bond:
    def __init__(self, **kwargs):
        self.coupons = kwargs['coupons']
        self.currency = kwargs['currency']
        self.notional = kwargs['notional']
        self.start_date = self.coupons.first_start_date
        self.end_dates = self.coupons.end_dates
        self.accrue_rate = self.coupons.get_accrue_rate()
        self.flows_amount = self.coupons.get_flows()

    def copy(self):
        return Bond({'coupons': self.coupons, 'currency': self.currency, 'notional': self.notional})

    def get_maturity_date(self) -> date:
        '''
        Returns the maturity date of the bond.
        '''
        return max(self.end_dates)

    def get_accrued_interest(self, date: date, rate: Rate=None) -> float:
        accrued_interest = self.coupons.get_accrued_interest(date, rate)
        return accrued_interest
        
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
    
    def get_present_value_zc(self, date: date, zc_curve: rates.ZeroCouponCurve) -> float:
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
        irr_initial_guess = self.accrue_rate.rate_value
        objective_value = present_value
        args = [date, irr_initial_guess, irr_rate_convention]
        args_irr_index = 1
        irr_value = solvers.newton_raphson_solver(objective_value, self._get_present_value_rate_value, irr_initial_guess, args, args_irr_index)
        irr = rates.Rate(irr_rate_convention, irr_value)
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
    
    def get_price(self, date: date, irr: rates.Rate, price_decimals: int=4, par_value_decimals: int=8) -> float:
        pv = self.get_present_value(date, irr)
        par_value = self.get_par_value(date, decimals=par_value_decimals)
        price = round(100.0 * pv/par_value, price_decimals)
        return price, par_value
    
    def get_duration(self, date: date, irr: rates.Rate, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual, time_fraction_base: int=365) -> float:
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
    
    def get_dv01_approx(self, date: date, irr: rates.Rate, fx=1.0) -> float:
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
    
    def get_dv01(self, date: date, irr: rates.Rate, fx=1.0) -> float:
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
