from abc import ABC, abstractmethod
import numpy as np
from datetime import date
from scipy import optimize

from .. import dates
from ..rates import Rate, RateConvention, ZeroCouponCurve
from .PaymentStructures import FixedRateCoupons

class Bond(ABC):
    def __init__(self, **kwargs):
        self.coupons: FixedRateCoupons = kwargs['coupons']
        self.currency: str = kwargs['currency']
        self.notional: int = kwargs['notional']
        self.accrue_start_date: date = self.coupons.first_start_date
        self.accrue_end_dates: np.ndarray = self.coupons.accrue_end_dates
        self.flows_amount: np.ndarray = self.coupons.get_flows()

    def copy(self):
        return Bond({'coupons': self.coupons.copy(), 'currency': self.currency, 'notional': self.notional})

    def get_maturity_date(self) -> date:
        return max(self.accrue_end_dates)

    def get_accrued_interest(self, date: date) -> float:
        accrued_interest = self.coupons.get_accrued_interest(date)
        return accrued_interest
    
    # Check if w method can be implemented in CLBond. It can't... this will be an abstract method
    @abstractmethod
    def get_flows_pv(self, date: date, irr: Rate) -> np.ndarray:
        raise NotImplementedError('Subclass must implement get_flows_pv')
    
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
        return self.get_present_value(date, Rate(rate_convention, irr_value))
    
    def get_present_value_zc(self, date: date, zc_curve: ZeroCouponCurve) -> float:
        end_dates = self.coupons.get_end_dates()
        future_flows_mask = end_dates > date
        flows_dfs = zc_curve.get_dfs(end_dates) * future_flows_mask
        pvs = self.flows_amount * flows_dfs 
        pv = sum(pvs)
        return pv
    
    @abstractmethod
    def _get_initial_guess_for_irr(self) -> float:
        raise NotImplementedError('Subclass must implement get_initial_guess_for_irr')

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
        irr_initial_guess = self._get_initial_guess_for_irr()
        def objective_function(irr_value: float):
            return self._get_present_value_rate_value(date, irr_value, irr_rate_convention)
        irr_value = optimize(objective_function, present_value, irr_initial_guess, tol=1e-6)
        irr = Rate(irr_rate_convention, irr_value)
        return irr
    
    def get_accrued_interest(self, date: date, accrue_rate: Rate=None) -> float:
        '''
        Calculates the accrued interest to date.
        ----------
        Parameters:
        ----
            date(date): The date for which the par value is calculated.
            accrue_Rate(Rate): The rate that will be used to accrue. If not specified, coupon_rate will be used.
        ----
        Returns:
        ----
            get_accrued_interest(float): The accrued_interest to date.
        '''
        
        return self.coupons.get_accrued_interest(date, accrue_rate=accrue_rate)

    def get_duration(self, date: date, irr: Rate, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual, time_fraction_base: int=365) -> float:
        '''
        Calculates the bond duration as sumproduct(pv_coupon_i*yf_i)/sum(pv_coupon_i).
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
        tenors = dates.get_time_fraction(date, self.coupons.payment_dates, day_count_convention, time_fraction_base)
        pvs = self.get_flows_pv(date, irr)
        total_pv = sum(pvs)
        duration = sum(pvs * tenors) / total_pv
        return duration
    
    def get_dv01_approx(self, date: date, irr: Rate) -> float:
        '''
        Calculate dv01 of the bond with approximation formula: - present_value * duration / 10.000
        ----------
        Parameters:
        ----
            date (date): The date for which the dv01 is calculated.
            irr (Rate): The interest rate.
        ----
        Returns:
        ----
            dv01 (float): The dv01 of the bond.
        '''
        dur = self.get_duration(date, irr)
        pv = self.get_present_value(date, irr) / 100
        dv01 = - pv * dur / 10_000
        dv01 *= self.notional 
        return dv01
