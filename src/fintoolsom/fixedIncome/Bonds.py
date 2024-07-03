from copy import copy
from dataclasses import dataclass, field
from datetime import date
import math
from typing import Self

import numpy as np
from scipy.optimize import newton, minimize

from .. import rates
from .. import dates
from ..rates import Rate, RateConvention


@dataclass(slots=True)
class Coupon:
    amortization: float
    interest: float
    residual: float
    start_date: date
    end_date: date
    accrue_rate_convention: rates.RateConvention

    flow: float = field(init=False)
    wf: float = field(init=False)
    accrue_rate: rates.Rate = field(init=False)

    def __post_init__(self):
        self.validate_inputs()
        self.amortization = float(self.amortization)
        self.interest = float(self.interest)
        self.flow = self.amortization + self.interest
        self.residual = float(self.residual)
        self.wf = (self.residual + self.interest) / self.residual
        self.accrue_rate = Rate.get_rate_from_wf(self.wf, self.start_date, self.end_date, self.accrue_rate_convention)

    def validate_inputs(self):
        if not isinstance(self.amortization, (float, int)):
            raise TypeError(f"Amortization must be an int or float. Got {type(self.amortization)}")
        if not isinstance(self.residual, (float, int)):
            raise TypeError(f"Residual must be an int or float. Got {type(self.residual)}")
        if self.residual <= 0:
            raise ValueError(f"Residual must be greater than 0. Got {self.residual}")
        if self.residual < self.amortization:
            raise ValueError(f"Amortization cannot be greater than residual. Amortization: {self.amortization}, Residual: {self.residual}")

        if not isinstance(self.interest, (float, int)):
            raise TypeError(f"Interest must be an int or float. Got {type(self.interest)}")
        
        if not isinstance(self.start_date, date):
            raise TypeError(f"Start date must be of type date. Got {type(self.start_date)}")
        if not isinstance(self.end_date, date):
            raise TypeError(f"End date must be of type date. Got {type(self.end_date)}")        
        if self.start_date >= self.end_date:
            raise ValueError(f"Start date must be earlier than end date. Start date: {self.start_date}, End date: {self.end_date}")
        
        if not isinstance(self.accrue_rate_convention, rates.RateConvention):
            raise TypeError(f"accrue_rate_convention must be an instance of rates.RateConvention. Got {type(self.accrue_rate_convention)}")

    def copy(self) -> Self:
        return Coupon(self.amortization, self.interest, self.residual, self.start_date, self.end_date, self.accrue_rate.rate_convention.copy())
        
    def get_accrued_interest(self, date: date, accrue_rate: Rate=None) -> float:
        accrue_rate = self.accrue_rate if accrue_rate is None else accrue_rate
        if date >= self.end_date or date <= self.start_date:
            return 0
        accrued_interest = accrue_rate.get_accrued_interest(self.residual, self.start_date, date)
        return accrued_interest
    
    def __copy__(self) -> Self:
        return self.copy()
    
@dataclass
class Coupons:
    coupons: list[Coupon]

    def __post_init__(self):
        if not isinstance(self.coupons, list):
            raise TypeError(f"coupons must be of type list. Got {type(self.coupons)}")
        for c in self.coupons:
            if not isinstance(c, Coupon):
                raise TypeError(f'coupons elements must be of type Coupon. Got {type(c)}')
        self.sort()
        self.validate_residuals()
        

    def copy(self) -> Self:
        return Coupons([copy(c) for c in self.coupons])

    def __copy__(self) -> Self:
        return self.copy()

    def sort(self):
        self.coupons = sorted(self.coupons, key=lambda c: c.start_date)
        self.first_start_date = self.coupons[0].start_date
        self.flows = self.get_flows()
        self.end_dates = self.get_end_dates()

    def validate_residuals(self):
        for i in range(len(self.coupons)):
            calculated_residual = sum([self.coupons[j].amortization for j in range(i, len(self.coupons))])
            if round(math.floor(calculated_residual, 4), 1) != round(math.floor(self.coupons[i].residual, 4), 1):
                raise ValueError(f'Residual of coupon {i} is {self.coupons[i].residual} but sum of remaining amortizations is {calculated_residual}')
    
    def get_accrue_rate(self) -> Rate:
        return self.coupons[0].accrue_rate
    
    def get_flows(self) -> np.ndarray:
        return np.array([c.flow for c in self.coupons])
    
    def get_remaining_flows(self, date: date) -> np.ndarray:
        return np.array([c.flow for c in self.coupons if date < c.end_date])
    
    def get_flows_between_dates(self, t1, t2) -> np.ndarray:
        return np.array([c.flow for c in self.coupons if t1 < c.end_date <= t2])
    
    def get_end_dates(self) -> np.ndarray:
        return np.array([c.end_date for c in self.coupons])
    
    def get_remaining_end_dates(self, date: date) -> np.ndarray:
        return np.array([c.end_date for c in self.coupons if date < c.end_date])
    
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
        return cc.get_accrued_interest(date, accrue_rate=rate)
    
    def adjust_to_notional(self, notional: float):
        coupons_notional = self.coupons[0].residual
        ratio = notional / coupons_notional
        n = 0
        for coupon in self.coupons:
            coupon.amortization *= ratio
            n += coupon.amortization
            coupon.residual *= ratio
        assert round(float(n), 1) == round(float(notional), 1), f'Could not adjust to notional {notional}. Sum of amortization is {n}.'
        self.validate_residuals()


class Bond:
    def __init__(self, **kwargs):
        self.coupons: Coupons = kwargs['coupons']
        self.currency: str = kwargs['currency']
        self.currency = self.currency.lower()
        self.notional: float = kwargs['notional']
        self.coupons.adjust_to_notional(100) # Everything is calculated to N = 100. Then adjusted to notional. This makes Bond is made with coupons with base 100.
        self.start_date = self.coupons.first_start_date
        self.end_dates = self.coupons.end_dates
        self.accrue_rate = self.coupons.get_accrue_rate()
        self.flows_amount = self.coupons.get_flows()

    def copy(self) -> Self:
        return Bond(coupons=copy(self.coupons), currency=self.currency, notional=self.notional)

    def __copy__(self) -> Self:
        return self.copy()

    def get_maturity_date(self) -> date:
        '''
        Returns the maturity date of the bond.
        '''
        return max(self.end_dates)

    def get_accrued_interest(self, date: date, rate: Rate=None) -> float:
        accrued_interest = self.coupons.get_accrued_interest(date, rate)
        return accrued_interest
        
    def get_flows_pv(self, date: date, irr: Rate) -> np.ndarray:
        remaining_flows = self.coupons.get_remaining_flows(date)
        remaining_dates = self.coupons.get_remaining_end_dates(date)
        discount_factors = irr.get_discount_factor(date, remaining_dates)
        pvs = remaining_flows * discount_factors
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
        total_pv = pvs.sum()
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
        '''
        Retuns present value using a Zero Coupon Curve object.
        ----------
            date (date): date at which the present value is calculated.
            zc_curve (ZeroCouponCurve): Zero Coupon Curve to discount bond flows.
        ----
        Returns:
        ----
            present_value (float): present value of the bond at the given date.
        '''
        remaining_end_dates = self.coupons.get_remaining_end_dates(date)
        remaining_flows = self.coupons.get_remaining_flows(date)
        flows_pv = zc_curve.get_dfs(remaining_end_dates) * remaining_flows
        pv = flows_pv.sum()
        return pv
    
    def get_z_spread(self, date: date, irr: rates.Rate, zc_curve: rates.ZeroCouponCurve) -> int:
        '''
        Returns the z-spread given the internal rate of return (IRR) and a Zero Coupon Curve.
        This is, basis points to do a parallel bump to the Zero Coupon Curve so that IRR_value == ZCC_value.
        ----------
            date (date): date at which the z-spread is calculated.
            irr (Rate): internal rate of return to match.
            zc_curve (ZeroCouponCurve): Zero Coupon Curve to discount bond flows.
        ----
        Returns:
        ----
            z_spread (int): the z-spread in basis points.
        '''
        irr_value = self.get_present_value(date, irr)
        current_zc_value = self.get_present_value_zc(date, zc_curve)
        dv01 = self.get_dv01(date, irr)
        initial_guess = (irr_value - current_zc_value) / dv01
        def bp_bump_curve_value(bp_bump: float) -> float:
            zc_curve_bumped = copy(zc_curve)
            zc_curve_bumped.parallel_bump_rates_bps(bp_bump)
            return (self.get_present_value_zc(date, zc_curve_bumped) - irr_value)**2
        
        result = minimize(bp_bump_curve_value, initial_guess, options={'maxiter': 50})

        if result.success or result.fun < 1e-4:
            return int(round(result.x[0], 0))
        else:
            raise ValueError(f'Could not solve z-spread.\n-----------\nOptimization result:\n-----------\n{result}')

    
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
        def objective_function(irr_value: float) -> float:
            return self._get_present_value_rate_value(date, irr_value, irr_rate_convention) - present_value
        irr_value = newton(objective_function, x0=irr_initial_guess, tol=1e-8, maxiter=100)
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
        end_dates = self.coupons.get_remaining_end_dates(date)
        tenors = dates.get_time_fraction(date, end_dates, day_count_convention, time_fraction_base)
        pvs = self.get_flows_pv(date, irr)
        total_pv = sum(pvs)
        duration = sum(pvs * tenors) / total_pv
        return duration
    
    def get_dv01(self, date: date, irr: rates.Rate) -> float:
        '''
        Calculate dv01 of the bond with: - present_value * duration / 10.000
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
        dv01 = - pv * dur # Base 100 DV01
        dv01 *= self.notional / 100 # Notional adjusted
        return dv01