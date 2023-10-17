from abc import ABC, abstractmethod
from datetime import date
from typing import Self

import numpy as np

from .. import dates
from ..rates import Rate
from .Coupons import Coupon, FixedRateCoupon, InternationalBondFixedRateCoupon

_min_date, _max_date = date(1,1,1), date(9999, 12, 31)
class Coupons(ABC):
    def __init__(self, coupons_list: list[Coupon]):
        self.coupons = coupons_list
        self._sort()
        self.first_start_date = self.coupons[0].accrue_start_date
        self.accrue_end_dates = self.get_end_dates()
        self.accrue_start_dates = self.get_start_dates()
        self.payment_dates = self.get_payment_dates()
        self.amortizations = self.get_amortizations()

    def copy(self) -> Self:
        return Coupons(self.coupons.copy())

    def _sort(self):
        self.coupons = sorted(self.coupons, key=lambda c: c.accrue_start_date)

    def get_amortizations(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        return np.array([c.amortization for c in self.coupons if c.payment_date >= start_date and c.payment_date <= end_date])

    @abstractmethod
    def get_interests(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        raise NotImplementedError('Subclass must implement get_interests')
    
    @abstractmethod
    def get_flows(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        raise NotImplementedError('Subclass must implement get_flows')
    
    def get_start_dates(self) -> np.ndarray:
        return np.array([c.accrue_start_date for c in self.coupons])
    
    def get_end_dates(self) -> np.ndarray:
        return np.array([c.accrue_end_date for c in self.coupons])
    
    def get_payment_dates(self) -> np.ndarray:
        return np.array([c.payment_date for c in self.coupons])
    
    def get_days_to_start_dates(self, date: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(date, self.accrue_start_dates, day_count_convention=day_count_convention)
    
    def get_days_to_end_dates(self, date: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(date, self.accrue_end_dates, day_count_convention=day_count_convention)
    
    def get_days_to_payment_dates(self, date: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(date, self.payment_dates, day_count_convention=day_count_convention)
    
    def get_current_coupon(self, date: date) -> Coupon:
        for c in self.coupons:
            if c.accrue_start_date <= date and c.payment_date > date:
                return c
        return None
    
    def get_remaining_coupons(self, date: date) -> Self:
        return Coupons([c for c in self.coupons if c.payment_date >= date])
    
    def get_residual_amount(self, date: date) -> float:
        current_coupon = self.get_current_coupon(date)
        residual = current_coupon.residual
        return residual

    @abstractmethod
    def get_accrued_interest(self, date: date) -> float:
        raise NotImplementedError('Subclass must implement get_accrued_interest')


class FixedRateCoupons(Coupons):
    def __init__(self, fixed_rate_coupons_list: list[FixedRateCoupon] | list[InternationalBondFixedRateCoupon]):
        super().__init__(fixed_rate_coupons_list)
        self.interest = self.get_interests()
        self.flows = self.get_flows()

    def get_interests(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        return np.array([c.interest for c in self.coupons if c.payment_date >= start_date and c.payment_date <= end_date])
    
    def get_flows(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        return np.array([c.flow for c in self.coupons if c.payment_date >= start_date and c.payment_date <= end_date])
        
    def get_accrued_interest(self, date: date, accrue_rate: Rate=None) -> float:
        return self.get_current_coupon(date).get_accrued_interest(date, accrue_rate=accrue_rate)