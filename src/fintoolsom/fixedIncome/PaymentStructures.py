from abc import ABC, abstractmethod
from datetime import date

import numpy as np

from .. import dates
from ..rates import Rate
from .Coupons import Coupon, FixedRateCoupon

_min_date, _max_date = date(1,1,1), date(9999, 12, 31)
class Coupons(ABC):
    def __init__(self, coupons_list: list[Coupon]):
        self.coupons = coupons_list
        self._sort()
        self.first_start_date = self.coupons[0].start_date
        self.end_dates = self.get_end_dates()
        self.start_dates = self.get_start_dates()
        self.payment_dates = self.get_payment_dates()
        self.amortizations = self.get_amortizations()

    def copy(self):
        return Coupons(self.coupons.copy())

    def _sort(self):
        self.coupons = sorted(self.coupons, key=lambda c: c.start_date)

    def get_amortizations(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        return np.array([c.amortization for c in self.coupons if c.payment_date >= start_date and c.payment_date <= end_date])

    @abstractmethod
    def get_interests(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        raise NotImplementedError('Subclass must implement get_interests')
    
    @abstractmethod
    def get_flows(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        raise NotImplementedError('Subclass must implement get_flows')
    
    def get_start_dates(self) -> np.ndarray:
        return np.array([c.start_date for c in self.coupons])
    
    def get_end_dates(self) -> np.ndarray:
        return np.array([c.end_date for c in self.coupons])
    
    def get_payment_dates(self) -> np.ndarray:
        return np.array([c.payment_date for c in self.coupons])
    
    def get_days_to_start_dates(self, date: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(date, self.start_dates, day_count_convention=day_count_convention)
    
    def get_days_to_end_dates(self, date: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(date, self.end_dates, day_count_convention=day_count_convention)
    
    def get_days_to_payment_dates(self, date: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(date, self.payment_dates, day_count_convention=day_count_convention)
    
    def get_current_coupon(self, date: date) -> Coupon:
        for c in self.coupons:
            if c.start_date <= date and c.payment_date > date:
                return c
        return None
    
    def get_remaining_coupons(self, date: date) -> list[Coupon]:
        return Coupons([c for c in self.coupons if c.payment_date >= date])
    
    def get_residual_amount(self, date: date) -> float:
        current_coupon = self.get_current_coupon(date)
        residual = current_coupon.residual
        return residual

    @abstractmethod
    def get_accrued_interest(self, date: date) -> float:
        raise NotImplementedError('Subclass must implement get_accrued_interest')


class FixedRateCoupons(Coupons):
    def __init__(self, fixed_rate_coupons_list: list[FixedRateCoupon]):
        super().__init__(fixed_rate_coupons_list)
        self.interest = self.get_interests()
        self.flows = self.get_flows()

    def get_interests(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        return np.array([c.interest for c in self.coupons if c.payment_date >= start_date and c.payment_date <= end_date])
    
    def get_flows(self, start_date: date=_min_date, end_date: date=_max_date) -> np.ndarray:
        return self.get_amortizations(start_date=start_date, end_date=end_date) + self.get_interests(start_date=start_date, end_date=end_date)
    
    def get_accrued_interest(self, date: date) -> float:
        return self.get_current_coupon(date).get_accrued_interest(date)
    
    def get_coupon_rate(self) -> Rate:
        c: FixedRateCoupon = self.coupons[0]
        return c.coupon_rate