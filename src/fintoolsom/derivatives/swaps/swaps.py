from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

from ...rates import Rate, ZeroCouponCurve

@dataclass
class Coupon(ABC):
    start_accrue_date: date
    end_accrue_date: date

    payment_date: date
    residual: float
    amortization: float
    
    @abstractmethod
    def get_flow(self, rate: Rate) -> float:
        pass

@dataclass
class NDCoupon(Coupon, ABC):
    fx_fixing_date: date

    def _get_fixing_spot(self, mkt) -> float:
        pass


@dataclass
class FixedCoupon(Coupon):
    fixed_rate: Rate    
    def get_accrual(self) -> float:
        return self.fixed_rate.get_accrued_interest(self.residual, self.start_accrue_date, self.end_accrue_date)
    
    def get_flow(self) -> float:
        return self.amortization + self.get_accrual(self.fixed_rate)
    
@dataclass
class NDFixedCoupon(NDCoupon, FixedCoupon):
    pass
    
@dataclass
class Floating(Coupon):
    fixed_rate: Rate    
    def get_accrual(self) -> float:
        return self.fixed_rate.get_accrued_interest(self.residual, self.start_accrue_date, self.end_accrue_date)
    
    def get_flow(self) -> float:
        return self.amortization + self.get_accrual(self.fixed_rate)
    



@dataclass
class NDCoupon(Coupon):
    fx_fixing_date: date

@dataclass


@dataclass
class Swap:
    active_coupons