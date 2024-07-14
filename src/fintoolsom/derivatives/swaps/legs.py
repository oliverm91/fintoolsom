from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .coupons import SwapCouponBase, FixedSwapCoupon, FloatingRateSwapCoupon, IborRateFloatingSwapCoupon, OvernightRateFloatingSwapCoupon
from ...market import Currency, Market, Locality
from ...dates import Calendar


@dataclass
class SwapLegBase:
    notional: float
    currency: Currency
    coupons: list[SwapCouponBase]

    start_dates: list[date] = field(init=False)
    end_dates: list[date] = field(init=False)
    payment_dates: list[date] = field(init=False)

    def __post_init__(self):
        self.coupons.sort(key=lambda coupon: coupon.start_accrual_date)
        if self.coupons[0].residual != self.notional:
            raise ValueError(f'Residual of first coupon must be equal to leg notional. First coupon residual: {self.coupons[0].residual}, Notional: {self.notional}')
        r = self.notional
        for coupon in self.coupons:
            if coupon.currency != self.currency:
                raise ValueError(f'Currency of coupon must be equal to leg currency. Coupon currency: {coupon.currency}, Leg currency: {self.currency}')
            if r != coupon.residual:
                raise ValueError(f'Coupon residual is not consistent with amortization of other coupons.')
            r -= coupon.amortization
        
        self.start_dates = [coupon.start_accrual_date for coupon in self.coupons]
        self.end_dates = [coupon.end_accrual_date for coupon in self.coupons]
        self.payment_dates = [coupon.payment_date for coupon in self.coupons]

    @abstractmethod        
    def get_flows_value(self, market: Market=None, locality: Locality=None) -> np.ndarray:
        pass

@dataclass
class DefaultSwapLeg(SwapLegBase):
    def __post_init__(self):
        super().__post_init__()
    def get_flows_value(self, market: Market=None, locality: Locality=None) -> np.ndarray:
        return np.array([coupon.get_flow_value(market=market, locality=locality) for coupon in self.coupons if coupon.payment_date > market.t])
    

@dataclass
class FixedSwapLeg(SwapLegBase):
    fixed_flows: np.ndarray = field(init=False)
    def __post_init__(self):
        super().__post_init__()
        for coupon in self.coupons:
            if not isinstance(coupon, FixedSwapCoupon):
                raise TypeError(f'Coupons must be of type FixedSwapCoupon. Got {type(coupon)}')
        self.fixed_flows = super().get_flows_value()

    def get_flows_value(self, market: Market = None, locality: Locality = None) -> np.ndarray:
        return self.fixed_flows
    

@dataclass
class FloatingSwapLeg(SwapLegBase):
    index_name: str

    amortizations: np.ndarray = field(init=False)
    residuals: np.ndarray = field(init=False)
    def __post_init__(self):
        super().__post_init__()
        for coupon in self.coupons:
            if not isinstance(coupon, FloatingRateSwapCoupon):
                raise TypeError(f'Coupons must be of type FloatingRateSwapCoupon. Got {type(coupon)}')
            
        self.amortizations = np.array([coupon.amortization for coupon in self.coupons])
        self.residuals = np.array([coupon.residual for coupon in self.coupons])

    def get_flows_value(self, **kwargs) -> np.ndarray:
        market: Market = kwargs.get('market')
        floating_curve = market.get_zero_coupon_curve(self.currency, self.index_name)
        future_payment_coupons = [coupon for coupon in self.coupons if coupon.payment_date > market.t]
        start_ix = 1 if self.first_coupon_is_fixed(market.t) else 0
        future_coupons_date_ix = future_payment_coupons + start_ix
        future_interests = floating_curve.get_accrued_interests(self.residuals[future_coupons_date_ix:], self.start_dates[future_coupons_date_ix:], self.end_dates[future_coupons_date_ix:])
        if start_ix == 1:
            future_payment_coupons_ix = len(self.coupons) - len(future_payment_coupons)
            current_coupon_flow = self.coupons[future_payment_coupons_ix].get_flow_value(**kwargs)
            current_coupon_interest = current_coupon_flow - self.amortizations[future_payment_coupons_ix]
            future_interests = np.insert(future_interests, 0, current_coupon_interest)

        flows = self.amortizations[future_payment_coupons:] + future_interests
        return flows

    @abstractmethod
    def first_coupon_is_fixed(self, t: date) -> bool:
        pass

class IborRateSwapLeg(FloatingSwapLeg):
    def __post_init__(self):
        super().__post_init__()
        for c in self.coupons:
            if not isinstance(c, IborRateFloatingSwapCoupon):
                raise TypeError(f'Coupons must be of type IborRateFloatingSwapCoupon. Got {type(c)}')
            
    def first_coupon_is_fixed(self, t: date) -> bool:
        return [c for c in self.coupons if c.payment_date > t][0].fixing_date <= t
    

class OvernightRateSwapLeg(FloatingSwapLeg):
    fixing_lag: int
    calendar: Calendar
    def __post_init__(self):
        super().__post_init__()
        for c in self.coupons:
            if not isinstance(c, OvernightRateFloatingSwapCoupon):
                raise TypeError(f'Coupons must be of type IborRateFloatingSwapCoupon. Got {type(c)}')
            
    def first_coupon_is_fixed(self, t: date) -> bool:
        fc: OvernightRateFloatingSwapCoupon = [c for c in self.coupons if c.payment_date > t][0]
        last_fixing_date = self.calendar.add_business_days(fc.end_accrual_date, -self.fixing_lag)
        return last_fixing_date <= t