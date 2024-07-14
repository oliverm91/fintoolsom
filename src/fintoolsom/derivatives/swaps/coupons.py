from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

from ...rates import Rate, RateConvention, ZeroCouponCurve
from ...dates import Calendar
from ...market import Currency, Market, Locality, Index


@dataclass
class SwapCouponBase(ABC):
    residual: float
    amortization: float
    start_accrual_date: date
    end_accrual_date: date
    currency: Currency

    payment_date: date

    @abstractmethod
    def get_flow_value(self, market: Market=None, locality: Locality=None) -> float:
        pass


@dataclass
class FixedSwapCoupon(SwapCouponBase):
    interest: float

    def __post_init__(self):
        self.flow = self.amortization + self.interest

    def get_flow_value(self, market: Market=None, locality: Locality=None) -> float:
        return self.flow


@dataclass
class FloatingRateSwapCoupon(SwapCouponBase):
    pass



@dataclass
class IborRateFloatingSwapCoupon(FloatingRateSwapCoupon):
    ibor_rate_name: str
    fixing_date: str

    def __post_init__(self):
        if self.fixing_date > self.start_accrual_date:
            raise ValueError(f'Fixing date cannot be greater than start accrual date. Fixing date: {self.fixing_date}, Start accrual date: {self.start_accrual_date}')

    def get_flow_value(self, market: Market=None, locality: Locality=None) -> float:
        if self.fixing_date <= market.t:
            rate = market.get_rate(self.fixing_date, self.ibor_rate_name)
            return self.amortization + rate.get_accrued_interest(self.residual, self.start_accrual_date, self.end_accrual_date)
        else:
            currency_index_curve = market.get_zero_coupon_curve(self.currency, index_name=self.ibor_rate_name)
            return self.amortization + currency_index_curve.get_accrued_interest(self.residual, self.start_accrual_date, self.end_accrual_date)
            

@dataclass
class OvernightRateFloatingSwapCoupon(FloatingRateSwapCoupon):
    rate_name: str

    def get_flow_value(self, calendar: Calendar=None, fixing_lag: int=0, market: Market=None, locality: Locality=None) -> float:
        if calendar is None:
            calendar = Calendar()

        t = self.start_accrual_date
        aux_index = 100
        # Accrue with past data
        while t < self.end_accrual_date:
            fixing_date = calendar.add_business_days(t, -fixing_lag)
            next_t = min(calendar.add_business_days(t, 1), self.end_accrual_date)
            if fixing_date <= market.t:
                rate = market.get_rate(fixing_date, self.rate_name)
                aux_index *= rate.get_wealth_factor(t, next_t)
            t = next_t
        
        # Now t is >= self.end_accrual_date
        rate_index = market.interest_rate_to_index_map[self.rate_name]
        index_curve = market.get_zero_coupon_curve(self.currency, index_name=rate_index)
        aux_index *= index_curve.get_wf(self.end_accrual_date)

        interest = (aux_index / 100 - 1) * self.residual

        return self.amortization + interest


@dataclass
class IndexedSwapCoupon(FloatingRateSwapCoupon):
    index_name: str
    payment_date: date

    def get_flow_value(self, market: Market=None, locality: Locality=None) -> float:
        index_curve = market.get_zero_coupon_curve(self.currency, index_name=self.index_name)
        market_date_index = market.get_index(market.t, self.index_name).value
        if self.start_accrual_date <= market.t:
            start_index = market.get_index(self.start_accrual_date, self.index_name).value
        else:
            start_index = market_date_index * index_curve.get_wf(self.start_accrual_date)

        if self.end_accrual_date <= market.t:
            end_index = market.get_index(self.end_accrual_date, self.index_name).value
        else:
            end_index = market_date_index * index_curve.get_wf(self.end_accrual_date)

        interest = self.residual * (end_index / start_index - 1)
        return self.amortization + interest