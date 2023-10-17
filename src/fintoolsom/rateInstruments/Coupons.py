from abc import ABC, abstractmethod
from datetime import date

from ..dates import DayCountConvention, get_day_count
from ..rates import RateConvention, InterestConvention, Rate


class Coupon(ABC):
    def __init__(self, amortization: float, residual: float, accrue_start_date: date, accrue_end_date: date, payment_date: date):
        if type(accrue_start_date)!=date or type(accrue_end_date)!=date or type(payment_date)!=date:
            try:
                accrue_start_date = accrue_start_date.date()
            except:
                raise TypeError(f'Could not cast start_date format to datetime.date:\n\t{type(accrue_start_date)} type received.')
            try:
                accrue_end_date = accrue_end_date.date()
            except:
                raise TypeError(f'Could not cast end_date format to datetime.date:\n\t{type(accrue_end_date)} type received.')
            try:
                payment_date = payment_date.date()
            except:
                raise TypeError(f'Could not cast payment_date format to datetime.date:\n\t{type(accrue_end_date)} type received.')
        if accrue_start_date >= accrue_end_date:
            raise ValueError(f'start_date {accrue_start_date} must be previous than end_date {accrue_end_date}.')
        if payment_date < accrue_end_date:
            raise ValueError(f'payment_date {payment_date} must be greater or equal to {accrue_end_date}.')
        
        self.accrue_start_date: date = accrue_start_date
        self.accrue_end_date: date = accrue_end_date
        self.payment_date: date = payment_date

        if residual < amortization:
            raise ValueError(f'residual {residual} must be greater or equal to amortization {amortization}')

        self.amortization = float(amortization)
        self.residual = float(residual)
        
    @abstractmethod
    def get_accrued_interest(self, date: date) -> float:
        raise NotImplementedError('Subclass must implement get_accrued_interest')
    
_default_fixed_rate_conv_linact360 = RateConvention(interest_convention=InterestConvention.Linear, day_count_convention=DayCountConvention.Actual, time_fraction_base=360)
class ABCFixedRateCoupon(Coupon):
    def __init__(self, amortization: float, residual: float, accrue_start_date: date, accrue_end_date: date, payment_date: date, interest: float):
        super().__init__(amortization, residual, accrue_start_date, accrue_end_date, payment_date)
        self.interest = interest
        self.flow = self.amortization + self.interest

class FixedRateCoupon(ABCFixedRateCoupon):
    def __init__(self, amortization: float, residual: float, accrue_start_date: date, accrue_end_date: date, payment_date: date, interest: float, coupon_rate_convention: RateConvention=_default_fixed_rate_conv_linact360):
        super().__init__(amortization, residual, accrue_start_date, accrue_end_date, payment_date, interest)
        self.coupon_rate = self._get_coupon_rate(coupon_rate_convention)

    def _get_coupon_rate(self, coupon_rate_convention: RateConvention) -> Rate:
        wf = (self.interest + self.residual) / self.residual
        return Rate.get_rate_from_wf(wf, self.accrue_start_date, self.accrue_end_date, coupon_rate_convention)
    
    def get_accrued_interest(self, date: date, accrue_rate: Rate=None) -> float:
        r = self.coupon_rate if accrue_rate is None else accrue_rate
        return r.get_accrued_interest(self.residual, self.accrue_start_date, date)

class InternationalBondFixedRateCoupon(ABCFixedRateCoupon):
    def __init__(self, amortization: float, residual: float, accrue_start_date: date, accrue_end_date: date, payment_date: date, interest: float, day_count_convention: DayCountConvention=DayCountConvention.Actual):
        super().__init__(amortization, residual, accrue_start_date, accrue_end_date, payment_date, interest)
        self.day_count_convention = day_count_convention

    def get_w(self, date: date) -> float:
        if not self.accrue_start_date <= date < self.accrue_end_date:
            raise ValueError(f'Date must be between accrue_start_date {self.accrue_start_date} and accrue_end_date {self.accrue_end_date}.')
        elapsed_days = get_day_count(self.accrue_start_date, date, day_count_convention=self.day_count_convention)
        total_coupon_days = get_day_count(self.accrue_start_date, self.accrue_end_date, day_count_convention=self.day_count_convention) 
        return elapsed_days / total_coupon_days
    
    def get_accrued_interest(self, date: date) -> float:
        return self.get_w(date) * self.interest