from abc import ABC, abstractmethod
from datetime import date

from ..dates import DayCountConvention
from ..rates import RateConvention, InterestConvention, Rate


class Coupon(ABC):
    def __init__(self, amortization: float, residual: float, start_date: date, end_date: date, payment_date: date):
        if type(start_date)!=date or type(end_date)!=date or type(payment_date)!=date:
            try:
                start_date = start_date.date()
            except:
                raise TypeError(f'Could not cast start_date format to datetime.date:\n\t{type(start_date)} type received.')
            try:
                end_date = end_date.date()
            except:
                raise TypeError(f'Could not cast end_date format to datetime.date:\n\t{type(end_date)} type received.')
            try:
                payment_date = payment_date.date()
            except:
                raise TypeError(f'Could not cast payment_date format to datetime.date:\n\t{type(end_date)} type received.')
        if start_date >= end_date:
            raise ValueError(f'start_date {start_date} must be previous than end_date {end_date}.')
        if payment_date < end_date:
            raise ValueError(f'payment_date {payment_date} must be greater or equal to {end_date}.')
        
        self.start_date: date = start_date
        self.end_date: date = end_date
        self.payment_date: date = payment_date

        if residual < amortization:
            raise ValueError(f'residual {residual} must be greater or equal to amortization {amortization}')

        self.amortization = float(amortization)
        self.residual = float(residual)
        
    @abstractmethod
    def get_accrued_interest(self, date: date) -> float:
        raise NotImplementedError('Subclass must implement get_accrued_interest')
    
_default_fixed_rate_convention = RateConvention(interest_convention=InterestConvention.Linear, day_count_convention=DayCountConvention.Actual, time_fraction_base=360)
class FixedRateCoupon(Coupon):
    def __init__(self, amortization: float, residual: float, start_date: date, end_date: date, payment_date: date, interest: float, coupon_rate_convention: RateConvention=_default_fixed_rate_convention):
        super().__init__(amortization, residual, start_date, end_date, payment_date)
        self.interest = interest
        self.coupon_rate = self._get_coupon_rate(coupon_rate_convention)

    def _get_coupon_rate(self, coupon_rate_convention: RateConvention) -> Rate:
        wf = (self.interest + self.residual) / self.residual
        return Rate.get_rate_from_wf(wf, self.start_date, self.end_date, coupon_rate_convention)
    
    def get_accrued_interest(self, date: date) -> float:
        return self.coupon_rate.get_accrued_interest(self.residual, self.start_date, date)

    
