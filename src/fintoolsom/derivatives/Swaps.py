from datetime import date
from typing import Union
from collections.abc import Sequence
import numpy as np

from .Derivatives import Derivative
from .enums import PaymentType, Position

from ..rates import Rate
from .. import dates
        
class SwapCouponDates:
    def __init__(self, accrue_start_date: date, accrue_end_date: date, index_fix_date: date, payment_date: date, fx_fix_date: date=None):
        self.accrue_start_date = accrue_start_date
        self.accrue_end_date = accrue_end_date
        self.index_fix_date = index_fix_date
        self.payment_date = payment_date
        if self.fx_fix_date is not None:
            self.fx_fix_date = fx_fix_date

class SwapFixedCoupon:
    def __init__(self, notional: float, amortization: float, dates: SwapCouponDates, fixed_rate: Rate):
        self.notional = notional
        self.dates = dates
        self.amortization = amortization
        self.fixed_rate = fixed_rate
        self.interest = self.fixed_rate.calculate_interest(self.notional, self.dates['start_date'], self.dates['end_date'])

class SwapFloatingCoupon:
    def __init__(self, notional, amortization, spread, dates, floating_index_name):
        self.notional = notional
        self.amortization = amortization
        self.dates = dates
        self.floating_index_name = floating_index_name
        self.spread = spread
        self.spread_interest = self.spread.calculate_interest(self.notional, self.dates['start_date'], self.dates['end_date'])

class SwapLeg:
    def __init__(self, coupons: Sequence, currency: str):
        '''
        :param cupones: lista Cupon
        :param recibe: boolean, True para Activo, False para Pasivo
        '''
        self.currency = currency
        self.coupons = coupons
        self.sort()

        self.payment_dates = None
        self.start_dates = None
        self.end_dates = None
        self.fixing_dates = None
        self.notionals = None
        self.is_floating = None
        self.floating_index_name = None
        self.spread_interests = None
        self.fixed_interests = None
        self.fx_fixing_compensation_dates = None
        self.fixed_flows = None

        self._initialize_vectors()

    def __copy__(self):
        lg = SwapLeg(self.coupons, self.currency)
        lg.payment_dates = self.payment_dates
        lg.start_dates = self.start_dates
        lg.end_dates = self.end_dates
        lg.fx_fixing_compensation_dates = self.fx_fixing_compensation_dates
        lg.fixing_dates = self.fixing_dates
        lg.notionals = self.notionals
        lg.is_floating = self.is_floating
        lg.floating_index_name = self.floating_index_name
        lg.fixed_flows = self.fixed_flows
        return lg

    def sort(self):
        self.coupons = sorted(self.coupons, key=lambda x: x.dates.payment_date)
        self._initialize_vectors()

    def _initialize_vectors(self):
        self.is_floating = isinstance(self.coupons[0], SwapFloatingCoupon)
        self.floating_index_name = self.coupons[0].floating_index_name if self.is_floating else None
        
        self.fixing_dates = np.array([np.datetime64(coupon.dates.index_fix_date) for coupon in self.coupons])
        if self.coupons[0].fx_fix_date is not None:
            self.fx_fixing_compensation_dates = np.array([np.datetime64(coupon.dates.fx_fix_date) for coupon in self.coupons])
        self.start_dates = np.array([np.datetime64(coupon.dates.start_date)  for coupon in self.coupons])
        self.end_dates = np.array([np.datetime64(coupon.dates.end_date) for coupon in self.coupons])
        self.payment_dates = np.array([np.datetime64(coupon.dates.payment_date) for coupon in self.coupons])
        self.notionals = np.array([coupon.notional for coupon in self.coupons])

        amortizations = np.array([coupon.amortization for coupon in self.coupons])
        spread_interests = np.array([coupon.spread_interest if isinstance(coupon, SwapFloatingCoupon) else 0 for coupon in self.coupons])
        fixed_interests = np.array([coupon.interest if isinstance(coupon, SwapFixedCoupon) else 0 for coupon in self.coupons])
        self.fixed_flows = spread_interests + fixed_interests + amortizations

class Swap(Derivative):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.legs = {}
        self.legs[Position.Active] = kwargs['active_leg']
        self.legs[Position.Passive] = kwargs['passive_leg']

    def reset_legs_vectors(self):
        for leg in self.legs.values():
            leg._initialize_vectors()

    def get_fixed_flows(self, position: Position) -> np.ndarray:
        return self.legs[position].fixed_flows

    def get_current_fixed_flows(self, t: date, position: Position) -> np.ndarray:
        fixed_flows = self.get_fixed_flows(position)
        current_fixed_flows = fixed_flows[self.legs[position].payment_dates > t]
        return current_fixed_flows    

    def get_payment_dates(self, position: Position) -> np.ndarray:
        return self.legs[position].payment_dates

    def get_current_payment_dates(self, t: date) -> np.ndarray:
        payment_dates = self.get_payment_dates()
        return payment_dates[payment_dates > t]

    def get_payment_day_count(self, position: Position, t: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(t, self.legs[position].payment_dates, day_count_convention)

    def get_current_payment_day_count(self, position: Position, t: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        pdc = self.get_payment_day_count(position, t, day_count_convention)
        return pdc[pdc > 0]

    def get_start_dates(self, position: Position) -> np.ndarray:
        return self.legs[position].start_dates

    def get_current_start_dates(self, position: Position, t: date) -> np.ndarray:
        start_dates = self.get_start_dates(position)
        return start_dates[start_dates > t]

    def get_start_day_count(self, position: Position, t: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(t, self.legs[position].start_dates, day_count_convention)

    def get_current_start_day_count(self, position: Position, t: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        sdc = self.get_start_day_count(position, t, day_count_convention)
        return sdc[sdc > 0]

    def get_end_dates(self, position: Position) -> np.ndarray:
        return self.legs[position].end_dates

    def get_current_end_dates(self, position: Position, t: date) -> np.ndarray:
        end_dates = self.get_end_dates(position)
        return end_dates[end_dates > t]

    def get_end_day_count(self, position: Position, t: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(t, self.legs[position].end_dates, day_count_convention)

    def get_current_end_day_count(self, position: Position, t: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        edc = self.get_end_day_count(position, t, day_count_convention)
        return edc[edc > 0]

    def get_fx_fixing_dates(self, position: Position) -> np.ndarray:
        return self.legs[position].fx_fixing_compensation_dates

    def get_current_fx_fixing_dates(self, position: Position, t: date) -> np.ndarray:
        fx_fixing_dates = self.get_fx_fixing_dates(position)
        return fx_fixing_dates[fx_fixing_dates > t]

    def get_fx_fixing_day_count(self, position: Position, t: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        return dates.get_day_count(t, self.legs[position].fx_fixing_compensation_dates, day_count_convention)

    def get_current_fx_fixing_day_count(self, position: Position, t: date, day_count_convention: dates.DayCountConvention=dates.DayCountConvention.Actual) -> np.ndarray:
        fx_fixing_day_count = self.get_fx_fixing_day_count(position, t, day_count_convention)
        return fx_fixing_day_count[fx_fixing_day_count > 0]

    def get_coupons(self, position: Position) -> np.ndarray:
        return self.legs[position].coupons

    def get_current_coupons(self, position: Position, t: date) -> np.ndarray:
        coupons = self.get_coupons(position)
        current_coupons = [c for c in coupons if c.payment_date > t]
        return current_coupons

    def get_current_coupon(self, position: Position, t: date) -> np.ndarray:
        coupons = self.get_current_coupons(position, t)
        return coupons[0]