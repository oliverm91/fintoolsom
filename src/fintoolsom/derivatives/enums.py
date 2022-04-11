import numpy as np
from enum import Enum

class DerivativeType(Enum):
    IRS = 'irs'
    CCS = 'ccs'
    FXFORWARD = 'fxforward'
    FXCALL = 'fxcall'
    FXPUT = 'fxput'
    
class Position(Enum):
    Active = 'active'
    Passive = 'passive'
    
class PaymentType(Enum):
    PD = 'physical delivery'
    CS = 'cash settle'

class Derivative:
    def __init__(self, operation_number, derivative_type, active_leg, passive_leg, payment_type, collateral_index_name=None, compensation_currency=None):
        self.operation_number = operation_number
        self.derivative_type = derivative_type
        self.collateral_index_name = collateral_index_name
        self.active_leg = active_leg
        self.passive_leg = passive_leg
        self.payment_type = payment_type
        self.compensation_currency = compensation_currency
        

class Leg:
    def __init__(self, position, coupons, currency, initialize_vectors=True):
        '''
        :param cupones: lista Cupon
        :param recibe: boolean, True para Activo, False para Pasivo
        '''
        self.position = position
        self.currency = currency
        self.coupons = coupons

        self.payment_dates = None
        self.start_dates = None
        self.end_dates = None
        self.fixing_dates = None
        self.notionals = None
        self.yfs = None
        self.is_floating = None
        self.floating_index_name = None
        self.spread_interests = None
        self.fixed_interests = None
        self.fx_fixing_compensation_dates = None
        self.fixed_flows = None
        if initialize_vectors:
            self.__initialize_vectors()

    def __copy__(self):
        lg = Leg(self.coupons, self.currency, False)
        lg.discount_curve = self.discount_curve
        lg.discount_locality = self.discount_locality
        lg.payment_dates = self.payment_dates
        lg.start_dates = self.start_dates
        lg.end_dates = self.end_dates
        lg.fx_fixing_compensation_dates = self.fx_fixing_compensation_dates
        lg.fixing_dates = self.fixing_dates
        lg.notionals = self.notionals
        lg.is_floating = self.is_floating
        lg.floating_index_name = self.floating_index_name
        lg.fixed_flows = self.fixed_flows
        lg.is_overnight = self.is_overnight
        return lg

    def __initialize_vectors(self):
        if isinstance(self.coupons[0], FXForwardCoupon):
            return
        
        self.is_floating = isinstance(self.coupons[0], FloatingCoupon)
        self.floating_index_name = self.coupons[0].floating_index_name if self.is_floating else None
        
        self.fixing_dates = np.array([np.datetime64(coupon.dates['fixing_date']) for coupon in self.coupons])
        self.fx_fixing_compensation_dates = np.array([np.datetime64(coupon.dates['fx_fixing_date']) for coupon in self.coupons])
        self.start_dates = np.array([np.datetime64(coupon.dates['start_date'])  for coupon in self.coupons])
        self.end_dates = np.array([np.datetime64(coupon.dates['end_date']) for coupon in self.coupons])
        self.payment_dates = np.array([np.datetime64(coupon.dates['payment_date']) for coupon in self.coupons])
        self.notionals = np.array([coupon.notional for coupon in self.coupons])

        amortizations = np.array([coupon.amortization for coupon in self.coupons])
        spread_interests = np.array([coupon.spread_interest if isinstance(coupon, FloatingCoupon) else 0 for coupon in self.coupons])
        fixed_interests = np.array([coupon.interest if isinstance(coupon, FixedCoupon) else 0 for coupon in self.coupons])
        self.fixed_flows = spread_interests + fixed_interests + amortizations
        

class FixedCoupon:
    def __init__(self, notional, amortization, dates, fixed_rate):
        self.notional = notional
        self.dates = dates
        self.amortization = amortization
        self.fixed_rate = fixed_rate
        self.interest = self.fixed_rate.calculate_interest(self.notional, self.dates['start_date'], self.dates['end_date'])


class FloatingCoupon:
    def __init__(self, notional, amortization, spread, dates, floating_index_name):
        self.notional = notional
        self.amortization = amortization
        self.dates = dates
        self.floating_index_name = floating_index_name
        self.spread = spread
        self.spread_interest = self.spread.calculate_interest(self.notional, self.dates['start_date'], self.dates['end_date'])


class FXForwardCoupon:
    def __init__(self, notional, date):
        self.notional = notional
        self.date = date