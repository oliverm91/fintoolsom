from datetime import date

import numpy as np

from ..rates import Rate
from Bonds import Bond
from Coupons import InternationalBondFixedRateCoupon


class InternationalBond(Bond):
    def __init__(self, **kwargs):
        coupons = kwargs['coupons']
        for c in coupons:
            if not isinstance(c, InternationalBondFixedRateCoupon):
                raise TypeError(f'Coupons must be instances of InternationalBondFixedRateCoupon class. Type received: {type(c)}.')
        super().__init__(**kwargs)
        self.freq: int = kwargs['freq'] 
    
    def _get_initial_guess_for_irr(self) -> float:
        c: InternationalBondFixedRateCoupon = self.coupons.coupons[0]
        wild_guess = (c.interest / c.residual) * 365 / (c.accrue_end_date - c.accrue_end_date).days
        return wild_guess
    
    def get_flows_pv(self, date: date, annual_rate_value: float) -> np.ndarray:
        rate_value = annual_rate_value / self.freq
        w: float = self.coupons.get_current_coupon(date).get_w(date)
        remaining_coupons = self.coupons.get_remaining_coupons(date)
        df_exponents = np.array(range(len(remaining_coupons))) + w
        dfs = np.power(1 + rate_value, df_exponents)
        return remaining_coupons.flows * dfs
    
    def get_dirty_price(self, date, irr: Rate, current_par_value: float) -> float:
        return self.get_present_value(date, irr) * 100 / current_par_value
    
    def get_clean_price(self, date, irr: Rate, current_par_value: float) -> float:
        dirty_price = self.get_present_value(date, irr, current_par_value)
        accrued_interest = self.get_accrued_interest(date, irr) * 100 / dirty_price
        return dirty_price - accrued_interest