from dataclasses import dataclass, field
from datetime import date

from ...rates import ZeroCouponCurve

class Forward:
    notional: float
    strike: float
    payment_date: date
    is_buy: bool

    sign: int = field(init=False)
    
    def __post_init__(self):
        self.sign = 1 if self.is_buy else -1

    def get_mtm(self, spot: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        df_d = domestic_curve.get_df(self.payment_date)
        df_f = foreign_curve.get_df(self.payment_date)
        notional_leg_vp = spot * self.notional * df_f
        strike_leg_vp = self.strike * self.notional * df_d

        mtm = self.sign * (notional_leg_vp - strike_leg_vp)
        return mtm
    
class NDF(Forward):
    fixing_date: date

    def __post_init__(self):
        super().__post_init__()

    def get_mtm(self, spot: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        df_d = domestic_curve.get_df(self.fixing_date)
        df_f = foreign_curve.get_df(self.fixing_date)
        
        s_t = spot * df_f / df_d

        fv = self.sign * self.notional (s_t - self.strike)
        mtm = fv * domestic_curve.get_df(self.payment_date)
        return mtm