from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import math

from scipy.stats import norm

from ...dates import get_time_fraction, DayCountConvention
from ...rates import ZeroCouponCurve, Rate, RateConvention, InterestConvention
from .volatility_surface import VolatilitySurface

class OptionType(Enum):
    Call = 'call'
    Put = 'put'

@dataclass
class Option(ABC):
    notional: float
    strike: float
    maturity: date
    
    _valuation_rate_convention: RateConvention = field(default=RateConvention(interest_convention=InterestConvention.Exponential,
                                                                              day_count_convention=DayCountConvention.Actual,
                                                                              time_fraction_base=365))
    _option_sign: int = field(init=False)
    _sign: int = field(init=False)

    def __post_init__(self):
        self._option_sign = 1 if self.option_type==OptionType.Call else -1

    def get_mtm(self, t: float, spot: float, volatility_surface: VolatilitySurface, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve):
        d1, d2 = self._get_both_ds(t, spot, volatility_surface, domestic_curve, foreign_curve)
        r, q = self._get_rates(t, domestic_curve, foreign_curve)
        yf = self._get_yf(t)
        mtm = self._sign * self.notional * (spot*math.exp(-q*yf)*norm.cdf(self._sign*d1) - self.strike*math.exp(-r*yf)*norm.cdf(self._sign*d2))
        return mtm

    def get_delta(self, t: float, spot: float, volatility_surface: VolatilitySurface, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        _, q = self.get_rates(t, domestic_curve, foreign_curve)
        yf = self._get_yf(t)
        d1 = self._get_d1(t, spot, volatility_surface, domestic_curve, foreign_curve)
        delta = self._sign * self.notional * math.exp(-q*yf)*norm.cdf(self._sign * d1)
        return delta
    
    def get_gamma(self, t: float, spot: float, volatility_surface: VolatilitySurface, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        _, q = self._get_rates(t, domestic_curve, foreign_curve)
        yf = self._get_yf(t)
        d1, vol_sqrt_yf = self._get_d1(t, spot, volatility_surface, domestic_curve, foreign_curve, return_vol_sqrt_yf=True)
        gamma = self.notional * math.exp(-q*yf) * norm.ppf(d1) / (spot*vol_sqrt_yf)
        return gamma
    
    def get_vega(self, t: float, spot: float, volatility_surface: VolatilitySurface, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        _, q = self._get_rates(t, domestic_curve, foreign_curve)
        yf = self._get_yf(t)
        d1 = self._get_d1(t, spot, volatility_surface, domestic_curve, foreign_curve)
        vega = self.notional * spot * math.exp(-q*yf) * math.sqrt(yf) * norm.ppf(d1)
        return vega
    
    def _get_d1(self, t: date, spot: float,  volatility_surface: VolatilitySurface, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve,
                return_vol_sqrt_yf: bool=False) -> float | dict[str, float]:
        yf = self._get_yf(t)
        r, q = self._get_rates(t, domestic_curve, foreign_curve)
        vol = volatility_surface.get_volatility(self.maturity, self.strike)
        vol_sqrt_yf = vol*math.sqrt(yf)
        d1 = (math.log(spot/self.strike)+(r-q+vol*vol/2)*yf)/vol_sqrt_yf
        if not return_vol_sqrt_yf:
            return d1
        else:
            return d1, vol_sqrt_yf
    
    def _get_d2(self, t: date, spot: float,  volatility_surface: VolatilitySurface, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve,
                return_both_ds: bool=False) -> float | tuple[float]:
        d1, vol_sqrt_yf = self._get_d1(t, spot, volatility_surface, domestic_curve, foreign_curve, return_vol_sqrt_yf=True)
        d2 = d1-vol_sqrt_yf
        if not return_both_ds:
            return d2
        else:
            d1, d2
        
    def _get_yf(self, t: date):
        return get_time_fraction(t, self.maturity, DayCountConvention.Actual, base_convention=365)

    def _get_rates(self, t: date, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> dict[str, float]:
        df_r = domestic_curve.get_df(self.maturity)
        r = Rate.get_rate_from_df(df_r, t, self.maturity, self._valuation_rate_convention).rate_value
        df_q = foreign_curve.get_df(self.maturity)
        q = Rate.get_rate_from_df(df_q, t, self.maturity, self._valuation_rate_convention).rate_value
        return r, q

    def _get_both_ds(self, t: date, spot: float,  volatility_surface: VolatilitySurface, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve
                     ) -> float | tuple[float]:
        return self._get_d2(t, spot, volatility_surface, domestic_curve, foreign_curve, return_both_ds=True)


class Call(Option):
    def __post_init__(self):
        self._sign = 1
        super().__post_init__()
    
class Put(Option):
    def __post_init__(self):
        self._sign = -1
        super().__post_init__()