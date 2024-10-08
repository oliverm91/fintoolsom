from abc import ABC
from dataclasses import dataclass, field
from datetime import date
import math

import numpy as np
from scipy.stats import norm

from ...rates import ZeroCouponCurve, RateConvention, ExponentialInterestConvention
from ...dates import ActualDayCountConvention

_default_rc = RateConvention(interest_convention=ExponentialInterestConvention,
                            day_count_convention=ActualDayCountConvention,
                            time_fraction_base=365)
@dataclass
class Option(ABC):
    notional: float
    strike: float
    maturity: date
    
    _valuation_rate_convention: RateConvention = field(default=None)
    _sign: int = field(init=False)

    def __post_init__(self):
        if self._valuation_rate_convention is None:
            self._valuation_rate_convention = _default_rc

    def get_log_moneyness(self, spot: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        df_r = domestic_curve.get_df(self.maturity)
        df_q = foreign_curve.get_df(self.maturity)
        fwd_price = spot * df_q / df_r
        return np.log(self.strike / fwd_price)

    def get_mtm(self, t: float, spot: float, volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        d1, d2 = self._get_both_ds(t, spot, volatility, domestic_curve, foreign_curve)
        r, q = self._get_rates(t, domestic_curve, foreign_curve)
        yf = self._get_yf(t)
        mtm = self._sign * self.notional * (spot*math.exp(-q*yf)*norm.cdf(self._sign*d1) - self.strike*math.exp(-r*yf)*norm.cdf(self._sign*d2))
        return mtm

    def get_delta(self, t: float, spot: float, volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        _, q = self._get_rates(t, domestic_curve, foreign_curve)
        yf = self._get_yf(t)
        d1 = self._get_d1(t, spot, volatility, domestic_curve, foreign_curve)
        delta = self._sign * self.notional * math.exp(-q*yf)*norm.cdf(self._sign * d1)
        return delta
    
    def get_gamma(self, t: float, spot: float, volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        _, q = self._get_rates(t, domestic_curve, foreign_curve)
        yf = self._get_yf(t)
        d1, vol_sqrt_yf = self._get_d1(t, spot, volatility, domestic_curve, foreign_curve, return_vol_sqrt_yf=True)
        gamma = self.notional * math.exp(-q*yf) * norm.ppf(d1) / (spot*vol_sqrt_yf)
        return gamma
    
    def get_vega(self, t: float, spot: float, volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> float:
        _, q = self._get_rates(t, domestic_curve, foreign_curve)
        yf = self._get_yf(t)
        d1 = self._get_d1(t, spot, volatility, domestic_curve, foreign_curve)
        vega = self.notional * spot * math.exp(-q*yf) * math.sqrt(yf) * norm.ppf(d1)
        return vega
    
    def _get_d1(self, t: date, spot: float,  volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve,
                return_vol_sqrt_yf: bool=False) -> float | dict[str, float]:
        yf = self._get_yf(t)
        r, q = self._get_rates(t, domestic_curve, foreign_curve)
        vol_sqrt_yf = volatility*math.sqrt(yf)
        d1 = (math.log(spot/self.strike)+(r-q+volatility*volatility/2)*yf)/vol_sqrt_yf
        if not return_vol_sqrt_yf:
            return d1
        else:
            return d1, vol_sqrt_yf
    
    def _get_d2(self, t: date, spot: float,  volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve,
                return_both_ds: bool=False) -> float | tuple[float, float]:
        d1, vol_sqrt_yf = self._get_d1(t, spot, volatility, domestic_curve, foreign_curve, return_vol_sqrt_yf=True)
        d2 = d1-vol_sqrt_yf
        if not return_both_ds:
            return d2
        else:
            return d1, d2
        
    def _get_yf(self, t: date) -> float:
        return ActualDayCountConvention.get_time_fraction(t, self.maturity, 365)

    def _get_rates(self, t: date, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve) -> tuple[float, float]:
        df_r = domestic_curve.get_df(self.maturity)
        rc = _default_rc if self._valuation_rate_convention is None else self._valuation_rate_convention
        interest_conv = rc.interest_convention
        yf = rc.day_count_convention.get_time_fraction(t, self.maturity, rc.time_fraction_base)
        r = interest_conv.get_rate_from_df(df_r, yf)
        df_q = foreign_curve.get_df(self.maturity)
        q = interest_conv.get_rate_from_df(df_q, yf)
        return r, q

    def _get_both_ds(self, t: date, spot: float,  volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve
                     ) -> tuple[float, float]:
        return self._get_d2(t, spot, volatility, domestic_curve, foreign_curve, return_both_ds=True)
    
    @staticmethod
    def get_strike_from_delta(delta: float, spot: float, volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve, maturity: date, sign: int) -> float:
        rate_convention = _default_rc
        df_r = domestic_curve.get_df(maturity)
        t = domestic_curve.curve_date
        yf = rate_convention.day_count_convention.get_time_fraction(t, maturity, rate_convention.time_fraction_base)
        r = rate_convention.interest_convention.get_rate_from_df(df_r, yf)
        df_q = foreign_curve.get_df(maturity)
        q = rate_convention.interest_convention.get_rate_from_df(df_q, yf)

        k = spot * np.exp(-(sign*norm.ppf(sign*delta*(1/df_q)) * volatility * np.sqrt(yf) - (r - q + volatility*volatility/2)*yf))
        return k

@dataclass
class Call(Option):
    def __post_init__(self):
        self._sign = 1

    @staticmethod
    def get_strike_from_delta(delta: float, spot: float, volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve, maturity: date) -> float:
        return Option.get_strike_from_delta(delta, spot, volatility, domestic_curve, foreign_curve, maturity, 1)

@dataclass
class Put(Option):
    def __post_init__(self):
        self._sign = -1

    @staticmethod
    def get_strike_from_delta(delta: float, spot: float, volatility: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve, maturity: date) -> float:
        return Option.get_strike_from_delta(delta, spot, volatility, domestic_curve, foreign_curve, maturity, -1)