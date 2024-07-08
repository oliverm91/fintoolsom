from copy import copy
from typing import Self
import numpy as np
from datetime import date
from dataclasses import dataclass, field

from .Rates import Rate, RateConvention
from ..dates import ActualDayCountConvention


@dataclass(slots=True)
class ZeroCouponCurvePoint:
    date: date
    rate: Rate
    
    def copy(self):
        return ZeroCouponCurvePoint(self.date, copy(self.rate))
    
    def __copy__(self):
        return self.copy()
    
    def simple_str(self) -> str:
        return f'({self.date}|{self.rate.rate_value})'

@dataclass(slots=True)
class ZeroCouponCurve:
    curve_date: date

    curve_points: list[ZeroCouponCurvePoint] = field(default_factory=list)
    date_dfs: list[tuple[date, float]] = field(default_factory=list)
    _rate_conv_for_dfs_init: RateConvention = field(default=None)

    wfs: np.ndarray = field(init=False, default=None)
    dfs: np.ndarray = field(init=False, default=None)
    days: np.ndarray = field(init=False, default=None)
    dates: list[date] = field(init=False, default=None)
    rates: list[Rate] = field(init=False, default=None)
    _cashed_dfs: dict[str, np.ndarray] = field(init=False, default=None)
    _str_dates_rates: str = field(init=False, default=None)
    def __post_init__(self):
        if self._rate_conv_for_dfs_init is None:
            self._rate_conv_for_dfs_init = RateConvention()
        if not self.curve_points and self.date_dfs:
            ic = self._rate_conv_for_dfs_init.interest_convention
            dcc = self._rate_conv_for_dfs_init.day_count_convention
            tfb = self._rate_conv_for_dfs_init.time_fraction_base
            self.curve_points = [ZeroCouponCurvePoint(t_i, Rate(self._rate_conv_for_dfs_init, ic.get_rate_from_df(df, dcc.get_time_fraction(self.curve_date, t_i, tfb))))
                                for t_i, df in self.date_dfs]
        elif not self.curve_points:
                raise ValueError(f'If curve_points is not set, then date_dfs (list[tuples[date, float]]) must be set.')
        
        self._cashed_dfs: dict[str, np.ndarray] = {}
        self.sort()

    def set_df_curve(self):
        self.wfs = np.array([cp.rate.get_wealth_factor(self.curve_date, cp.date) 
                             for cp in self.curve_points])
        self.dfs = 1 / self.wfs
    
    def get_days(self) -> np.ndarray:
        days = ActualDayCountConvention.get_day_count(self.curve_date, self.dates)
        return days
    
    def sort(self):
        self.curve_points.sort(key=lambda cp: cp.date)
        for i, cp in enumerate(self.curve_points):
            if cp.date > self.curve_date:
                break
        self.curve_points = self.curve_points[i:]
        self.dates = [cp.date for cp in self.curve_points]
        self.rates = [cp.rate for cp in self.curve_points]
        self.days = self.get_days()
        self.set_df_curve()
        self._cashed_dfs.clear()
        
    def delete_point(self, date: date):
        self.curve_points = list(filter(lambda cp: cp.date != date, self.curve_points))
        self.sort()       
        
    def add_point(self, curve_point: ZeroCouponCurvePoint):
        if curve_point.date < self.curve_date:
            raise ValueError(f"Cannot add point with date before curve date. Curve date: {self.curve_date}, point date: {curve_point.date}")
        
        self.delete_point(curve_point.date)
        self.curve_points.append(curve_point)
        self.sort()

    def parallel_bump_rates_bps(self, bps: float):
        if isinstance(bps, np.ndarray):
            bps = float(bps[0])
        for zccp in self.curve_points:
            zccp.rate.rate_value += bps / 10_000
        
        # Copy of part of sort method. Other lines are not necessary here.
        self.set_df_curve()
        self._cashed_dfs.clear()

    def get_df(self, date: date) -> float:
        return self.get_dfs([date])[0]
    
    def get_dfs(self, dates_t: list[date]) -> np.ndarray:
        hashed_inputs = '|'.join(map(str, dates_t))
        if hashed_inputs in self._cashed_dfs:
            return self._cashed_dfs[hashed_inputs]
        tenors = ActualDayCountConvention.get_day_count(self.curve_date, dates_t)
        min_tenor = min(self.days)
        max_tenor = max(self.days)
        tenors_smaller_than_min = tenors[tenors<min_tenor]
        tenors_greater_than_max = tenors[tenors>max_tenor]
        normal_tenors = tenors[(tenors >= min_tenor) & (tenors <= max_tenor)]
        small_tenors_amount = len(tenors_smaller_than_min)
        greater_tenors_amount = len(tenors_greater_than_max)
        dfs = np.zeros(len(tenors))
        if small_tenors_amount:
            dfs[:small_tenors_amount] = self.curve_points[0].rate.get_discount_factor(self.curve_date, dates_t[:small_tenors_amount])
        if greater_tenors_amount:
            dfs[-greater_tenors_amount:] = self.curve_points[-1].rate.get_discount_factor(self.curve_date, dates_t[-greater_tenors_amount:])
        
        dfs[small_tenors_amount:len(tenors)-greater_tenors_amount] = np.exp(np.interp(normal_tenors, self.days, np.log(self.dfs))) # Log-Linear interpolation of discount factors
        self._cashed_dfs[hashed_inputs] = dfs
        return dfs
    
    def get_df_fwd(self, start_date: date, end_date: date) -> float:
        return self.get_df(end_date) / self.get_df(start_date)

    def get_dfs_fwds(self, start_dates: list[date], end_dates: list[date]) -> np.ndarray:
        if len(start_dates) != len(end_dates):
            raise ValueError(f"Start and end dates must have the same length. Start dates: {start_dates}, end dates: {end_dates}")
        end_dfs = self.get_dfs(end_dates)
        start_dfs = self.get_dfs(start_dates)
        fwds = end_dfs/start_dfs
        return fwds
    
    def get_wf(self, date: date) -> float:
        return 1 / self.get_df(date)

    def get_wfs(self, dates: list[date]) -> np.ndarray:
        return 1 / self.get_dfs(dates)
    
    def get_wf_fwd(self, start_date: date, end_date: date) -> float:
        return 1 / self.get_df_fwd(start_date, end_date)

    def get_wfs_fwds(self, start_dates: list[date], end_dates: list[date]) -> np.ndarray:
        if len(start_dates) != len(end_dates):
            raise ValueError(f"Start and end dates must have the same length. Start dates: {start_dates}, end dates: {end_dates}")
        df_fwds = self.get_dfs_fwds(start_dates, end_dates)
        wfs_fwds = 1 / df_fwds
        return wfs_fwds
    
    def get_forward_rates(self, start_dates: date | list[date], end_dates: date | list[date], rate_convention: RateConvention) -> Rate | list[Rate]:
        start_wfs = self.get_wfs(start_dates)
        end_wfs = self.get_wfs(end_dates)
        fwd_wfs = (end_wfs/start_wfs)
        tfb = rate_convention.time_fraction_base
        tfs = rate_convention.day_count_convention.get_time_fraction(start_dates, end_dates, tfb)
        fwd_rates = rate_convention.interest_convention.get_rate_from_wf(fwd_wfs, tfs)
        return fwd_rates

    def get_forward_rates_values(self, start_dates: date | list[date], end_dates: date | list[date], rate_convention: RateConvention=None) -> float | np.ndarray:
        rates_obj = self.get_forward_rates(start_dates, end_dates, rate_convention)
        if isinstance(rates_obj, np.ndarray):
            return rates_obj
        elif isinstance(rates_obj, list):
            return np.array([r.rate_value for r in rates_obj])
        else: 
            return rates_obj.rate_value

    def get_zero_rates(self, rate_convention: RateConvention=None) -> list[Rate]:
        if rate_convention is None:
            return [cp.copy() for cp in self.curve_points]
        else:
            rates_obj = []
            for cp in self.curve_points:
                r = cp.rate.copy()
                if r.rate_convention == rate_convention:
                    rates_obj.append(r)
                else:
                    r.convert_rate_convention(rate_convention)
                    rates_obj.append(r)
            return rates_obj
        
    def get_zero_rate(self, date: date,  rate_convention: RateConvention=None) -> Rate:
        if rate_convention is None:
            rate_convention = self.curve_points[0].rate.rate_convention
        
        df = self.get_df(date)
        r = rate_convention.interest_convention.get_rate_from_df(df, self.curve_date, date)
        return r

    def get_zero_rates_values(self, rate_convention: RateConvention=None) -> np.ndarray:
        rates_obj = self.get_zero_rates(rate_convention)
        return np.array([r.rate_value for r in rates_obj])
    
    def __len__(self) -> int:
        return len(self.curve_points)
  
    def copy(self) -> Self:
        return ZeroCouponCurve(self.curve_date, [zccp.copy() for zccp in self.curve_points])
    
    def __copy__(self) -> Self:
        return self.copy()
    
    def __str__(self) -> str:
        return f'ZeroCouponCurve class. Curve date {self.curve_date}. {str([cp.simple_str() for cp in self.curve_points])}'