from copy import copy
import numpy as np
from datetime import date
from dataclasses import dataclass, field

from .. import rates
from .. import dates


@dataclass(slots=True)
class ZeroCouponCurvePoint:
    date: date
    rate: rates.Rate
    
    def copy(self):
        return ZeroCouponCurvePoint(self.date, copy(self.rate))
    
    def __copy__(self):
        return self.copy()

@dataclass
class ZeroCouponCurve:
    curve_date: date

    curve_points: list[ZeroCouponCurvePoint] = field(default_factory=list)
    date_dfs: list[tuple[date, float]] = field(default_factory=list)
    _rate_conv_for_dfs_init: rates.RateConvention = field(default=None)
    def __post_init__(self,):
        if self._rate_conv_for_dfs_init is None:
            self._rate_conv_for_dfs_init = rates.RateConvention()
        if not self.curve_points and self.date_dfs:
            self.curve_points = [ZeroCouponCurvePoint(t_i, rates.Rate.get_rate_from_df(df, self.curve_date, t_i, self._rate_conv_for_dfs_init)) 
                                    for t_i, df in self.date_dfs]
        elif not self.curve_points:
                raise ValueError(f'If curve_points is not set, then date_dfs (list[tuples[date, float]]) must be set.')
        
        self._cashed_dfs: dict[str, np.ndarray] = {}
        self.sort()

    def __len__(self) -> int:
        return len(self.curve_points)
  
    def copy(self):
        return ZeroCouponCurve(self.curve_date, [copy(zccp) for zccp in self.curve_points])
    
    def set_df_curve(self):
        self.wfs = np.array([cp.rate.get_wealth_factor(self.curve_date, cp.date) 
                             if cp.date >= self.curve_date else 1 
                             for cp in self.curve_points])
        self.dfs = 1 / self.wfs
        self._cashed_dfs.clear()
        
    def set_days(self):
        self.days = self.get_days()
        self._cashed_dfs.clear()
    
    def get_days(self) -> np.ndarray:
        days = dates.get_day_count(self.curve_date, self.dates, dates.DayCountConvention.Actual)
        return days
    
    def sort(self):
        self.curve_points = sorted(self.curve_points, key=lambda cp: cp.date)
        self.dates, self.rates = list(map(list, zip(*[(cp.date, cp.rate) for cp in self.curve_points])))
        self.days = self.get_days()
        self.set_df_curve()
        self._str_dates_rates = str(self.dates)+str(self.rates)
        
    def delete_point(self, date: date):
        self.curve_points = list(filter(lambda cp: cp.date != date, self.curve_points))
        self.sort()       
        
    def add_point(self, curve_point: ZeroCouponCurvePoint):
        if curve_point.date < self.curve_date:
            raise ValueError(f"Cannot add point with date before curve date. Curve date: {self.curve_date}, point date: {curve_point.date}")
        
        self.delete_point(curve_point.date)
        self.curve_points.append(curve_point)
        self.sort()

    def get_df(self, date: date) -> float:
        return self.get_dfs([date])[0]
    
    def get_dfs(self, dates_t: list[date] | np.ndarray) -> np.ndarray:
        hashed_inputs = self._str_dates_rates + str(dates_t)
        if hashed_inputs in self._cashed_dfs:
            return self._cashed_dfs[hashed_inputs]
        tenors = dates.get_day_count(self.curve_date, dates_t, dates.DayCountConvention.Actual)
        min_tenor = min(self.get_days())
        max_tenor = max(self.get_days())
        tenors_smaller_than_min = tenors[tenors<min_tenor]
        tenors_greater_than_max = tenors[tenors>max_tenor]
        normal_tenors = tenors[(tenors >= min_tenor) & (tenors <= max_tenor)]
        small_tenors_amount = len(tenors_smaller_than_min)
        greater_tenors_amount = len(tenors_greater_than_max)
        normal_tenors_amount = len(normal_tenors)
        first_dfs = np.zeros(len(tenors))
        last_dfs = np.zeros(len(tenors))
        normal_dfs = np.zeros(len(tenors))
        for ix in range(small_tenors_amount):
            first_dfs[ix] = self.curve_points[0].rate.get_discount_factor(self.curve_date, dates_t[ix])
        for ix in range(greater_tenors_amount):
            index = small_tenors_amount + normal_tenors_amount + ix
            last_dfs[index] = self.curve_points[-1].rate.get_discount_factor(self.curve_date, dates_t[index])
        
        dfs = np.exp(np.interp(normal_tenors, self.days, np.log(self.dfs))) # Log-Linear interpolation of discount factors
        normal_dfs[small_tenors_amount:small_tenors_amount+normal_tenors_amount] = dfs

        total_dfs = first_dfs + normal_dfs + last_dfs
        self._cashed_dfs[hashed_inputs] = total_dfs
        return total_dfs
    
    def get_df_fwd(self, start_date: date, end_date: date) -> float:
        return self.get_df(end_date) / self.get_df(start_date)

    def get_dfs_fwds(self, start_dates: list[date] | np.ndarray, end_dates: list[date] | np.ndarray) -> np.ndarray:
        if len(start_dates) != len(end_dates):
            raise ValueError(f"Start and end dates must have the same length. Start dates: {start_dates}, end dates: {end_dates}")
        end_dfs = self.get_dfs(end_dates)
        start_dfs = self.get_dfs(start_dates)
        fwds = end_dfs/start_dfs
        return fwds
    
    def get_wf(self, date: date) -> float:
        return 1 / self.get_df(date)

    def get_wfs(self, dates: list[date] | np.ndarray) -> np.ndarray:
        return 1 / self.get_dfs(dates)
    
    def get_wf_fwd(self, start_date: date, end_date: date) -> float:
        return 1 / self.get_df_fwd(start_date, end_date)

    def get_wfs_fwds(self, start_dates: list[date] | np.ndarray, end_dates: list[date] | np.ndarray) -> np.ndarray:
        if len(start_dates) != len(end_dates):
            raise ValueError(f"Start and end dates must have the same length. Start dates: {start_dates}, end dates: {end_dates}")
        df_fwds = self.get_dfs_fwds(start_dates, end_dates)
        wfs_fwds = 1 / df_fwds
        return wfs_fwds
    
    def get_forward_rates(self, start_dates: list[date] | np.ndarray, end_dates: list[date] | np.ndarray, rate_convention: rates.RateConvention) -> list[rates.Rate] | rates.Rate:
        if len(start_dates) != len(end_dates):
            raise ValueError(f"Start and end dates must have the same length. Start dates: {start_dates}, end dates: {end_dates}")
        start_wfs = self.get_wfs(start_dates)
        end_wfs = self.get_wfs(end_dates)
        fwd_wfs = (end_wfs/start_wfs)
        fwd_rates = rates.Rate.get_rate_from_wf(fwd_wfs, start_dates, end_dates, rate_convention)
        return fwd_rates

    def get_forward_rates_values(self, start_dates: list[date] | np.ndarray, end_dates: list[date] | np.ndarray, rate_convention: rates.RateConvention=None) -> np.ndarray | float:
        if len(start_dates) != len(end_dates):
            raise ValueError(f"Start and end dates must have the same length. Start dates: {start_dates}, end dates: {end_dates}")        
        rates_obj = self.get_forward_rates(start_dates, end_dates, rate_convention)
        if isinstance(rates_obj, list):
            return np.array([r.rate_value for r in rates_obj])
        else: 
            return rates_obj.rate_value

    def get_zero_rates(self, rate_convention: rates.RateConvention=None) -> list[rates.Rate]:
        if rate_convention is None:
            return [cp.copy() for cp in self.curve_points]
        else:
            rates_obj = []
            for cp in self.curve_points:
                r = copy(cp.rate)
                if r.rate_convention == rate_convention:
                    rates_obj.append(r)
                else:
                    r.convert_rate_conventions(rate_convention)
                    rates_obj.append(r)
            return rates_obj
        
    def get_zero_rate(self, date: date,  rate_convention: rates.RateConvention=None) -> rates.Rate:
        if rate_convention is None:
            rate_convention = self.curve_points[0].rate.rate_convention
        
        df = self.get_df(date)
        r = rates.Rate.get_rate_from_df(df, self.curve_date, date, rate_convention)
        return r

    def get_zero_rates_values(self, rate_convention: rates.RateConvention=None) -> np.ndarray:
        rates_obj = self.get_zero_rates(rate_convention)
        return np.array([r.rate_value for r in rates_obj])
    
    def parallel_bump_rates_bps(self, bps: float):
        for zccp in self.curve_points:
            zccp.rate.rate_value += bps / 10_000
        self.sort()