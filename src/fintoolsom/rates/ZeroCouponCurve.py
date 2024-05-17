import numpy as np
from typing import List, Self, Union
import mathsom.interpolations as interps
from datetime import date
from dataclasses import dataclass

from .. import rates
from .. import dates

@dataclass
class ZeroCouponCurvePoint:
    date: date
    rate: rates.Rate
    
    def copy(self) -> Self:
        return ZeroCouponCurvePoint(self.date, self.rate.copy())
        

class ZeroCouponCurve:
    def __init__(self, curve_date: date, curve_points: List[ZeroCouponCurvePoint]):
        self.curve_date = curve_date
        self.curve_points = curve_points
        self.sort()
        self._cashed_dfs: [str, np.ndarray] = {}
  
    def copy(self) -> Self:
        return ZeroCouponCurve(self.curve_date, self.curve_points)
    
    def set_df_curve(self):
        self.wfs = np.array([cp.rate.get_wealth_factor(self.curve_date, cp.date) 
                             if cp.date >= self.curve_date else 1 
                             for cp in self.curve_points])
        self.dfs = 1 / self.wfs
        
    def set_tenors(self):
        self.tenors = self.get_tenors()
    
    def get_tenors(self) -> np.ndarray:
        tenors = dates.get_day_count(self.curve_date, self.dates, dates.DayCountConvention.Actual)
        return tenors
    
    def sort(self):
        self.curve_points = sorted(self.curve_points, key=lambda cp: cp.date)
        self.dates, self.rates = list(map(list, zip(*[(cp.date, cp.rate) for cp in self.curve_points])))
        self.tenors = self.get_tenors()
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
        min_tenor = min(self.get_tenors())
        max_tenor = max(self.get_tenors())
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
        
        dfs = interps.interpolate(normal_tenors, self.tenors, self.dfs, interps.InterpolationMethod.LOGLINEAR)
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
    
    def get_forward_rates(self, start_dates: list[date] | np.ndarray, end_dates: list[date] | np.ndarray, rate_convention: rates.RateConvention) -> List[float]:
        if len(start_dates) != len(end_dates):
            raise ValueError(f"Start and end dates must have the same length. Start dates: {start_dates}, end dates: {end_dates}")
        start_wfs = self.get_wfs(start_dates)
        end_wfs = self.get_wfs(end_dates)
        fwd_wfs = (end_wfs/start_wfs)
        fwd_rates = rates.Rate.get_rate_from_wf(fwd_wfs, start_dates, end_dates, rate_convention)
        return fwd_rates

    def get_forward_rates_values(self, start_dates: list[date] | np.ndarray, end_dates: list[date] | np.ndarray, rate_convention: rates.RateConvention=None) -> np.ndarray:
        if len(start_dates) != len(end_dates):
            raise ValueError(f"Start and end dates must have the same length. Start dates: {start_dates}, end dates: {end_dates}")        
        rates_obj = self.get_forward_rates(start_dates, end_dates, rate_convention)
        return np.array([r.rate_value for r in rates_obj])

    def get_zero_rates(self, rate_convention: rates.RateConvention=None) -> List[rates.Rate]:
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

    def get_zero_rates_values(self, rate_convention: rates.RateConvention=None) -> np.ndarray:
        rates_obj = self.get_zero_rates(rate_convention)
        return np.array([r.rate_value for r in rates_obj])