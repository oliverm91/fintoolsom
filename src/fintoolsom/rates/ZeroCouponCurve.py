import numpy as np
from collections.abc import Sequence
from typing import Union
import mathsom.interpolations as interps

from .. import rates
from .. import dates


class ZeroCouponCurvePoint:
    def __init__(self, date, rate: rates.Rate):
        self.date = date
        self.rate = rate
        
        
class ZeroCouponCurve:
    def __init__(self, curve_date, curve_points: Sequence):
        self.curve_date = curve_date
        self.curve_points = curve_points
        self.sort()
  
    def copy(self):
        return ZeroCouponCurve(self.curve_date, self.curve_points)
    
    def set_df_curve(self):
        self.wfs = np.array([cp.rate.get_wealth_factor(self.curve_date, cp.date) 
                             if cp.date >= self.curve_date else 1 
                             for cp in self.curve_points])
        self.dfs = 1/self.wfs
        
    def set_tenors(self):
        self.tenors = self.get_tenors()
    
    def get_tenors(self):
        points_dates = [cp.date for cp in self.curve_points]
        tenors = dates.get_day_count(self.curve_date, points_dates, dates.DayCountConvention.Actual)
        return tenors
    
    def sort(self):
        self.curve_points = sorted(self.curve_points, key=lambda cp: cp.date)
        self.dates, self.rates = list(map(list, zip(*[(cp.date, cp.rate) for cp in self.curve_points])))
        self.tenors = self.get_tenors()
        self.set_df_curve()
        
    def delete_point(self, date):
        self.curve_points = [cp for cp in self.curve_points if cp.date != date]
        self.dates, self.rates = list(map(list, zip(*[(cp.date, cp.rate) for cp in self.curve_points])))
        self.sort()       
        
    def add_point(self, curve_point: ZeroCouponCurvePoint):
        if curve_point.date <= self.curve_date:
            raise(f'Curve date is {self.curve_date.strftime("%Y-%m-%d")}. Date {curve_point.date.strftime("%Y-%m-%d")} was passed to add.')
        
        self.delete_point(curve_point.date)
        self.curve_points.append(curve_point)
        self.sort()
    
    def get_dfs(self, dates: Union(Sequence, np.ndarray)):
        tenors = dates.get_day_count(self.curve_date, dates, dates.DayCountConvention.Actual)
        future_tenors_mask = tenors > 0
        dfs = interps.interpolate(tenors, self.tenors, self.dfs, interps.InterpolationMethod.LOGLINEAR)
        dfs = dfs*future_tenors_mask + 1 * np.invert(future_tenors_mask)
        return dfs
    
    def get_wfs(self, dates):
        return 1 / self.get_dfs(dates)
    
    def get_forward_rates(self, start_dates: Union(Sequence, np.ndarray), end_dates: Union(Sequence, np.ndarray), rate_convention: rates.RateConvention):
        start_wfs = self.get_wfs(start_dates)
        end_wfs = self.get_wfs(end_dates)
        fwd_wfs = (end_wfs/start_wfs)
        actual_tenors = dates.get_day_count(start_dates, end_dates, dates.DayCountConvention.Actual)
        tenors_wfs = zip(actual_tenors, fwd_wfs)
        lin_act_360_convention = rates.RateConvention(rates.InterestConvention.Linear, dates.DayCountConvention.Actual, 360)
        lin_act_360_frs = [rates.Rate(lin_act_360_convention, (fwd_wf - 1)/(tenor / 360.0)) for tenor, fwd_wf in tenors_wfs]
        la360fw_sd_ed = zip(lin_act_360_frs, start_dates, end_dates)
        forward_rates = [fwd_rate.convert_rate_conventions(rate_convention, start_date, end_date) for fwd_rate, start_date, end_date in la360fw_sd_ed]
        return forward_rates
    
    def get_zero_rates(self, rate_convention: rates.RateConvention):
        r = rates.Rate(rate_convention, 0)
        rates = r.get_rate_value_from_wf(self.wfs, self.curve_date, self.dates)
        return rates