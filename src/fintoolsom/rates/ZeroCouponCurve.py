from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import curve_fit
import numpy as np

from .. import rates
from .. import dates


class InterpolationType(Enum):
    Linear = 'linear'
    CubicSpline = 'cubic-spline'
    LogLinear = 'loglinear'
    NelsonSiegelSvensson = 'nelson-siegel-svensson'


@dataclass
class InterpolationModel(ABC):
    x_values: np.ndarray = field(init=False)
    y_values: np.ndarray = field(init=False)

    extrapolate: bool= field(default=True)
    
    _extrapolate_arg: str = field(init=False, default_factory=dict)
    def __post_init__(self):
        if self.extrapolate:
            self._extrapolate_args = {'fill_na': 'extrapolate'}

    @abstractmethod
    def get_y_value(self, x: float | np.ndarray) -> float | np.ndarray:
        pass

@dataclass
class LinearInterpolation(InterpolationModel):
    def __post_init__(self):
        super().__post_init__()
        self._interpolator = interp1d(self.x_values, self.y_values, **self._extrapolate_args)

    
    def get_y_value(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._interpolator(x)


@dataclass 
class CubicSplineInterpolation(InterpolationModel):
    def __post_init__(self):
        self._interpolator = CubicSpline(self.x_values, self.y_values, **self._extrapolate_args)
    
    def get_y_value(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._interpolator(x)

@dataclass
class LogLinearInterpolation(InterpolationModel):
    def __post_init__(self):
        super().__post_init__()
        self.y_values = np.log(self.y_values)
        self._interpolator = interp1d(self.x_values, self.y_values, **self._extrapolate_args)
    
    def get_y_value(self, x: float | np.ndarray) -> float | np.ndarray:
        return np.exp(self._interpolator(x))

@dataclass
class NelsonSiegelSvensson(InterpolationModel):
    b_0: float = field(init=False)
    b_1: float = field(init=False)
    b_2: float = field(init=False)
    lambda_0: float = field(init=False)
    def __post_init__(self):
        # Params are initialized in calibration
        self.calibrate()

    def calibrate(self):
        # Initial guess for parameters
        initial_guess = [0.05, 0, 0, 0.1]
        
        popt, _ = curve_fit(self._get_rate, self.x_values, self.y_values, p0=initial_guess)
        
        self.b_0, self.b_1, self.b_2, self.lambda_0 = popt

    def get_y_value(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._get_rate(x, self.b_0, self.b_1, self.b_2, self.lambda_0)

    def _get_rate(self, t: float | np.ndarray, *params) -> float | np.ndarray:
        b0, b1, b2, l = params
        lambda_t = l*t
        exp_lambda_t = np.exp(-lambda_t)
        term2 = b1 * (1 - exp_lambda_t) / lambda_t
        term3 = b2 * ((1 - exp_lambda_t) / lambda_t - exp_lambda_t)
        rate = b0 + term2 + term3
        return rate

@dataclass
class ZeroCouponCurvePoint:
    date: date
    rate: rates.Rate
    
    def copy(self):
        return ZeroCouponCurvePoint(self.date, self.rate.copy())
        
@dataclass
class ZeroCouponCurve:
    curve_date: date

    curve_points: list[ZeroCouponCurvePoint] = field(default=None)
    date_dfs: list[tuple[date, float]] = field(default=None)
    _rate_conv_for_dfs_init: rates.RateConvention = field(default=rates.RateConvention())
    rate_interpolation_type: InterpolationType = field(default=InterpolationType.CubicSpline)
    df_interpolation_type: InterpolationType = field(default=InterpolationType.LogLinear)
    _rate_interpolator: InterpolationModel = field(init=False)
    _df_interpolator: InterpolationModel = field(init=False)
    def __post_init__(self,):
        if self.curve_points is None:
            if self.date_dfs is not None:
                self.curve_points = [ZeroCouponCurvePoint(t_i, rates.Rate.get_rate_from_df(df, self.curve_date, t_i, self._rate_conv_for_dfs_init)) 
                                    for t_i, df in self.date_dfs]
            else:
                raise ValueError(f'If curve_points is not set, then date_dfs (list[tuples[date, float]]) must be set.')
        
        if self.df_interpolation_type==InterpolationType.Linear:
            raise ValueError(f'InterpolationType value Linear not admitted for df_interpolation_type.')

        self._interpolation_model_mapper: dict[InterpolationType, InterpolationModel] = {
            InterpolationType.Linear: LinearInterpolation,
            InterpolationType.CubicSpline: CubicSplineInterpolation,
            InterpolationType.LogLinear: LogLinearInterpolation,
            InterpolationType.NelsonSiegelSvensson: NelsonSiegelSvensson
        }
        self._rate_conv = self.curve_points[0].rate.rate_convention
        self._complete_curve()
        self._cashed_dfs: dict[str, np.ndarray] = {}

  
    def copy(self):
        return ZeroCouponCurve(self.curve_date, [zccp.copy() for zccp in self.curve_points])
    
    def _set_df_curve(self):
        self.wfs = np.array([cp.rate.get_wealth_factor(self.curve_date, cp.date) 
                             if cp.date >= self.curve_date else 1 
                             for cp in self.curve_points])
        self.dfs = 1 / self.wfs

    def forward_curve(self, forward_date: date, inplace: bool=False):
        if forward_date <= self.curve_date:
            raise ValueError(f'forward_date must be posterior to self.curve_date. forward_date received: {forward_date}, self.curve_date {self.curve_date}')
        date_dfs = [(zccp.date, self.get_df_fwd(forward_date, zccp.date)) for zccp in self.curve_points if zccp.date > forward_date]
        zcc = ZeroCouponCurve(forward_date, date_dfs=date_dfs, _rate_conv_for_dfs_init=self.curve_points[0].rate.rate_convention)
        if inplace:
            self.curve_points = zcc.curve_points
            self._complete_curve()
        else:
            return zcc
        
    def get_days(self) -> np.ndarray:
        days = dates.get_day_count(self.curve_date, self.dates, self._rate_conv.day_count_convention)
        return days
    
    def get_yfs(self) -> np.ndarray:
        days = self.get_days()
        return days / self._rate_conv.time_fraction_base
    
    def _complete_curve(self):
        self.curve_points = sorted(self.curve_points, key=lambda cp: cp.date)
        self.dates, self.rates_values = list(map(list, zip(*[(cp.date, cp.rate.rate_value) for cp in self.curve_points])))
        self.days = self.get_days()
        self.yfs = self.get_yfs()
        self._set_df_curve()
        self._str_dates_rates = str(self.dates)+str(self.rates_values)
        rate_interpolator_cls = self._interpolation_model_mapper[self.rate_interpolation_type]
        df_interpolator_cls = self._interpolation_model_mapper[self.df_interpolation_type]
        self._rate_interpolator = rate_interpolator_cls(self.yfs, self.rates_values)
        self._df_interpolator = df_interpolator_cls(self.yfs, self.dfs)        
        
    def delete_point(self, date: date):
        self.curve_points = list(filter(lambda cp: cp.date != date, self.curve_points))
        self._complete_curve()       
        
    def add_point(self, curve_point: ZeroCouponCurvePoint):
        if curve_point.date < self.curve_date:
            raise ValueError(f"Cannot add point with date before curve date. Curve date: {self.curve_date}, point date: {curve_point.date}")
        
        self.delete_point(curve_point.date)
        self.curve_points.append(curve_point)
        self._complete_curve()

    def get_df(self, date: date) -> float:
        return self.get_dfs([date])[0]
    
    def get_dfs(self, dates_t: list[date] | np.ndarray) -> np.ndarray:
        hashed_inputs = self._str_dates_rates + str(dates_t)
        if hashed_inputs in self._cashed_dfs:
            return self._cashed_dfs[hashed_inputs]
        

        if self._df_interpolator.extrapolate:
            yfs = dates.get_time_fraction(self.curve_date, dates_t, self._rate_conv.day_count_convention, base_convention=self._rate_conv.time_fraction_base)
            if isinstance(self._df_interpolator, NelsonSiegelSvensson):
                interpolated_rates_values = self._rate_interpolator.get_y_value(yfs)
                dfs = np.array([rates.Rate(self._rate_conv, r_i).get_discount_factor(self.curve_date, t_i) for t_i, r_i in zip(dates_t, interpolated_rates_values)])
            else:            
                dfs = self._df_interpolator.get_y_value(yfs)
        else:
            tenors = dates.get_day_count(self.curve_date, dates_t, self._rate_conv.day_count_convention)
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
            
            normal_yfs = normal_tenors / self._rate_conv.time_fraction_base
            if isinstance(self._df_interpolator, NelsonSiegelSvensson):
                interpolated_rates_values = self._rate_interpolator.get_y_value(normal_yfs)
                dfs = np.array([rates.Rate(self._rate_conv, r_i).get_discount_factor(self.curve_date, t_i) for t_i, r_i in zip(dates_t, interpolated_rates_values)])
            else:            
                dfs = self._df_interpolator.get_y_value(normal_yfs)
            normal_dfs[small_tenors_amount:small_tenors_amount+normal_tenors_amount] = dfs
            dfs = first_dfs + normal_dfs + last_dfs
        self._cashed_dfs[hashed_inputs] = dfs
        return dfs
    
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
    
    def get_forward_rates(self, start_dates: list[date] | np.ndarray, end_dates: list[date] | np.ndarray, rate_convention: rates.RateConvention) -> list[float]:
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

    def get_zero_rates(self, rate_convention: rates.RateConvention=None) -> list[rates.Rate]:
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
    
    def parallel_shift(self, shift_in_bps: float):
        cps = []
        for zccp in self.curve_points:
            cp_date = zccp.date
            cp_rate = zccp.rate.copy()
            cp_rate.rate_value += shift_in_bps / 10_000
            cps.append(ZeroCouponCurvePoint(cp_date, cp_rate))
        self.curve_points = cps
        self._complete_curve()