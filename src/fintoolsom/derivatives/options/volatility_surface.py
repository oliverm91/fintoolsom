from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize

from fintoolsom.rates import ZeroCouponCurve
from fintoolsom.derivatives.options.options import Put


class InterpolationMethod(Enum):
    DoubleLinear = 'linear'
    DoubleCubicSpline = 'cubicspline'
    eSSVI = 'essvi'


@dataclass
class InterpolationModel(ABC):
    vol_surface_df: pd.DataFrame = field(repr=False)
    spot: float
    domestic_curve: ZeroCouponCurve
    foreign_curve: ZeroCouponCurve

    log_moneyness_df: pd.DataFrame = field(repr=False, init=False)
    days: np.ndarray = field(init=False)
    def __post_init__(self):
        delta_puts = self.vol_surface_df.columns.to_numpy()
        self.days = self.vol_surface_df.index.to_numpy()

        log_moneyness_np = np.zeros((len(self.days), len(self.vol_surface_df.columns)))
        vol_surface_df_np = self.vol_surface_df.values
        for row in range(log_moneyness_np.shape[0]):
            for column in range(log_moneyness_np.shape[1]):
                days = self.days[row]
                delta = delta_puts[column]
                if delta=='atm':
                    log_moneyness=0
                else:
                    delta = -int(delta)/100
                    maturity = self.domestic_curve.curve_date + timedelta(days=int(days))
                    df_r = self.domestic_curve.get_df(maturity)
                    df_q = self.foreign_curve.get_df(maturity)
                    fwd_price = self.spot * df_q / df_r

                    volatility = vol_surface_df_np[row, column]
                    k = Put.get_strike_from_delta(delta, self.spot, volatility, self.domestic_curve, self.foreign_curve, maturity)
                    log_moneyness = np.log(k / fwd_price)
                log_moneyness_np[row, column] = log_moneyness
        
        self.log_moneyness_df = pd.DataFrame(log_moneyness_np, columns=self.vol_surface_df.columns, index=self.vol_surface_df.index)

    @abstractmethod
    def interpolate_surface(self, log_moneyness: float, days: int) -> float:
        pass

@dataclass
class DoubleInterpolationModel(InterpolationModel, ABC):
    _interpolator1d: Any = field(init=False)
    def __post_init__(self):
        super().__post_init__()
        
    def interpolate_surface(self, log_moneyness: float | np.ndarray, days: int) -> float | np.ndarray:
        smile = self.get_smile_t(days) # First interpolation
        smile_moneynesses = self.get_smile_log_moneynesses(days)
        vol = self._interpolator1d(smile_moneynesses, smile, axis=0)(log_moneyness)
        return vol

    def get_smile_log_moneynesses(self, days: int) -> np.ndarray:
        return self._interpolator1d(self.log_moneyness_df.index, self.log_moneyness_df.values, axis=0)(days)

    def get_smile_t(self, days: int) -> np.ndarray:
        return self._interpolator1d(self.vol_surface_df.index, self.vol_surface_df.values, axis=0)(days)

class DoubleInterpolationLinear(DoubleInterpolationModel):
    def __post_init__(self):
        super().__post_init__()
        self._interpolator1d = interp1d

class DoubleInterpolationCubicSpline(DoubleInterpolationModel):
    def __post_init__(self):
        super().__post_init__()
        self._interpolator1d = CubicSpline
    

class eSSVI(InterpolationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.theta_params = []
        self.rho_params = []
        self.phi_params = []
        self._all_params = [self.theta_params, self.rho_params, self.phi_params]
        self.do_kwargs(**kwargs)
        
        self.calibrate()

    def do_kwargs(self, **kwargs):
        self.parameters_type = kwargs.get('parameter_type', 'parametric')
        if self.parameters_type=='functional_form':
            self._function_type = kwargs.get('function_type', 'exponential')
            if self._function_type=='exponential':
                self.parameter_formula = self.exponential_formula
                for i in range(3):
                    for _ in range(2):
                        self._all_params[i].append(None)
            elif self._function_type=='polynomial':
                self.parameter_formula = self.polynomial_formula
                self.polynomial_order = kwargs.get('polynomial_order', 2)
                for i in range(3):
                    for _ in range(1 + self.polynomial_order):
                        self._all_params[i].append(None)
            else:
                raise ValueError(f'function_type value {self._function_type} not valid.')
        elif self.parameters_type=='parametric':
            self.parameter_formula = self.parametric_formula
            parameter_interpolation_method = kwargs.get('parameters_interpolation_method', 'linear')
            if parameter_interpolation_method=='linear':
                self._parameter_interpolator = interp1d
            elif parameter_interpolation_method=='cubic-spline':
                self._parameter_interpolator = CubicSpline
            else:
                raise ValueError(f'parameters_interpolation_method value {parameter_interpolation_method} not valid. Try "linear" or "cubic-spline".')
            for i in range(3):
                for _ in range(len(self.vol_surface_df.index)):
                    self._all_params[i].append(None)
        else:
            raise ValueError(f'parameter_type value {self.parameters_type} not valid.')

    def exponential_formula(self, t: float | np.ndarray, *params) -> float | np.ndarray:
        return params[0] * np.exp(params[1] * t)
    
    def polynomial_formula(self, t: float | np.ndarray, *params) -> float | np.ndarray:
        ixs = np.arange(len(params))
        return np.sum(np.array(params)[ixs] * t.reshape(-1,1) ** ixs, axis=1)
    
    def parametric_formula(self, t: float | np.ndarray, *params) -> float | np.ndarray:
        return self._parameter_interpolator(self.vol_surface_df.index / 360, params)(t)
    
    def interpolate_surface(self, log_moneyness: float | np.ndarray, days: int | np.ndarray) -> float | np.ndarray:
        flatten_params = list(itertools.chain(*self._all_params))
        total_variance = self._get_total_variance(log_moneyness, days, *flatten_params)
        vol = np.sqrt(total_variance / (days/360))
        return vol
    
    def _get_total_variance(self, log_moneyness: float | np.ndarray, days: int | np.ndarray, *flatten_params: list[float], validate_shapes: bool=True) -> float | np.ndarray:
        t = days / 360
        if validate_shapes:
            # Ensure log_moneyness and days are numpy arrays for easier manipulation
            if isinstance(log_moneyness, float):
                log_moneyness = np.array([log_moneyness])
            if isinstance(t, float):
                t = np.array([t])
            
            # Check if both log_moneyness and days are arrays
            if isinstance(log_moneyness, np.ndarray) and isinstance(t, np.ndarray):
                if log_moneyness.ndim == 1 and t.ndim == 1:
                    # Both are simple arrays, no need to reshape
                    pass
                elif log_moneyness.ndim == 1 and t.ndim == 2:
                    if t.shape[0] == len(log_moneyness) and t.shape[1] == 1:
                        # log_moneyness is an array and days is a matrix of size len(log_moneyness)x1
                        pass
                    else:
                        raise ValueError("When both log_moneyness and days are arrays, days must be of shape (len(log_moneyness), 1).")
                else:
                    raise ValueError("Invalid dimensions for log_moneyness and days.")

        params_amount = int(len(flatten_params) / 3)

        theta = self.parameter_formula(t, *flatten_params[0:params_amount])
        rho = self.parameter_formula(t, *flatten_params[params_amount:2*params_amount])
        phi = self.parameter_formula(t, *flatten_params[2*params_amount:3*params_amount])

        phi_k = phi * log_moneyness
        total_variance = (theta / 2) * (1 + rho * phi_k + np.sqrt((phi_k + rho)**2 + 1 - rho**2))
        
        return total_variance
    
    def calibrate(self):
        days = self.vol_surface_df.index.to_numpy()[:, None]
        mkt_vol_surface_matrix = self.vol_surface_df.values
        mkt_total_variances_matrix = (days / 360) * mkt_vol_surface_matrix**2

        #Initial guesses for theta, rho and phi. Independently of parameter form. theta represents level, rho correlation (positive for FX) and phi skewness (slightly positive for FX)
        initial_params = self._all_params.copy() # Until now params are all None
        if self.parameters_type=='parametric':
            average_vols_per_tenor = np.mean(mkt_total_variances_matrix, axis=1)
            for i in range(mkt_total_variances_matrix.shape[0]):
                initial_params[0][i] = average_vols_per_tenor[i] # Thetas
                initial_params[1][i] = 0.3 # rhos
                initial_params[2][i] = 0.3 # phis
        else:
            if self._function_type=='exponential':
                initial_params = [
                    [mkt_total_variances_matrix.mean(), 0],
                    [0.3, 0], 
                    [0.3, 0]
                ]
            if self._function_type=='polynomial':
                initial_params[0][0] = mkt_total_variances_matrix.mean() # Thetas
                initial_params[1][0] = 0.3 # rhos
                initial_params[2][0] = 0.3 # phis
                for i in range(1, 1 + self.polynomial_order):
                    initial_params[0][i] = 0
                    initial_params[1][i] = 0
                    initial_params[2][i] = 0

        flatten_initial_params = list(itertools.chain(*initial_params))
        bounds = [(-1, 1)] * len(flatten_initial_params)
        log_moneyness_np = self.log_moneyness_df.values        
        def objective_func(flatten_params):
            model_total_variances = self._get_total_variance(log_moneyness_np, days, *flatten_params, validate_shapes=False)
            return np.sum((model_total_variances - mkt_total_variances_matrix)**2)

        initial_model_total_variances = self._get_total_variance(log_moneyness_np, days, *flatten_initial_params, validate_shapes=False)
        print(f'\tStarting calibration... Initial error: {np.sum((initial_model_total_variances - mkt_total_variances_matrix)**2)}')
        result = minimize(objective_func, flatten_initial_params, bounds=bounds, method='trust-constr')

        if result.success:
            print(f'\tCalibration worked. Error achieved: {result.fun}\n')
            fitted_params = result.x
            params_amount = int(len(fitted_params) / 3)
            #print("Fitted parameters:", fitted_params)
            self._all_params = [fitted_params[0:params_amount], fitted_params[params_amount:2*params_amount], fitted_params[2*params_amount:3*params_amount]]
        else:
            raise ValueError("Fitting failed")


class VolatilitySurface:
    def __init__(self, vol_surface_df: pd.DataFrame, spot: float, domestic_curve: ZeroCouponCurve, foreign_curve: ZeroCouponCurve,
                        interpolation_method: InterpolationMethod=InterpolationMethod.eSSVI, **kwargs):
        self.vol_surface_df = vol_surface_df
        self.spot = spot
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.interpolation_method = interpolation_method
        for t in self.vol_surface_df.index:
            if not isinstance(t, int):
                raise TypeError(f'All indexes in vol_surface_df must be of type int, representing days to maturity.')
        self.vol_surface_df.sort_index(inplace=True, ascending=True)

        for col in self.vol_surface_df.columns:
            for t in self.vol_surface_df.index:
                if not isinstance(self.vol_surface_df[col][t], float):
                    raise TypeError(f'All values in vol_surface_df must be of type float. Ex: 14% is 0.14')

        self.vol_surface_np = self.vol_surface_df.values
        
        if self.interpolation_method==InterpolationMethod.eSSVI:
            self._interpolation_model = eSSVI(self.vol_surface_df, self.spot, self.domestic_curve, self.foreign_curve, **kwargs)
        elif self.interpolation_method==InterpolationMethod.DoubleLinear:
            self._interpolation_model = DoubleInterpolationLinear(self.vol_surface_df, self.spot, self.domestic_curve, self.foreign_curve)
        elif self.interpolation_method==InterpolationMethod.DoubleCubicSpline:
            self._interpolation_model = DoubleInterpolationCubicSpline(self.vol_surface_df, self.spot, self.domestic_curve, self.foreign_curve)
        else:
            raise NotImplementedError(f'Interpolation method: {self.interpolation_method} not implemented.')

    def get_volatility(self, log_moneyness: float, days: int) -> float:
        vol = self._interpolation_model.interpolate_surface(log_moneyness, days)
        return vol