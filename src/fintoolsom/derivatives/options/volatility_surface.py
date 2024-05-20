from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
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
    

@dataclass
class eSSVI(InterpolationModel):
    theta_0: float = field(init=False)
    theta_1: float = field(init=False)

    rho_0: float = field(init=False)
    rho_1: float = field(init=False)

    phi_0: float = field(init=False)
    phi_1: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.calibrate()

    def exponential_formula(self, base_multiplier: float, exponent_multiplier: float, t: float | np.ndarray) -> float | np.ndarray:
        return base_multiplier * np.exp(exponent_multiplier * t)
    
    def interpolate_surface(self, log_moneyness: float | np.ndarray, days: int | np.ndarray) -> float | np.ndarray:
        total_variance = self._get_total_variance(log_moneyness, days, self.theta_0, self.theta_1, self.rho_0, self.rho_1, self.phi_0, self.phi_1)
        vol = np.sqrt(total_variance / (days/360))
        return vol
    
    def _get_total_variance(self, log_moneyness: float | np.ndarray, days: int | np.ndarray, *params, validate_shapes: bool=True) -> float | np.ndarray:
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

        theta_0, theta_1, rho_0, rho_1, phi_0, phi_1 = params
        theta = self.exponential_formula(theta_0, theta_1, t)
        rho = self.exponential_formula(rho_0, rho_1, t)
        phi = self.exponential_formula(phi_0, phi_1, t)

        phi_k = phi * log_moneyness
        total_variance = (theta / 2) * (1 + rho * phi_k + np.sqrt((phi_k + rho)**2 + 1 - rho**2))
        
        return total_variance
    
    def calibrate(self):
        days = self.vol_surface_df.index.to_numpy()[:, None]
        mkt_vol_surface_matrix = self.vol_surface_df.values
        mkt_total_variances_matrix = (days / 360) * mkt_vol_surface_matrix**2

        #Initial guesses for theta, rho and phi. Initial guess for rho_0 is positive as is thought to be used in FX markets. Change to -0.4 for Equity markets.
        initial_params = [
            mkt_total_variances_matrix.mean(), 0,
            0.4, 0, 
            0.3, 0
        ]
        bounds = [
            (0, 1), (-1, 1),
            (-1, 1), (-1, 1),
            (0, 1), (-1, 1)
        ]

        log_moneyness_np = self.log_moneyness_df.values
        def objective_func(params):
            model_total_variances = self._get_total_variance(log_moneyness_np, days, *params, validate_shapes=False)
            return np.sum((model_total_variances - mkt_total_variances_matrix)**2)

        result = minimize(objective_func, initial_params, bounds=bounds, method='trust-constr')

        if result.success:
            fitted_params = result.x
            print("Fitted parameters:", fitted_params)
            self.theta_0, self.theta_1, self.rho_0, self.rho_1, self.phi_0, self.phi_1 = fitted_params
        else:
            raise ValueError("Fitting failed")


@dataclass
class VolatilitySurface:
    vol_surface_df: pd.DataFrame = field(repr=False)
    spot: float
    domestic_curve: ZeroCouponCurve = field(repr=False)
    foreign_curve: ZeroCouponCurve = field(repr=False)
    
    interpolation_method: InterpolationMethod = field(default=InterpolationMethod.eSSVI, repr=True)
    vol_surface_np: np.ndarray = field(init=False)

    _interpolation_model: InterpolationModel = field(init=False)
    def __post_init__(self):
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
            self._interpolation_model = eSSVI(self.vol_surface_df, self.spot, self.domestic_curve, self.foreign_curve)
        elif self.interpolation_method==InterpolationMethod.DoubleLinear:
            self._interpolation_model = DoubleInterpolationLinear(self.vol_surface_df, self.spot, self.domestic_curve, self.foreign_curve)
        elif self.interpolation_method==InterpolationMethod.DoubleCubicSpline:
            self._interpolation_model = DoubleInterpolationCubicSpline(self.vol_surface_df, self.spot, self.domestic_curve, self.foreign_curve)
        else:
            raise NotImplementedError(f'Interpolation method: {self.interpolation_method} not implemented.')

    def get_volatility(self, log_moneyness: float, days: int) -> float:
        vol = self._interpolation_model.interpolate_surface(log_moneyness, days)
        return vol