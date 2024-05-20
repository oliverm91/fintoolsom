from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
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

    interpolation_method: InterpolationMethod

    log_moneyness_df: pd.DataFrame = field(repr=False, init=False)
    delta_puts: np.ndarray = field(init=False)
    days: np.ndarray = field(init=False)
    def __post_init__(self):
        self.delta_puts = self.vol_surface_df.columns.to_numpy()
        self.days = self.vol_surface_df.index.to_numpy()

        log_moneyness_np = np.zeros((len(self.days), len(self.vol_surface_df.columns)))
        for row in range(log_moneyness_np.shape[0]):
            for column in range(log_moneyness_np.shape[1]):
                days = self.days[row]
                delta = self.delta_puts[column]
                
                maturity = self.domestic_curve.curve_date + timedelta(days=days)
                df_r = self.domestic_curve.get_df(maturity)
                df_q = self.foreign_curve.get_df(maturity)
                fwd_price = self.spot * df_q / df_r

                volatility = self.log_moneyness_df.values[row, column]
                k = Put.get_strike_from_delta(delta, self.spot, volatility, self.domestic_curve, self.foreign_curve, maturity)
                log_moneyness = np.log(k / fwd_price)
                log_moneyness_np[row, column] = log_moneyness
        
        self.log_moneyness_df = pd.DataFrame(log_moneyness_np, columns=self.delta_puts, index=self.vol_surface_df.index)

    @abstractmethod
    def interpolate_surface(self, log_moneyness: float, days: int) -> float:
        pass

@dataclass
class DoubleInterpolationModel(InterpolationModel, ABC):
    def __post_init__(self):
        if self.interpolation_method not in [InterpolationMethod.DoubleLinear, InterpolationMethod.DoubleCubicSpline]:
            raise ValueError(f'DoubleInterpolationModel can not implement InterpolationMethod {self.interpolation_method}.')

    def interpolate_surface(self, log_moneyness: float, days: int | np.ndarray) -> float | np.ndarray:
        smile = self.get_smile_t(days) # First interpolation
        smile_moneynesses = self.get_smile_log_moneynesses(days)
        vol = self._second_interp(log_moneyness, smile_moneynesses, smile) # Second interpolation
        return vol
    
    @abstractmethod
    def _second_interp(log_moneyness: float, smile_moneynesses: np.ndarray, smile: np.ndarray) -> float:
        pass

    def get_smile_log_moneynesses(self, days: int) -> np.ndarray:
        df: pd.DataFrame =self.log_moneyness_df.reindex(self.log_moneyness_df.index.tolist()+[days]).interpolate(method=self.interpolation_method.value)
        return df.loc[days].to_numpy()

    def get_smile_t(self, days: int) -> np.ndarray:
        df: pd.DataFrame =self.vol_surface_df.reindex(self.vol_surface_df.index.tolist()+[days]).interpolate(method=self.interpolation_method.value)
        return df.loc[days].to_numpy()

class DoubleInterpolationLinear(DoubleInterpolationModel):
    def _second_interp(self, log_moneyness: float, smile_moneynesses: np.ndarray, smile: np.ndarray) -> float:
        vol = np.interp(log_moneyness, smile_moneynesses, smile)
        return vol
    
class DoubleInterpolationCubicSpline(DoubleInterpolationModel):
    def _second_interp(self, log_moneyness: float, smile_moneynesses: np.ndarray, smile: np.ndarray) -> float:
        cs = CubicSpline(smile_moneynesses, smile)
        vol = cs(log_moneyness)
        return vol


@dataclass
class eSSVI(InterpolationModel):
    theta_0: float
    theta_1: float

    rho_0: float
    rho_1: float

    phi_0: float
    phi_1: float

    def __post_init__(self):
        self.calibrate()

    def exponential_formula(self, base_multiplier: float, exponent_multiplier: float, days: int | np.ndarray) -> float | np.ndarray:
        return base_multiplier * np.exp(exponent_multiplier * days)
    
    def interpolate_surface(self, log_moneyness: float | np.ndarray, days: int | np.ndarray) -> float | np.ndarray:
        total_variance = self._get_total_variance(log_moneyness, days, self.theta_0, self.theta_1, self.rho_0, self.rho_1, self.phi_0, self.phi_1)
        vol = np.sqrt(total_variance / days)
        return vol
    
    def _get_total_variance(self, log_moneyness: float | np.ndarray, days: int | np.ndarray, *params, validate_shapes: bool=True) -> float | np.ndarray:
        if validate_shapes:
            # Ensure log_moneyness and days are numpy arrays for easier manipulation
            if isinstance(log_moneyness, float):
                log_moneyness = np.array([log_moneyness])
            if isinstance(days, float):
                days = np.array([days])
            
            # Check if both log_moneyness and days are arrays
            if isinstance(log_moneyness, np.ndarray) and isinstance(days, np.ndarray):
                if log_moneyness.ndim == 1 and days.ndim == 1:
                    # Both are simple arrays, no need to reshape
                    pass
                elif log_moneyness.ndim == 1 and days.ndim == 2:
                    if days.shape[0] == len(log_moneyness) and days.shape[1] == 1:
                        # log_moneyness is an array and days is a matrix of size len(log_moneyness)x1
                        pass
                    else:
                        raise ValueError("When both log_moneyness and days are arrays, days must be of shape (len(log_moneyness), 1).")
                else:
                    raise ValueError("Invalid dimensions for log_moneyness and days.")

        theta_0, theta_1, rho_0, rho_1, phi_0, phi_1 = params
        theta = self.exponential_formula(theta_0, theta_1, days)
        rho = self.exponential_formula(rho_0, rho_1, days)
        phi = self.exponential_formula(phi_0, phi_1, days)

        phi_k = phi * log_moneyness
        total_variance = (theta / 2) * (1 + rho * phi_k + np.sqrt((phi_k + rho)**2 + 1 - rho**2))
        
        return total_variance
    
    def calibrate(self):
        days = self.vol_surface_df.index.to_numpy()[:, None]
        mkt_vol_surface_matrix = self.vol_surface_df.values
        mkt_total_variances_matrix = days * mkt_vol_surface_matrix**2

        #Initial guesses for theta, rho and phi. Initial guess for rho_0 is positive as is thought to be used in FX markets. Change to -0.4 for Equity markets.
        initial_params = [
            mkt_total_variances_matrix.mean(), 0,
            0.4, 0, 
            0.3, 0
        ]
        bounds = [
            (0, None), (None, None),
            (-1, 1), (None, None),
            (0, None), (None, None)
        ]

        def objective_func(params):
            model_total_variances = self._get_total_variance(self.log_moneyness_df.values, days, *params, validate_shapes=False)
            return np.sum((model_total_variances - mkt_total_variances_matrix)**2)

        result = minimize(objective_func, initial_params, bounds=bounds)

        if result.success:
            fitted_params = result.x
            print("Fitted parameters:", fitted_params)
            self.theta_0, self.theta_1, self.rho_0, self.rho_1, self.phi_0, self.phi_1 = fitted_params
        else:
            raise ValueError("Fitting failed")


@dataclass
class VolatilitySurface:
    vol_surface_df: pd.DataFrame = field(repr=False)
    log_moneyness_columns: list[float] = field(repr=True)

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

        if len(self.log_moneyness_columns)!=len(self.vol_surface_df.columns):
            raise ValueError(f'Length of log_moneyness_columns must match the amount of columns in vol_surface_df.')
        for log_moneyness in self.log_moneyness_columns:
            if not isinstance(log_moneyness, float):
                raise TypeError(f'All log_moneyness_columns items must be of type float.')

        self.vol_surface_np = self.vol_surface_df.values
        self.log_moneyness_columns = np.array(self.log_moneyness_columns)

        if self.interpolation_method==InterpolationMethod.eSSVI:
            self._interpolation_model = eSSVI(self.vol_surface_df, self.log_moneyness_columns, self.interpolation_method)
        elif self.interpolation_method==InterpolationMethod.DoubleLinear:
            self._interpolation_model = DoubleInterpolationLinear(self.vol_surface_df, self.log_moneyness_columns, self.interpolation_method)
        elif self.interpolation_method==InterpolationMethod.DoubleCubicSpline:
            self._interpolation_model = DoubleInterpolationCubicSpline(self.vol_surface_df, self.log_moneyness_columns, self.interpolation_method)
        else:
            raise NotImplementedError(f'Interpolation method: {self.interpolation_method} not implemented.')

    def get_volatility(self, log_moneyness: float, days: int) -> float:
        vol = self._interpolation_model.interpolate_surface(log_moneyness, days)
        return vol