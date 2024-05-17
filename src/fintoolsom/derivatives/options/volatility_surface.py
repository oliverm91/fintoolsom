from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


class InterpolationMethod(Enum):
    Linear = 'linear'
    CubicSpline = 'spline'


@dataclass
class VolatilitySurfaceStrikeColumn:
    t: date
    strike: float
    vol_surface_column: pd.Series

    def __post_init__(self):
        for t in self.vol_surface_column.index:
            if not isinstance(t, date):
                raise TypeError(f'All indexes in vol_surface_column must be of type date')
            
        self.vol_surface_column.index = [(t_i - self.t).days for t_i in self.vol_surface_column.index]
        self.vol_surface_column.sort_index(ascending=True)

    def get_volatility(self, maturity: date, interpolation_type: InterpolationMethod) -> float:
        days = (maturity - self.t).days
        x, y = self.vol_surface_column.index, self.vol_surface_column.values
        if interpolation_type==InterpolationMethod.Linear:
            vol = np.interp(days, x, y)
        elif interpolation_type==InterpolationMethod.CubicSpline:
            cs = CubicSpline(x, y)
            vol = cs(days)

        return vol
    

@dataclass
class VolatilitySurface:
    t: date
    vol_surface_strike_columns: list[VolatilitySurfaceStrikeColumn]

    strikes_cols: list[float] = field(init=False)
    def __post_init__(self):
        if not all(vssc.vol_surface_column.index.tolist()==self.vol_surface_strike_columns[0].vol_surface_column.index.tolist() 
                   for vssc in self.vol_surface_strike_columns):
            raise ValueError(f'All maturities index in vol_surface_strike_columns must be the same')
        self.vol_surface_strike_columns.sort(key=lambda x: x.strike)
        self.strikes = [vssc.strike for vssc in self.vol_surface_strike_columns]
        

    def get_volatility(self, t: date, strike: float, interpolation_type: InterpolationMethod=InterpolationMethod.CubicSpline) -> float:
        interpolated_vol_column = self._interpolate_column(strike, interpolation_type)
        vol = interpolated_vol_column.get_volatility(t, interpolation_type)
        return vol
    
    def _interpolate_column(self, strike: float, interpolation_method: InterpolationMethod) -> VolatilitySurfaceStrikeColumn:
        days = [d for d in self.vol_surface_strike_columns[0].vol_surface_column.index]
        vol_column = []
        for d in days:
            vols_row = [vssc.vol_surface_column[d] for vssc in self.vol_surface_strike_columns]
            x, y = self.strikes, vols_row
            if strike < min(x):
                vol = min(x)
            elif strike > max(x):
                vol = max(x)
            elif interpolation_method==InterpolationMethod.Linear:
                vol = np.interp(strike, x, y)
            elif interpolation_method==InterpolationMethod.CubicSpline:
                cs = CubicSpline(x, y)
                vol = cs(strike)
            else:
                raise NotImplementedError(f'InterpolationMethod {interpolation_method} not implemented.')
            vol_column.append(vol)
        srs = pd.Series(vol_column)
        srs.index = [self.t + d for d in days]
        vssc = VolatilitySurfaceStrikeColumn(self.t, strike, srs)
        return vssc