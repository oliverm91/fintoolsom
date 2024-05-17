from dataclasses import dataclass, field
from datetime import date

import pandas as pd

@dataclass
class VolatilitySurfaceStrikeColumn:
    stike: float
    vol_surface: pd.DataFrame

    def __post_init__(self):
        #Check that index are dates
        # Sort them
        pass
    

@dataclass
class VolatilitySurface:
    vol_surface_strike_columns: list[VolatilitySurfaceStrikeColumn]

    def __post_init__(self):
        # Check that indexes in all vol_surface_strike_columns are the same.
        # Sort them by strike
        pass

    def get_volatility(t: date, spot: float):
        raise NotImplementedError()