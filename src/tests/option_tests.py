from datetime import date

import pandas as pd

from fintoolsom.derivatives.options.volatility_surface import VolatilitySurface, VolatilitySurfaceStrikeColumn
from fintoolsom.dates.datetools import add_tenor, AdjustmentDateConvention, DayCountConvention

def run_tests():
    t = date(2024, 5, 15)
    maturity_vols_dict = {
        '1W': 25.3,
        '2W': 22.4,
        '1M': 18.3,
        '2M': 15,
        '3M': 14.3,
        '6M': 14.2,
        '1Y': 13.3
    }
    maturity_vol_data = [{
                            'maturity': add_tenor(t, tenor, adj_convention=AdjustmentDateConvention.Following),
                            'vol': vol
                            }
                           for tenor, vol in maturity_vols_dict.items()]
    
    df = pd.DataFrame(maturity_vol_data)
    df.set_index('maturity', inplace=True)
    series = df['vol']
    vssc = VolatilitySurfaceStrikeColumn(t, 850, series)
    print(vssc.vol_surface_column)