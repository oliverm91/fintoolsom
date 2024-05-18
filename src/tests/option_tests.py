from datetime import date

from fintoolsom.derivatives.options.volatility_surface import VolatilitySurface, VolatilitySurfaceStrikeColumn
from fintoolsom.dates.datetools import add_tenor, AdjustmentDateConvention, DayCountConvention

def run_tests():
    t = date(2024, 5, 15)
    maturity_tenors = [
        '1W',
        '2W',
        '1M',
        '2M',
        '3M',
        '6M',
        '1Y'
    ]
    maturity_dates = [add_tenor(t, tenor) for tenor in maturity_tenors]
    print(maturity_tenors)