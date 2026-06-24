from datetime import date

import numpy as np
import pandas as pd

from fintoolsom.derivatives.options.options import Call
from fintoolsom.derivatives.options.volatility_surface import VolatilitySurface, InterpolationMethod


def _sample_vol_surface_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            25: [0.15, 0.15],
            50: [0.15, 0.15],
            75: [0.15, 0.15],
        },
        index=[30, 90],
    )


def test_volatility_surface_flat_smile_returns_flat_vol(sample_zero_coupon_curve):
    vol_surf_df = _sample_vol_surface_df()
    spot = 940
    vs = VolatilitySurface(vol_surf_df, spot, sample_zero_coupon_curve, sample_zero_coupon_curve, InterpolationMethod.DoubleCubicSpline)
    vol = vs.get_volatility(0.1, 60)
    assert vol == 0.15


def test_call_mtm_is_positive_for_in_the_money_option(sample_zero_coupon_curve):
    vol_surf_df = _sample_vol_surface_df()
    spot = 940
    strike = 900
    maturity = date(2024, 10, 12)
    t_val = date(2024, 8, 12)

    vs = VolatilitySurface(vol_surf_df, spot, sample_zero_coupon_curve, sample_zero_coupon_curve, InterpolationMethod.DoubleCubicSpline)
    call = Call(1000, strike, maturity)
    log_moneyness = call.get_log_moneyness(spot, sample_zero_coupon_curve, sample_zero_coupon_curve)
    vol = vs.get_volatility(log_moneyness, (maturity - t_val).days)
    mtm = call.get_mtm(t_val, spot, vol, sample_zero_coupon_curve, sample_zero_coupon_curve)
    assert mtm > 0
