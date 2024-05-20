from datetime import date
import pickle

import pandas as pd

from PropTools.data.external_data.bbg.bbgInterAPI import get_vol_surface, bdp, get_curve, get_fwd_points

from fintoolsom.derivatives.options.volatility_surface import VolatilitySurface, InterpolationMethod
from fintoolsom.derivatives.options.options import Put
from fintoolsom.dates.datetools import add_tenor, AdjustmentDateConvention, DayCountConvention
from fintoolsom.rates import ZeroCouponCurve, ZeroCouponCurvePoint, RateConvention, InterestConvention, Rate

def run_tests():
    t = date.today()
    ''' # remove # to change executing code
    vol_surf_df = get_vol_surface(use_maturities=True) / 100
    vol_surf_df.index = [(t_i - t).days for t_i in vol_surf_df.index]
    vol_surf_df = vol_surf_df[vol_surf_df.index>100]
    
    spot = bdp('USDCLP Curncy', 'PX_LAST').values[0][0]
    foreign_curve_df = get_curve(currency='usd', locality='us')
    foreign_short_rate = 0.05
    df_2_days = 1 / (1 + foreign_short_rate * (add_tenor(t, '2d')-t).days/360)
    foreign_curve_df['px_last'] *= df_2_days
    foreign_curve_df['yf'] = (foreign_curve_df['maturity'] - t).apply(lambda x: x.days)/360
    foreign_curve_df['linear_act_360_rate'] = (1/foreign_curve_df['px_last']-1)/foreign_curve_df['yf']
    rc = RateConvention(interest_convention=InterestConvention.Linear, time_fraction_base=360)
    foreign_zcc = ZeroCouponCurve(t, foreign_curve_df.apply(lambda row: ZeroCouponCurvePoint(row['maturity'], Rate(rc, row['linear_act_360_rate'])), axis=1).tolist())
    usdclp_fwd_point = get_fwd_points(add_maturities=True)
    foreign_dfs = foreign_zcc.get_dfs(usdclp_fwd_point['maturity'])
    local_dfs = (spot / (spot + usdclp_fwd_point['px_last'])) * foreign_dfs
    local_dfs = pd.DataFrame({'df': local_dfs})
    local_dfs['maturity']=usdclp_fwd_point['maturity']
    local_dfs['yf'] = (local_dfs['maturity'] - t).apply(lambda x: x.days)/360
    local_dfs['linear_act_360_rate'] = (1/local_dfs['df']-1)/local_dfs['yf']
    local_zcc = ZeroCouponCurve(t, local_dfs.apply(lambda x: ZeroCouponCurvePoint(x['maturity'], Rate(rc, x['linear_act_360_rate'])), axis=1).tolist())
    saved_data = [vol_surf_df, spot, local_zcc, foreign_zcc]
    with open('mkt_data.pkl', 'wb') as file:
        pickle.dump(saved_data, file)
    '''
    with open('mkt_data.pkl', 'rb') as file:
        loaded_lst = pickle.load(file)
        vol_surf_df, spot, local_zcc, foreign_zcc = loaded_lst
    #'''

    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.DoubleLinear)
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2)

    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.DoubleCubicSpline)
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2)

    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.eSSVI)
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2)