from datetime import date
import pickle

from fintoolsom.derivatives.options.volatility_surface import VolatilitySurface, InterpolationMethod
from fintoolsom.derivatives.options.options import Put, Call

def run_tests():
    t = date.today()
    with open('mkt_data.pkl', 'rb') as file:
        loaded_lst = pickle.load(file)
        vol_surf_df, spot, local_zcc, foreign_zcc = loaded_lst
    
    print('Original mkt vols:')
    print(vol_surf_df,'\n')

    print('Interpolating for:')
    print('\t1: Log-moneyness: 0, days: 200')
    print('\t2: Log-moneyness: 0.1, days: 300\n')

    print('---------------------\n')
    print('Two-step linear vols:')
    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.DoubleLinear)
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2, '\n')

    print('Two-step cubic-spline vols:')
    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.DoubleCubicSpline)
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2, '\n')

    print('eSSVI functional-exponential vols:')
    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.eSSVI, parameter_type='functional_form', function_type='exponential')
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2, '\n')

    print('eSSVI functional-polynomial(2) vols:')
    vol1 = vs.get_volatility(0, 200)
    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.eSSVI, parameter_type='functional_form', function_type='polynomial')
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2, '\n')

    print('eSSVI functional-polynomial(3) vols:')
    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.eSSVI, parameter_type='functional_form', function_type='polynomial', polynomial_order=3)
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2, '\n')

    print('eSSVI functional-polynomial(4) vols:')
    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.eSSVI, parameter_type='functional_form', function_type='polynomial', polynomial_order=4)
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2, '\n')

    print('eSSVI parametric vols (linear parameter interpolation):')
    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.eSSVI, parameter_type='parametric', parameters_interpolation_method='linear')
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2, '\n')

    print('eSSVI parametric vols (cupic-spline parameter interpolation):')
    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.eSSVI, parameter_type='parametric', parameters_interpolation_method='cubic-spline')
    vol1 = vs.get_volatility(0, 200)
    vol2 = vs.get_volatility(0.1, 300)
    print(vol1, vol2, '\n')


    vs = VolatilitySurface(vol_surf_df, spot, local_zcc, foreign_zcc, InterpolationMethod.DoubleCubicSpline)
    option = Call(100_000, 970, date(2024, 11, 12))
    option_log_moneyness = option.get_log_moneyness(spot, local_zcc, foreign_zcc)
    vol = vs.get_volatility(option_log_moneyness, (option.maturity - t).days)
    mtm = option.get_mtm(t, spot, vol, local_zcc, foreign_zcc)
    print(f'Call MtM: {mtm}')

    option = Put(100_000, 970, date(2024, 11, 12))
    option_log_moneyness = option.get_log_moneyness(spot, local_zcc, foreign_zcc)
    vol = vs.get_volatility(option_log_moneyness, (option.maturity - t).days)
    mtm = option.get_mtm(t, spot, vol, local_zcc, foreign_zcc)
    print(f'Put MtM: {mtm}')

    print('\n\nOptions tests: Ok')