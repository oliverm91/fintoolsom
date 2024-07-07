import cProfile

from datetime import date, timedelta
import io
import pstats
from . import create_bonds

from fintoolsom.rates import ZeroCouponCurve, ZeroCouponCurvePoint, Rate, RateConvention, CompoundedInterestConvention
from fintoolsom.dates import ActualDayCountConvention

# Create Random nominal Zero Coupon Curve

t = date.today()
nominal_curve_points = [
    ZeroCouponCurvePoint(t + timedelta(days=1), Rate(RateConvention(), 0.0575)),
    ZeroCouponCurvePoint(t + timedelta(days=30), Rate(RateConvention(), 0.0565)),
    ZeroCouponCurvePoint(t + timedelta(days=60), Rate(RateConvention(), 0.055)),
    ZeroCouponCurvePoint(t + timedelta(days=90), Rate(RateConvention(), 0.0525)),
    ZeroCouponCurvePoint(t + timedelta(days=180), Rate(RateConvention(), 0.0515)),
    ZeroCouponCurvePoint(t + timedelta(days=365), Rate(RateConvention(), 0.0505)),
    ZeroCouponCurvePoint(t + timedelta(days=720), Rate(RateConvention(), 0.05)),
    ZeroCouponCurvePoint(t + timedelta(days=1000), Rate(RateConvention(), 0.05)),
    ZeroCouponCurvePoint(t + timedelta(days=1500), Rate(RateConvention(), 0.05)),
    ZeroCouponCurvePoint(t + timedelta(days=2000), Rate(RateConvention(), 0.0475)),
]
nominal_zcc = ZeroCouponCurve(t, nominal_curve_points)
nominal_bond = create_bonds.create_bond('BTP0470930', 100_000_000)

tir_value = 6 / 100
tir_convention = RateConvention(interest_convention=CompoundedInterestConvention, day_count_convention=ActualDayCountConvention, time_fraction_base=365)
irr_obj = Rate(tir_convention, tir_value)


amount_ph = nominal_bond.get_amount_value(t, irr_obj)
print(f'Nominal bond amount to pay with IRR: {amount_ph}')
irr_pv = nominal_bond.get_present_value(t, irr_obj)
print(f'Nominal bond PV (base 100) with IRR: {irr_pv}')
zcc_pv = nominal_bond.get_present_value_zc(t, nominal_zcc)
print(f'Nominal bond value with ZCC: {zcc_pv}')

dv01 = nominal_bond.get_dv01(t, irr_obj)
print(f'Nominal bond DV01: {dv01}')
dv01 = nominal_bond.get_dv01(t, irr_obj)
print(f'Nominal bond DV01 base 100: {dv01*100/nominal_bond.notional}')
z_spread = nominal_bond.get_z_spread(t, irr_obj, nominal_zcc)
print(f'Nominal bond Z-Spread: {z_spread}')