import cProfile

from datetime import date, timedelta
import io
import pstats

from fintoolsom.dates.calendars import get_ny_calendar
from fintoolsom.dates.schedules import Tenor
from . import create_bonds

from fintoolsom.rates import ZeroCouponCurve, ZeroCouponCurvePoint, Rate, RateConvention, CompoundedInterestConvention
from fintoolsom.dates import ActualDayCountConvention, ModifiedFollowingConvention, get_cl_calendar, ScheduleGenerator

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
nominal_nemo = 'BTP0470930'
notional = 100_000_000
nominal_bond = create_bonds.create_bond(nominal_nemo, notional)

tir_value = 6 / 100
tir_convention = RateConvention(interest_convention=CompoundedInterestConvention, day_count_convention=ActualDayCountConvention, time_fraction_base=365)
irr_obj = Rate(tir_convention, tir_value)


amount_ph = nominal_bond.get_amount_value(t, irr_obj)
print(f'Nominal bond: {nominal_nemo}')
print(f'Notional: {notional}')
print(f'IRR: {irr_obj.rate_value}')
print(f'Zero Coupon Curve: {nominal_zcc}')
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
print('-------')
print()

print('Adding tenors')
tenors_str = [
    '1m', '2m', '6m', '7m', '1w', '1y', '10y', '1d'
]
t = date(2024,3,31)
adj_conv = ModifiedFollowingConvention(get_cl_calendar())
for tenor_str in tenors_str:
    tenor = Tenor(tenor_str, adj_conv)
    print(f'Unadjusted {t} + {tenor_str} = Unadjusted: {tenor.get_unadjusted_maturity(t)}, Adjusted: {tenor.get_adjusted_maturity(t)}')
print('-------')
print()

print(f'Generating Schedules')
ny_cal = get_ny_calendar()
cl_cal = get_cl_calendar()
combined_cal = ny_cal + cl_cal

frequency_tenors = ['1m', '6M']
maturity_tenors = ['3m', '6m', '9M', '12M', '2Y', '6Y']
adj_conv = ModifiedFollowingConvention(combined_cal)
for ft in frequency_tenors:
    for mt in maturity_tenors:
        sf = False
        ls = False
        schedule = ScheduleGenerator.generate_schedule(t, mt, ft, adj_conv, stub_first=sf, long_stub=ls)
        print(f'{t} + {mt}, freq {ft}. Long stub: {ls}, Stub first: {sf} = {schedule}\n')