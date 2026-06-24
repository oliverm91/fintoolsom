import pytest

from fintoolsom.rates import Rate, RateConvention, CompoundedInterestConvention
from fintoolsom.dates import ActualDayCountConvention


def test_bond_present_value_with_irr(sample_bond, curve_date):
    irr_convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    irr = Rate(irr_convention, 0.06)
    pv = sample_bond.get_present_value(curve_date, irr)
    assert pv > 0


def test_bond_present_value_with_zero_coupon_curve(sample_bond, sample_zero_coupon_curve, curve_date):
    pv_zc = sample_bond.get_present_value_zc(curve_date, sample_zero_coupon_curve)
    assert pv_zc > 0


def test_bond_dv01_is_negative_for_positive_irr(sample_bond, curve_date):
    irr_convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    irr = Rate(irr_convention, 0.06)
    dv01 = sample_bond.get_dv01(curve_date, irr)
    assert dv01 < 0


def test_bond_z_spread_matches_irr_present_value(sample_bond, sample_zero_coupon_curve, curve_date):
    irr_convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    irr = Rate(irr_convention, 0.06)
    z_spread = sample_bond.get_z_spread(curve_date, irr, sample_zero_coupon_curve)

    irr_pv = sample_bond.get_present_value(curve_date, irr)
    sample_zero_coupon_curve.parallel_bump_rates_bps(z_spread)
    bumped_zc_pv = sample_bond.get_present_value_zc(curve_date, sample_zero_coupon_curve)
    sample_zero_coupon_curve.parallel_bump_rates_bps(-z_spread)

    assert bumped_zc_pv == pytest.approx(irr_pv, abs=1e-4)
