from datetime import date

import pytest

from fintoolsom.fixedIncome import CLBond, Coupon, Coupons
from fintoolsom.rates import (
    Rate,
    RateConvention,
    CompoundedInterestConvention,
    LinearInterestConvention,
)
from fintoolsom.dates import ActualDayCountConvention, Days30EDayCountConvention


def test_bond_present_value_with_irr(sample_bond, curve_date):
    irr_convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    irr = Rate(irr_convention, 0.06)
    pv = sample_bond.get_present_value(curve_date, irr)
    assert pv > 0


def test_bond_present_value_with_zero_coupon_curve(
    sample_bond, sample_zero_coupon_curve, curve_date
):
    pv_zc = sample_bond.get_present_value_zc(curve_date, sample_zero_coupon_curve)
    assert pv_zc > 0


def test_bond_dv01_is_negative_for_positive_irr(sample_bond, curve_date):
    irr_convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    irr = Rate(irr_convention, 0.06)
    dv01 = sample_bond.get_dv01(curve_date, irr)
    assert dv01 < 0


def test_bond_z_spread_matches_irr_present_value(
    sample_bond, sample_zero_coupon_curve, curve_date
):
    irr_convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    irr = Rate(irr_convention, 0.06)
    z_spread = sample_bond.get_z_spread(curve_date, irr, sample_zero_coupon_curve)

    irr_pv = sample_bond.get_present_value(curve_date, irr)
    sample_zero_coupon_curve.parallel_bump_rates_bps(z_spread)
    bumped_zc_pv = sample_bond.get_present_value_zc(
        curve_date, sample_zero_coupon_curve
    )
    sample_zero_coupon_curve.parallel_bump_rates_bps(-z_spread)

    assert bumped_zc_pv == pytest.approx(irr_pv, abs=1e-4)


def test_irr_from_amount_round_trips_for_real_notional(sample_bond, curve_date):
    """Regression test: initial_guess in get_irr_from_amount must scale tera_value by
    notional/100 to match the notional-scaled amount and dv01, otherwise the Newton
    solve diverges for any notional != 100 (sample_bond uses 100_000_000)."""
    irr_convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    irr = Rate(irr_convention, 0.06)
    amount = sample_bond.get_amount_value(curve_date, irr)

    recovered_irr = sample_bond.get_irr_from_amount(
        curve_date, amount, irr_rate_convention=irr_convention
    )
    recovered_amount = sample_bond.get_amount_value(curve_date, recovered_irr)

    assert recovered_amount == amount


def test_init_raises_on_wrong_coupons_type():
    with pytest.raises(TypeError):
        CLBond(coupons="not coupons", currency="clp", notional=100)


def test_init_raises_on_wrong_currency_type():
    coupon_rate_convention = RateConvention(
        LinearInterestConvention, Days30EDayCountConvention, 360
    )
    coupons = Coupons(
        [
            Coupon(
                100, 2, 100, date(2024, 1, 1), date(2024, 7, 1), coupon_rate_convention
            )
        ]
    )
    with pytest.raises(TypeError):
        CLBond(coupons=coupons, currency=123, notional=100)


def test_init_raises_on_non_positive_notional():
    coupon_rate_convention = RateConvention(
        LinearInterestConvention, Days30EDayCountConvention, 360
    )
    coupons = Coupons(
        [
            Coupon(
                100, 2, 100, date(2024, 1, 1), date(2024, 7, 1), coupon_rate_convention
            )
        ]
    )
    with pytest.raises(ValueError):
        CLBond(coupons=coupons, currency="clp", notional=0)
