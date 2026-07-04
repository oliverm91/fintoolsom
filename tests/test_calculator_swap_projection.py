from datetime import date

import pytest

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps.coupons import FixedCoupon, OvernightCoupon
from fintoolsom.derivatives.swaps.legs import FixedLeg, OvernightLeg
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.market import Currency, Market, RateIndex

SOFR = RateIndex("SOFR", currency=Currency.USD)


def _sofr_market(curve_date, sample_zero_coupon_curve) -> Market:
    # A single curve stored at (SOFR, USD) serves both as the riskless discount
    # curve for USD and as SOFR's own projection curve (get_projection_df/dfs
    # look it up via (index, index.currency)).
    return Market(t=curve_date, curves={(SOFR, Currency.USD): sample_zero_coupon_curve})


def test_leg_pv_projects_overnight_leg_via_market_get_curve(
    curve_date, sample_zero_coupon_curve
):
    market = _sofr_market(curve_date, sample_zero_coupon_curve)

    start_date = date(2024, 9, 10)
    end_date = date(2024, 10, 10)
    coupon = OvernightCoupon(
        residual=1_000_000.0,
        amortization=0.0,
        start_date=start_date,
        end_date=end_date,
        payment_date=end_date,
        time_fraction=(end_date - start_date).days / 360,
    )
    leg = OvernightLeg(coupons=[coupon], index=SOFR)

    pv = Calculator._leg_pv(leg, None, market, SOFR)

    fwd_df = sample_zero_coupon_curve.get_df_fwd(start_date, end_date)
    expected_flow = coupon.residual * (fwd_df - 1)
    expected_pv = expected_flow * sample_zero_coupon_curve.get_df(end_date)
    assert pv == pytest.approx(expected_pv)


def test_swap_mtm_discounts_projected_floating_leg_against_fixed_leg(
    curve_date, sample_zero_coupon_curve
):
    market = _sofr_market(curve_date, sample_zero_coupon_curve)

    start_date = date(2024, 9, 10)
    end_date = date(2024, 10, 10)
    time_fraction = (end_date - start_date).days / 360

    fixed_coupon = FixedCoupon(
        residual=1_000_000.0,
        amortization=0.0,
        start_date=start_date,
        end_date=end_date,
        payment_date=end_date,
        time_fraction=time_fraction,
        rate=0.03,
    )
    receive_leg = FixedLeg(coupons=[fixed_coupon], currency=Currency.USD)

    overnight_coupon = OvernightCoupon(
        residual=1_000_000.0,
        amortization=0.0,
        start_date=start_date,
        end_date=end_date,
        payment_date=end_date,
        time_fraction=time_fraction,
    )
    pay_leg = OvernightLeg(coupons=[overnight_coupon], index=SOFR)

    swap = Swap(receive_leg=receive_leg, pay_leg=pay_leg)

    mtm = Calculator.get_swap_mtm(swap, market, SOFR, Currency.USD)

    fixed_pv = fixed_coupon.flow * sample_zero_coupon_curve.get_df(end_date)
    fwd_df = sample_zero_coupon_curve.get_df_fwd(start_date, end_date)
    floating_pv = (
        overnight_coupon.residual * (fwd_df - 1) * sample_zero_coupon_curve.get_df(end_date)
    )
    assert mtm == pytest.approx(fixed_pv - floating_pv)
