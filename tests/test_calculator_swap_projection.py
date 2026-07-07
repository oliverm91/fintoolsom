from datetime import date

import pytest

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps.coupons import FixedCoupon, OvernightCoupon
from fintoolsom.derivatives.swaps.legs import FixedLeg, OvernightLeg
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.rates import Rate, RateConvention, LinearInterestConvention
from fintoolsom.dates import ActualDayCountConvention

SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)
ZERO_SPREAD = Rate(RateConvention(), 0.0)
FIXED_RATE = Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 360), 0.03)


def _sofr_market(curve_date, sample_zero_coupon_curve) -> Market:
    # A single curve stored at (SOFR, USD) serves both as the riskless discount curve
    # for USD and — via get_projection's fallback view — as SOFR's projection curve.
    return Market(t=curve_date, curves={(SOFR, Currency.USD): sample_zero_coupon_curve})


def test_leg_pv_projects_overnight_leg_via_projection_curve(
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
        spread=ZERO_SPREAD,
    )
    leg = OvernightLeg(coupons=[coupon], index=SOFR, spread=ZERO_SPREAD)

    pv = Calculator._leg_pv(leg, None, market, SOFR)

    # Overnight coupon accrues the forward WEALTH factor (>1): interest = residual*(wf-1).
    wf_fwd = sample_zero_coupon_curve.get_wf_fwd(start_date, end_date)
    expected_flow = coupon.residual * (wf_fwd - 1)
    expected_pv = expected_flow * sample_zero_coupon_curve.get_df(end_date)
    assert pv == pytest.approx(expected_pv)


def test_swap_mtm_discounts_projected_floating_leg_against_fixed_leg(
    curve_date, sample_zero_coupon_curve
):
    market = _sofr_market(curve_date, sample_zero_coupon_curve)

    start_date = date(2024, 9, 10)
    end_date = date(2024, 10, 10)

    fixed_coupon = FixedCoupon(
        residual=1_000_000.0,
        amortization=0.0,
        start_date=start_date,
        end_date=end_date,
        payment_date=end_date,
        rate=FIXED_RATE,
    )
    receive_leg = FixedLeg(coupons=[fixed_coupon], currency=Currency.USD)

    overnight_coupon = OvernightCoupon(
        residual=1_000_000.0,
        amortization=0.0,
        start_date=start_date,
        end_date=end_date,
        payment_date=end_date,
        spread=ZERO_SPREAD,
    )
    pay_leg = OvernightLeg(coupons=[overnight_coupon], index=SOFR, spread=ZERO_SPREAD)

    swap = Swap(receive_leg=receive_leg, pay_leg=pay_leg)

    mtm = Calculator.get_swap_mtm(swap, market, SOFR, Currency.USD)

    fixed_pv = fixed_coupon.flow * sample_zero_coupon_curve.get_df(end_date)
    wf_fwd = sample_zero_coupon_curve.get_wf_fwd(start_date, end_date)
    floating_pv = (
        overnight_coupon.residual * (wf_fwd - 1) * sample_zero_coupon_curve.get_df(end_date)
    )
    assert mtm == pytest.approx(fixed_pv - floating_pv)
