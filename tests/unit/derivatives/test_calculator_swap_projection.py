from datetime import date
import pytest
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps.legs import FixedLeg, OvernightLeg
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.rates import Rate, RateConvention, LinearInterestConvention
from fintoolsom.dates import ActualDayCountConvention
from tests.factories.curve_factory import index_curve_from_zero_coupon_curve


SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)


ZERO_SPREAD = Rate(RateConvention(), 0.0)


FIXED_RATE = Rate(
    RateConvention(LinearInterestConvention, ActualDayCountConvention, 360), 0.03
)


def _sofr_market(curve_date, sample_zero_coupon_curve) -> Market:
    # SOFR's index curve (SOFR is the risk-free index) both discounts USD and projects SOFR.
    sofr = index_curve_from_zero_coupon_curve(sample_zero_coupon_curve, SOFR)
    return Market(t=curve_date, risk_free_index=SOFR, curves=[sofr])


def test_leg_pv_projects_overnight_leg_via_projection_curve(
    curve_date, sample_zero_coupon_curve
):
    market = _sofr_market(curve_date, sample_zero_coupon_curve)

    start_date = date(2024, 9, 10)
    end_date = date(2024, 10, 10)
    notional = 1_000_000.0
    leg = OvernightLeg(
        start_dates=[start_date],
        end_dates=[end_date],
        residuals=[notional],
        index=SOFR,
        spread=ZERO_SPREAD,
    )

    pv = Calculator._leg_pv(leg, None, market)

    # Overnight coupon accrues the forward WEALTH factor (>1): interest = residual*(wf-1),
    # paid with the notional repayment.
    wf_fwd = sample_zero_coupon_curve.get_wf_fwd(start_date, end_date)
    expected_flow = notional * (wf_fwd - 1) + notional
    expected_pv = expected_flow * sample_zero_coupon_curve.get_df(end_date)
    assert pv == pytest.approx(expected_pv)


def test_swap_mtm_discounts_projected_floating_leg_against_fixed_leg(
    curve_date, sample_zero_coupon_curve
):
    market = _sofr_market(curve_date, sample_zero_coupon_curve)

    start_date = date(2024, 9, 10)
    end_date = date(2024, 10, 10)

    notional = 1_000_000.0
    dates = {
        "start_dates": [start_date],
        "end_dates": [end_date],
        "residuals": [notional],
    }
    receive_leg = FixedLeg(**dates, rate=FIXED_RATE, currency=Currency.USD)
    pay_leg = OvernightLeg(**dates, index=SOFR, spread=ZERO_SPREAD)

    swap = Swap(receive_leg=receive_leg, pay_leg=pay_leg)

    mtm = Calculator.get_swap_mtm(swap, market, Currency.USD)

    # Both legs repay the notional on the same date: it cancels.
    df = sample_zero_coupon_curve.get_df(end_date)
    fixed_pv = FIXED_RATE.get_accrued_interest(notional, start_date, end_date) * df
    wf_fwd = sample_zero_coupon_curve.get_wf_fwd(start_date, end_date)
    floating_pv = notional * (wf_fwd - 1) * df
    assert mtm == pytest.approx(fixed_pv - floating_pv)
