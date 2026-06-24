from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.fixedIncome import Bond, Coupon, Coupons
from fintoolsom.models import NelsonSiegelSvensson
from fintoolsom.rates import (
    Rate,
    RateConvention,
    CompoundedInterestConvention,
    ExponentialInterestConvention,
    ZeroCouponCurve,
)
from fintoolsom.dates import ActualDayCountConvention


def _zero_coupon_bond(
    calibration_date: date, years: int, notional: float = 1_000_000
) -> Bond:
    convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    maturity = calibration_date + timedelta(days=365 * years)
    coupon = Coupon(100, 5, 100, calibration_date, maturity, convention)
    return Bond(coupons=Coupons([coupon]), currency="clp", notional=notional)


def test_nss_calibrate_fits_market_irrs():
    calibration_date = date(2024, 7, 8)
    convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    bonds_irr_list = [
        (_zero_coupon_bond(calibration_date, 1), Rate(convention, 0.05)),
        (_zero_coupon_bond(calibration_date, 3), Rate(convention, 0.052)),
        (_zero_coupon_bond(calibration_date, 5), Rate(convention, 0.054)),
        (_zero_coupon_bond(calibration_date, 10), Rate(convention, 0.056)),
    ]

    nss = NelsonSiegelSvensson()
    nss.calibrate(calibration_date, bonds_irr_list)

    for bond, irr in bonds_irr_list:
        market_pv = bond.get_present_value(calibration_date, irr)
        t = (bond.get_maturity_date() - calibration_date).days / 365
        ns_pv = bond.coupons.get_flows()[0] * nss.get_df(t)
        assert ns_pv == pytest.approx(market_pv, rel=0.02)


def test_nss_calibrate_from_curve_fits_zero_rates():
    curve_date = date(2024, 7, 8)
    convention = RateConvention(
        ExponentialInterestConvention, ActualDayCountConvention, 365
    )
    source_nss = NelsonSiegelSvensson()
    (
        source_nss.b0,
        source_nss.b1,
        source_nss.b2,
        source_nss.b3,
        source_nss.lambda_,
        source_nss.mu_,
    ) = 0.05, -0.01, 0.005, 0.01, 1.5, 8

    maturities = [
        curve_date + timedelta(days=365 * years)
        for years in (1, 2, 3, 5, 7, 10, 15, 20)
    ]
    ts = np.array([(mat - curve_date).days / 365 for mat in maturities])
    dfs = source_nss.get_df(ts)
    curve = ZeroCouponCurve(curve_date, date_dfs=list(zip(maturities, dfs)))

    fitted_nss = NelsonSiegelSvensson()
    fitted_nss.calibrate_from_curve(curve)

    fitted_rates = fitted_nss.get_rate(ts)
    target_rates = curve.get_zero_rates_values(convention)
    assert fitted_rates == pytest.approx(target_rates, abs=1e-4)


def test_get_rate_and_get_df_scalar_vs_array_shapes():
    nss = NelsonSiegelSvensson()

    rate_scalar = nss.get_rate(1.0)
    rate_array = nss.get_rate(np.array([1.0, 2.0, 5.0]))
    assert np.isscalar(rate_scalar) or rate_scalar.shape == ()
    assert rate_array.shape == (3,)
    assert rate_array[0] == pytest.approx(rate_scalar)

    df_scalar = nss.get_df(1.0)
    df_array = nss.get_df(np.array([1.0, 2.0, 5.0]))
    assert df_array.shape == (3,)
    assert df_array[0] == pytest.approx(df_scalar)
    assert df_scalar == pytest.approx(np.exp(-rate_scalar * 1.0))


def test_get_curve_is_consistent_with_calibrated_model_df():
    calibration_date = date(2024, 7, 8)
    convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    bonds_irr_list = [
        (_zero_coupon_bond(calibration_date, 1), Rate(convention, 0.05)),
        (_zero_coupon_bond(calibration_date, 3), Rate(convention, 0.052)),
        (_zero_coupon_bond(calibration_date, 5), Rate(convention, 0.054)),
        (_zero_coupon_bond(calibration_date, 10), Rate(convention, 0.056)),
    ]

    nss = NelsonSiegelSvensson()
    curve = nss.get_curve(calibration_date, bonds_irr_list)

    maturity = bonds_irr_list[0][0].get_maturity_date()
    t = (maturity - calibration_date).days / 365
    assert curve.get_df(maturity) == pytest.approx(nss.get_df(t), abs=1e-3)


def test_design_matrix_is_finite_at_calibration_bounds():
    t = np.array([0.25, 1, 5, 10, 20])
    for lambda_, mu_ in [(0.5, 2), (0.5, 20), (3, 2), (3, 20)]:
        design_matrix = NelsonSiegelSvensson._get_design_matrix(t, lambda_, mu_)
        assert np.all(np.isfinite(design_matrix))
