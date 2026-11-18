from datetime import date, timedelta
import numpy as np
import pytest
from fintoolsom.fixedIncome import CLBond, Coupon, Coupons
from fintoolsom.models import NelsonSiegelSvensson
from fintoolsom.models.NelsonSiegelSvensson import (
    HUMP_PEAK,
    _DEFAULT_HUMPS,
    _HUMP_BOUNDS,
    nss_factors,
)
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
) -> CLBond:
    convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    maturity = calibration_date + timedelta(days=365 * years)
    coupon = Coupon(100, 5, 100, calibration_date, maturity, convention)
    return CLBond(coupons=Coupons([coupon]), currency="clp", notional=notional)


def _bonds_irr_list(calibration_date: date) -> list[tuple[CLBond, Rate]]:
    convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    return [
        (_zero_coupon_bond(calibration_date, 1), Rate(convention, 0.05)),
        (_zero_coupon_bond(calibration_date, 3), Rate(convention, 0.052)),
        (_zero_coupon_bond(calibration_date, 5), Rate(convention, 0.054)),
        (_zero_coupon_bond(calibration_date, 10), Rate(convention, 0.056)),
    ]


def test_nss_calibrate_from_bonds_fits_market_irrs():
    calibration_date = date(2024, 7, 8)
    bonds_irr_list = _bonds_irr_list(calibration_date)

    nss = NelsonSiegelSvensson()
    nss.calibrate_from_bonds(calibration_date, bonds_irr_list)

    for bond, irr in bonds_irr_list:
        market_pv = bond.get_present_value(calibration_date, irr)
        t = (bond.get_maturity_date() - calibration_date).days / 365
        ns_pv = bond.coupons.get_flows()[0] * nss.get_df(t)
        assert ns_pv == pytest.approx(market_pv, rel=0.02)


def test_calibrate_is_a_deprecated_alias_of_calibrate_from_bonds():
    calibration_date = date(2024, 7, 8)
    expected = NelsonSiegelSvensson()
    expected.calibrate_from_bonds(calibration_date, _bonds_irr_list(calibration_date))

    nss = NelsonSiegelSvensson()
    with pytest.warns(DeprecationWarning, match="use calibrate_from_bonds instead"):
        nss.calibrate(calibration_date, _bonds_irr_list(calibration_date))
    assert nss.get_params() == pytest.approx(expected.get_params())


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


def test_zero_rate_is_linear_in_the_betas_with_finite_limits_at_zero():
    params = (0.05, -0.01, 0.01, 0.005, 1.5, 0.3)
    model = NelsonSiegelSvensson()
    model.b0, model.b1, model.b2, model.b3, model.lambda_, model.mu_ = params
    t = np.array([0.0, 0.1, 1.0, 5.0, 20.0])
    design = NelsonSiegelSvensson._get_design_matrix(t, *params[4:])
    assert model.get_rate(t) == pytest.approx(design @ np.array(params[:4]), abs=1e-15)
    # t=0 limits: short rate b0 + b1 and df = 1, no 0/0.
    assert model.get_rate(0.0) == pytest.approx(params[0] + params[1], abs=1e-15)
    assert model.get_df(0.0) == 1.0
    assert model.get_df(t[1:]) == pytest.approx(
        np.exp(-model.get_rate(t[1:]) * t[1:]), rel=1e-15
    )


def test_design_matrix_is_finite_at_calibration_bounds():
    t = np.array([0.25, 1, 5, 10, 20])
    for lambda_, mu_ in [(0.5, 2), (0.5, 20), (3, 2), (3, 20)]:
        design_matrix = NelsonSiegelSvensson._get_design_matrix(t, lambda_, mu_)
        assert np.all(np.isfinite(design_matrix))


# ── lambda_ and mu_ are the years to each hump ──────────────────────────────


def _h(x):
    return (1 - np.exp(-x)) / x - np.exp(-x)


def _svensson_rate(t, betas, first_hump, second_hump):
    """Svensson's zero rate written out, humps at `first_hump` and `second_hump` years:
    time scales tau = hump / HUMP_PEAK (an independent check of nss_factors)."""
    x1, x2 = HUMP_PEAK * t / first_hump, HUMP_PEAK * t / second_hump
    b0, b1, b2, b3 = betas
    return b0 + b1 * (1 - np.exp(-x1)) / x1 + b2 * _h(x1) + b3 * _h(x2)


def test_hump_peak_solves_the_first_order_condition_of_the_curvature_loading():
    # h'(x) = 0  <=>  x² + x + 1 = e^x
    assert HUMP_PEAK**2 + HUMP_PEAK + 1 == pytest.approx(np.exp(HUMP_PEAK), rel=1e-15)
    x = np.linspace(0.01, 20, 200_001)
    assert x[np.argmax(_h(x))] == pytest.approx(HUMP_PEAK, abs=1e-4)
    assert _h(HUMP_PEAK) == pytest.approx(0.2984, abs=1e-4)


@pytest.mark.parametrize(
    "lambda_, mu_", [(0.5, 2), (2, 5), (1.5, 12), (3, 20)], ids=str
)
def test_each_hump_loading_peaks_at_its_parameter(lambda_, mu_):
    t = np.linspace(0.0005, 40, 80_000)
    loadings = nss_factors(t, lambda_, mu_)
    assert t[np.argmax(loadings[:, 2])] == pytest.approx(lambda_, abs=1e-3)
    assert t[np.argmax(loadings[:, 3])] == pytest.approx(mu_, abs=1e-3)


def test_loadings_are_svensson_with_time_scales_hump_over_hump_peak():
    t = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 30])
    lambda_, mu_ = 1.5, 12.0
    x1, x2 = HUMP_PEAK * t / lambda_, HUMP_PEAK * t / mu_
    expected = np.column_stack(
        [np.ones_like(t), (1 - np.exp(-x1)) / x1, _h(x1), _h(x2)]
    )
    np.testing.assert_allclose(nss_factors(t, lambda_, mu_), expected, rtol=1e-14)


def test_default_humps_and_starting_points_are_inside_the_bounds():
    model = NelsonSiegelSvensson()
    assert (model.lambda_, model.mu_) == _DEFAULT_HUMPS == (2.0, 5.0)
    assert _HUMP_BOUNDS == ((0.5, 3.0), (2.0, 20.0))
    for value, (low, high) in zip(_DEFAULT_HUMPS, _HUMP_BOUNDS):
        assert low <= value <= high


@pytest.mark.parametrize(
    "betas, first_hump, second_hump",
    [((0.05, -0.01, 0.01, 0.03), 1.5, 12.0), ((0.045, 0.01, -0.02, 0.02), 2.5, 8.0)],
    ids=["second-hump-12y", "second-hump-8y"],
)
def test_calibrate_from_curve_recovers_a_long_second_hump(
    betas, first_hump, second_hump
):
    # Before the humps were in years, mu_ was a decay rate whose hump stayed within
    # 0.09-0.9y: these fits ended on the bounds, 4-8bp off.
    curve_date = date(2024, 7, 8)
    years = [0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30]
    maturities = [curve_date + timedelta(days=round(365 * y)) for y in years]
    ts = np.array([(m - curve_date).days / 365 for m in maturities])
    target = _svensson_rate(ts, betas, first_hump, second_hump)
    curve = ZeroCouponCurve(
        curve_date, date_dfs=list(zip(maturities, np.exp(-target * ts)))
    )

    nss = NelsonSiegelSvensson()
    nss.calibrate_from_curve(curve)

    assert nss.lambda_ == pytest.approx(first_hump, abs=0.1)
    assert nss.mu_ == pytest.approx(second_hump, abs=0.1)
    assert np.max(np.abs(nss.get_rate(ts) - target)) < 1e-5


def test_calibrate_from_bonds_converges_on_a_long_second_hump():
    # It raised "could not converge" here before. Its loose tolerance (tol=0.01 on
    # squared base-100 prices) still stops about 12bp from the exact fit.
    calibration_date = date(2024, 7, 8)
    convention = RateConvention(
        CompoundedInterestConvention, ActualDayCountConvention, 365
    )
    betas = (0.05, -0.01, 0.01, 0.03)
    bonds_irr_list = []
    for years in (1, 2, 3, 5, 7, 10, 12, 15, 20, 30):
        bond = _zero_coupon_bond(calibration_date, years)
        t = (bond.get_maturity_date() - calibration_date).days / 365
        df = np.exp(-_svensson_rate(np.array(t), betas, 1.5, 12.0) * t)
        bonds_irr_list.append((bond, Rate(convention, float(df ** (-1 / t) - 1))))

    nss = NelsonSiegelSvensson()
    nss.calibrate_from_bonds(calibration_date, bonds_irr_list)

    ts = np.array(
        [(b.get_maturity_date() - calibration_date).days / 365 for b, _ in bonds_irr_list]
    )
    assert nss.mu_ > 5 and nss.lambda_ <= 3
    assert np.max(np.abs(nss.get_rate(ts) - _svensson_rate(ts, betas, 1.5, 12.0))) < 2e-3
