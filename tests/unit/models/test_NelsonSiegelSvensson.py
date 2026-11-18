import sys
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from fintoolsom.fixedIncome import CLBond, Coupon, Coupons
from fintoolsom.models import NelsonSiegel, NelsonSiegelSvensson
from fintoolsom.models.NelsonSiegelSvensson import (
    HUMP_PEAK,
    ns_df,
    ns_factors,
    nss_factors,
)
from fintoolsom.rates import (
    Rate,
    RateConvention,
    CompoundedInterestConvention,
    ExponentialInterestConvention,
    InterpolationMethod,
    ModelCurve,
    PillarCurve,
)
from fintoolsom.dates import ActualDayCountConvention
from tests.factories import clbond_factory
from tests.factories.rate_factory import comp_act365


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
    curve = PillarCurve.from_dfs(
        curve_date, maturities, list(dfs), InterpolationMethod.PiecewiseConstant
    )

    fitted_nss = NelsonSiegelSvensson()
    fitted_nss.calibrate_from_curve(curve)

    fitted_rates = fitted_nss.get_rate(ts)
    target_rates = [curve.get_zero_rate(m, convention).rate for m in maturities]
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
    design = NelsonSiegelSvensson._factors(t, *params[4:])
    assert model.get_rate(t) == pytest.approx(design @ np.array(params[:4]), abs=1e-15)
    # t=0 limits: short rate b0 + b1 and df = 1, no 0/0.
    assert model.get_rate(0.0) == pytest.approx(params[0] + params[1], abs=1e-15)
    assert model.get_df(0.0) == 1.0
    assert model.get_df(t[1:]) == pytest.approx(
        np.exp(-model.get_rate(t[1:]) * t[1:]), rel=1e-15
    )


def test_design_matrix_is_finite_at_calibration_bounds():
    t = np.array([0.25, 1, 5, 10, 20])
    for lambda_, mu_ in [(0.25, 2), (0.25, 20), (3, 2), (3, 20)]:
        design_matrix = NelsonSiegelSvensson._factors(t, lambda_, mu_)
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
    nss = NelsonSiegelSvensson()
    assert (nss.lambda_, nss.mu_) == (2.0, 5.0)
    assert NelsonSiegelSvensson._HUMP_BOUNDS == ((0.25, 3.0), (2.0, 20.0))
    assert NelsonSiegel().lambda_ == 2.0
    assert NelsonSiegel._HUMP_BOUNDS == ((0.25, 20.0),)
    for model in (NelsonSiegel(), nss):
        humps = model.get_params()[-len(model._HUMP_BOUNDS) :]
        for value, (low, high) in zip(humps, model._HUMP_BOUNDS):
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
    curve = PillarCurve.from_dfs(
        curve_date,
        maturities,
        list(np.exp(-target * ts)),
        InterpolationMethod.PiecewiseConstant,
    )

    nss = NelsonSiegelSvensson()
    nss.calibrate_from_curve(curve)

    assert nss.lambda_ == pytest.approx(first_hump, abs=0.1)
    assert nss.mu_ == pytest.approx(second_hump, abs=0.1)
    assert np.max(np.abs(nss.get_rate(ts) - target)) < 1e-5


def test_calibrate_from_bonds_converges_on_a_long_second_hump():
    # Powell raised "could not converge" here, then stopped about 12bp off with its
    # loose tolerance. Least squares on the bond residuals recovers the curve.
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
    assert nss.lambda_ == pytest.approx(1.5, abs=1e-4)
    assert nss.mu_ == pytest.approx(12.0, abs=1e-3)
    assert np.max(np.abs(nss.get_rate(ts) - _svensson_rate(ts, betas, 1.5, 12.0))) < 1e-9


def test_calibrate_from_bonds_leaves_the_callers_list_in_order():
    calibration_date = date(2024, 7, 8)
    bonds_irr_list = _bonds_irr_list(calibration_date)[::-1]
    before = list(bonds_irr_list)
    NelsonSiegelSvensson().calibrate_from_bonds(calibration_date, bonds_irr_list)
    assert bonds_irr_list == before


def test_parameters_are_constructor_fields_betas_then_humps():
    nss = NelsonSiegelSvensson(0.05, -0.01, 0.02, 0.01, 1.5, 12.0)
    assert nss.get_params() == [0.05, -0.01, 0.02, 0.01, 1.5, 12.0]
    ns = NelsonSiegel(b0=0.05, lambda_=4.0)
    assert ns.get_params() == [0.05, 0.01, 0.0, 4.0]


# ── Nelson-Siegel ───────────────────────────────────────────────────────────


def test_nelson_siegel_and_svensson_are_siblings():
    assert not issubclass(NelsonSiegelSvensson, NelsonSiegel)
    assert not issubclass(NelsonSiegel, NelsonSiegelSvensson)
    assert NelsonSiegel() != NelsonSiegelSvensson()


def test_ns_factors_are_the_first_three_svensson_factors():
    t = np.array([0.0, 0.1, 1.0, 5.0, 30.0])
    np.testing.assert_allclose(
        ns_factors(t, 1.5), nss_factors(t, 1.5, 12.0)[:, :3], rtol=1e-15
    )
    assert ns_factors(2.0, 1.5).shape == (3,)


def test_nelson_siegel_is_svensson_without_the_second_curvature():
    t = np.array([0.0, 0.25, 1.0, 5.0, 10.0, 30.0])
    ns = NelsonSiegel(0.05, -0.01, 0.02, 1.5)
    nss = NelsonSiegelSvensson(0.05, -0.01, 0.02, 0.0, 1.5, 12.0)
    np.testing.assert_allclose(ns.get_rate(t), nss.get_rate(t), rtol=1e-15)
    np.testing.assert_allclose(ns.get_df(t), ns_df(t, *ns.get_params()), rtol=1e-15)
    assert ns.get_rate(0.0) == pytest.approx(0.04, abs=1e-15)
    assert ns.get_df(0.0) == 1.0


def test_nelson_siegel_curvature_peaks_at_lambda():
    t = np.linspace(0.0005, 40, 80_000)
    for lambda_ in (0.5, 2.0, 7.0):
        assert t[np.argmax(ns_factors(t, lambda_)[:, 2])] == pytest.approx(
            lambda_, abs=1e-3
        )


@pytest.mark.parametrize("hump", [1.0, 4.0, 9.0])
def test_nelson_siegel_calibrate_from_curve_recovers_its_hump(hump):
    curve_date = date(2024, 7, 8)
    years = [0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30]
    maturities = [curve_date + timedelta(days=round(365 * y)) for y in years]
    ts = np.array([(m - curve_date).days / 365 for m in maturities])
    source = NelsonSiegel(0.05, -0.015, 0.02, hump)
    curve = PillarCurve.from_dfs(
        curve_date,
        maturities,
        list(source.get_df(ts)),
        InterpolationMethod.PiecewiseConstant,
    )

    ns = NelsonSiegel()
    ns.calibrate_from_curve(curve)

    assert ns.lambda_ == pytest.approx(hump, abs=0.05)
    assert np.max(np.abs(ns.get_rate(ts) - source.get_rate(ts))) < 1e-6


def test_nelson_siegel_calibrate_from_bonds_fits_market_irrs():
    calibration_date = date(2024, 7, 8)
    bonds_irr_list = _bonds_irr_list(calibration_date)

    ns = NelsonSiegel()
    ns.calibrate_from_bonds(calibration_date, bonds_irr_list)

    assert len(ns.get_params()) == 4
    for bond, irr in bonds_irr_list:
        market_pv = bond.get_present_value(calibration_date, irr)
        t = (bond.get_maturity_date() - calibration_date).days / 365
        assert bond.coupons.get_flows()[0] * ns.get_df(t) == pytest.approx(
            market_pv, rel=0.02
        )


# ── Calibration internals ───────────────────────────────────────────────────

MODELS = [
    NelsonSiegel(0.05, -0.015, 0.02, 4.0),
    NelsonSiegelSvensson(0.05, -0.01, 0.01, 0.03, 1.5, 12.0),
]
IDS = ["NS", "NSS"]


def _humps(model):
    return model.get_params()[model._n_betas :]


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_calibration_design_is_the_factors_away_from_zero(model):
    t = np.array([0.01, 0.5, 1.0, 5.0, 30.0])
    humps = _humps(model)
    design = model._calibration_design(t, humps)
    # Same formulas; s - e at small t loses a few digits to cancellation.
    np.testing.assert_allclose(design, model._factors(t, *humps), rtol=1e-12)
    with_derivatives, _ = model._design_and_derivatives(t, humps)
    np.testing.assert_allclose(with_derivatives, design, rtol=1e-15)
    # A column of humps gives one design per row.
    batch = [np.array([[h], [2 * h]]) for h in humps]
    stacked = model._calibration_design(t, batch)
    assert stacked.shape == (2, t.size, len(humps) + 2)
    np.testing.assert_allclose(stacked[1], model._factors(t, *(2 * h for h in humps)))


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_design_derivatives_match_central_differences(model):
    t = np.array([0.25, 1.0, 3.0, 7.0, 15.0, 30.0])
    humps = np.array(_humps(model))
    _, derivatives = model._design_and_derivatives(t, humps)
    for j, derivative in enumerate(derivatives):
        step = 1e-6 * humps[j]
        up, down = humps.copy(), humps.copy()
        up[j] += step
        down[j] -= step
        numeric = (
            model._calibration_design(t, up) - model._calibration_design(t, down)
        ) / (2 * step)
        np.testing.assert_allclose(derivative, numeric, rtol=1e-7, atol=1e-11)


def _coupon_bonds(model):
    """Ten bullet bonds, 1 to 30 years, priced on `model` and quoted at their IRRs."""
    calibration_date = date(2024, 7, 8)
    curve = ModelCurve(calibration_date, model, calibration_date + timedelta(days=11500))
    bonds = []
    for years in (1, 2, 3, 5, 7, 10, 12, 15, 20, 30):
        bond = clbond_factory.bullet_clbond(
            start=calibration_date, n_coupons=2 * years, coupon_amount=2.5
        )
        pv = bond.get_present_value_zc(calibration_date, curve)
        bonds.append(
            (bond, bond.get_irr_from_present_value(calibration_date, pv, comp_act365()))
        )
    return calibration_date, bonds


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_calibrate_from_bonds_recovers_the_model_from_coupon_bonds(model):
    calibration_date, bonds = _coupon_bonds(model)
    fitted = type(model)()
    fitted.calibrate_from_bonds(calibration_date, bonds)
    np.testing.assert_allclose(fitted.get_params(), model.get_params(), rtol=1e-6)


def test_fewer_bonds_than_parameters_still_fit():
    # Four bonds for six NSS parameters: an underdetermined fit still reprices them.
    calibration_date, bonds = _coupon_bonds(MODELS[1])
    nss = NelsonSiegelSvensson()
    nss.calibrate_from_bonds(calibration_date, bonds[:4])
    curve = ModelCurve(calibration_date, nss, calibration_date + timedelta(days=11500))
    for bond, irr in bonds[:4]:
        assert bond.get_present_value_zc(calibration_date, curve) == pytest.approx(
            bond.get_present_value(calibration_date, irr), rel=1e-9
        )
    assert 0.25 <= nss.lambda_ <= 3 and 2 <= nss.mu_ <= 20


def _curve_of(model, years=(0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30)):
    curve_date = date(2024, 7, 8)
    maturities = [curve_date + timedelta(days=round(365 * y)) for y in years]
    ts = np.array([(m - curve_date).days / 365 for m in maturities])
    curve = PillarCurve.from_dfs(
        curve_date,
        maturities,
        list(model.get_df(ts)),
        InterpolationMethod.PiecewiseConstant,
    )
    return curve, ts


def test_an_initial_guess_outside_the_bounds_is_clipped():
    curve, ts = _curve_of(MODELS[1])
    nss = NelsonSiegelSvensson()
    nss.calibrate_from_curve(curve, initial_guess=[0.1, 50.0])
    assert np.max(np.abs(nss.get_rate(ts) - MODELS[1].get_rate(ts))) < 1e-10


def _dense_grid_best_sse(model, curve, ts, axes):
    pillars = [d for d in curve.knot_dates if d > curve.curve_date]
    target = -np.log(curve.get_dfs(pillars)) / ts
    best = np.inf
    for humps in np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T:
        X = model._calibration_design(ts, humps)
        betas = np.linalg.lstsq(X, target, rcond=None)[0]
        best = min(best, float(np.sum((X @ betas - target) ** 2)))
    return best, float(np.sum((model.get_rate(ts) - target) ** 2))


def test_a_first_hump_down_to_a_quarter_year_is_recovered():
    # The first hump's lower bound is 0.25 years: a 0.3-year hump is inside it.
    source = NelsonSiegel(0.05, -0.015, 0.03, 0.3)
    curve, ts = _curve_of(source)
    ns = NelsonSiegel()
    ns.calibrate_from_curve(curve)
    assert ns.lambda_ == pytest.approx(0.3, rel=1e-4)
    # A hump before the first pillar (0.5y) is weakly identified: ~1e-5bp.
    assert np.max(np.abs(ns.get_rate(ts) - source.get_rate(ts))) < 1e-8


def test_a_hump_whose_best_fit_is_below_its_bound_ends_at_the_bound():
    # Nelson-Siegel data with the hump at 0.1 years: within 0.25-20 the best lambda_
    # is the bound. With the first pillar at 0.5y the objective is nearly flat
    # there, so the fit stays within the 1% start margin of the bound, and its
    # errors are within 0.001bp of the fit exactly on it.
    curve, ts = _curve_of(NelsonSiegel(0.05, -0.015, 0.03, 0.1))
    ns = NelsonSiegel()
    errors = ns.calibrate_from_curve(curve)
    assert 0.25 <= ns.lambda_ <= 0.25 * 1.01 + 1e-12
    pillars = [d for d in curve.knot_dates if d > curve.curve_date]
    target = -np.log(curve.get_dfs(pillars)) / ts
    on_bound = ns._calibration_design(ts, [0.25])
    betas = np.linalg.lstsq(on_bound, target, rcond=None)[0]
    errors_on_bound = 1e4 * (on_bound @ betas - target)
    assert max(abs(e["error_bps"]) for e in errors) <= (
        np.max(np.abs(errors_on_bound)) + 1e-3
    )


def test_a_misspecified_curve_gets_the_global_best_fit_within_the_bounds():
    # Svensson data with the second hump at 40 years, past the 2-20 bounds: the
    # best fit is inside them, and no point of a dense hump grid beats it.
    curve, ts = _curve_of(NelsonSiegelSvensson(0.05, -0.01, 0.01, 0.03, 1.5, 40.0))
    nss = NelsonSiegelSvensson()
    nss.calibrate_from_curve(curve)
    axes = [np.linspace(0.25, 3.0, 56), np.linspace(2.0, 20.0, 181)]
    grid_sse, fit_sse = _dense_grid_best_sse(nss, curve, ts, axes)
    assert fit_sse <= grid_sse * (1 + 1e-9)


def test_a_solver_that_does_not_converge_raises(monkeypatch):
    def failing(fun, x0, **kwargs):
        return SimpleNamespace(success=False, message="max evaluations", x=x0)

    # The package exports the class under the module's name: patch the module.
    monkeypatch.setattr(sys.modules[NelsonSiegel.__module__], "least_squares", failing)
    curve, _ = _curve_of(MODELS[0])
    with pytest.raises(ValueError, match="NelsonSiegel did not converge"):
        NelsonSiegel().calibrate_from_curve(curve)


# ── Calibration errors ──────────────────────────────────────────────────────


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_an_exact_curve_fit_reports_zero_errors_at_each_pillar(model):
    curve, ts = _curve_of(model)
    errors = type(model)().calibrate_from_curve(curve)
    assert [e["term"] for e in errors] == pytest.approx(list(ts), abs=1e-15)
    # ~1e-7bp: the normal equations square the design's condition number.
    assert max(abs(e["error_bps"]) for e in errors) < 1e-5


def test_curve_errors_are_the_fitted_minus_the_curve_zero_rates_in_bps():
    # Svensson data fitted with Nelson-Siegel: real misfits.
    curve, ts = _curve_of(MODELS[1])
    ns = NelsonSiegel()
    errors = ns.calibrate_from_curve(curve)
    pillars = [d for d in curve.knot_dates if d > curve.curve_date]
    observed = -np.log(curve.get_dfs(pillars)) / ts
    expected = 1e4 * (ns.get_rate(ts) - observed)
    assert [e["error_bps"] for e in errors] == pytest.approx(list(expected), abs=1e-9)
    assert max(abs(e["error_bps"]) for e in errors) > 0.1


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_an_exact_bond_fit_reports_zero_errors(model):
    calibration_date, bonds = _coupon_bonds(model)
    errors = type(model)().calibrate_from_bonds(calibration_date, bonds)
    assert len(errors) == len(bonds)
    assert max(abs(e["error_bps"]) for e in errors) < 1e-4


def test_bond_errors_are_irr_differences_at_the_fitted_curve_duration():
    calibration_date, bonds = _coupon_bonds(MODELS[1])
    bonds = bonds[::-1]  # errors come back in the order given
    ns = NelsonSiegel()
    errors = ns.calibrate_from_bonds(calibration_date, bonds)

    curve = ModelCurve(calibration_date, ns, calibration_date + timedelta(days=11500))
    for (bond, irr), error in zip(bonds, errors):
        dates = bond.coupons.get_remaining_end_dates(calibration_date)
        flows = bond.coupons.get_remaining_flows(calibration_date)
        ts = np.array([(d - calibration_date).days / 365 for d in dates])
        pvs = flows * curve.get_dfs(dates)
        assert error["term"] == pytest.approx(float(np.sum(ts * pvs) / np.sum(pvs)), rel=1e-12)
        assert error["term"] == pytest.approx(
            bond.get_duration_from_curve(calibration_date, curve), rel=1e-12
        )
        fitted_irr = bond.get_irr_from_present_value(
            calibration_date, float(np.sum(pvs)), irr.rate_convention
        )
        assert error["error_bps"] == pytest.approx(
            1e4 * (fitted_irr.rate - irr.rate), abs=1e-6
        )
    terms = [e["term"] for e in errors]
    assert terms == sorted(terms, reverse=True)
    assert max(abs(e["error_bps"]) for e in errors) > 0.1


def test_a_zero_coupon_bond_duration_is_its_maturity():
    calibration_date = date(2024, 7, 8)
    bonds = _bonds_irr_list(calibration_date)
    errors = NelsonSiegel().calibrate_from_bonds(calibration_date, bonds)
    maturities = [(b.get_maturity_date() - calibration_date).days / 365 for b, _ in bonds]
    assert [e["term"] for e in errors] == pytest.approx(maturities, rel=1e-12)
