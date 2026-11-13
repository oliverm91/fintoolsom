"""VolatilitySurface: two-step interpolation, SABR smile and conventions shared
with the option pricer (Calculator / derivatives.calculator._options)."""

from datetime import timedelta

import numpy as np
import pytest
from scipy.special import ndtri

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.options.options import Put
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair
from fintoolsom.market.volatility_surface import (
    DAYS_IN_YEAR,
    DoubleInterpolationLinear,
    DoubleInterpolationMonotoneCubic,
    DoubleInterpolationSABR,
    SABRParameters,
    VolatilitySurface,
    VolatilitySurfaceInterpolationMethod as Method,
    calibrate_sabr,
    eSSVI,
    sabr_volatility,
)
from tests.factories.curve_factory import flat_zero_coupon_curve

USDCLP = CurrencyPair(Currency.USD, Currency.CLP)
Q, R, SPOT = 0.04, 0.06, 940.0  # foreign (USD) rate, domestic (CLP) rate, spot

DAYS = [30, 90, 365]
DELTAS = [10, 25, "atm", 75, 90]
VOLS = np.array(
    [
        [0.135, 0.125, 0.120, 0.118, 0.121],
        [0.140, 0.130, 0.124, 0.121, 0.124],
        [0.150, 0.139, 0.131, 0.127, 0.130],
    ]
)
THREE_DELTAS = [25, "atm", 75]
THREE_VOLS = VOLS[:, [1, 2, 3]]

TWO_STEP = [Method.DoubleLinear, Method.DoubleMonotoneCubic, Method.SABR]
LOCAL_SMILES = [Method.DoubleLinear, Method.DoubleMonotoneCubic]


@pytest.fixture
def foreign(curve_date):
    return flat_zero_coupon_curve(curve_date, Q)


@pytest.fixture
def domestic(curve_date):
    return flat_zero_coupon_curve(curve_date, R)


def _surface(foreign, method=Method.SABR, deltas=DELTAS, vols=VOLS, days=DAYS):
    return VolatilitySurface(days, deltas, vols, foreign, method)


def _closed_form_k(delta_pct, vol, t, df_q):
    return ndtri(delta_pct / 100 / df_q) * vol * np.sqrt(t) + vol**2 * t / 2


# --- method selection ----------------------------------------------------


def test_sabr_is_the_default_method(foreign):
    surface = VolatilitySurface(DAYS, DELTAS, VOLS, foreign)
    assert surface.interpolation_method is Method.SABR
    assert isinstance(surface._interpolation_model, DoubleInterpolationSABR)


@pytest.mark.parametrize(
    "method, model",
    [
        (Method.DoubleLinear, DoubleInterpolationLinear),
        (Method.DoubleMonotoneCubic, DoubleInterpolationMonotoneCubic),
        (Method.SABR, DoubleInterpolationSABR),
    ],
)
def test_each_method_builds_its_model(foreign, method, model):
    assert isinstance(_surface(foreign, method)._interpolation_model, model)


@pytest.mark.parametrize(
    "method, deltas, vols",
    [
        (Method.SABR, [25, 75], VOLS[:, [1, 3]]),
        (Method.DoubleMonotoneCubic, [25], VOLS[:, [1]]),
    ],
    ids=["sabr-two-columns", "monotone-cubic-one-column"],
)
def test_too_few_delta_columns_are_rejected(foreign, method, deltas, vols):
    with pytest.raises(ValueError, match="at least"):
        _surface(foreign, method, deltas, vols)


# --- step 1: total variance in time ---------------------------------------


@pytest.mark.parametrize("method", TWO_STEP)
def test_time_step_is_linear_in_total_variance(foreign, method):
    model = _surface(foreign, method)._interpolation_model
    t0, t1, t = 30 / 365, 90 / 365, 60 / 365
    w0, w1 = VOLS[0] ** 2 * t0, VOLS[1] ** 2 * t1
    expected = np.sqrt((w0 + (w1 - w0) * (t - t0) / (t1 - t0)) / t)
    np.testing.assert_allclose(model.get_smile_t(60), expected, rtol=0, atol=1e-15)


def test_worked_example_atm_vol_at_60_days(foreign):
    # 12% at 30d and 13% at 90d: w halfway is 0.0026753, sqrt(w / (60/365)) = 12.757%
    # (linear in vol would give 12.5%).
    vols = np.array([[0.12, 0.12, 0.12], [0.13, 0.13, 0.13]])
    model = _surface(foreign, Method.DoubleLinear, THREE_DELTAS, vols, [30, 90])
    model = model._interpolation_model
    assert model.get_smile_t(60)[1] == pytest.approx(0.1275735, abs=1e-7)


def test_total_variance_never_decreases_in_time(foreign):
    model = _surface(foreign, Method.DoubleLinear)._interpolation_model
    days = np.arange(1, 731)
    w = np.array([model.get_smile_t(d) ** 2 * d / 365 for d in days])
    assert (np.diff(w, axis=0) >= -1e-15).all()


def test_vol_is_flat_before_the_first_and_after_the_last_tenor(foreign):
    model = _surface(foreign, Method.DoubleLinear)._interpolation_model
    np.testing.assert_array_equal(model.get_smile_t(7), VOLS[0])
    np.testing.assert_array_equal(model.get_smile_t(730), VOLS[-1])


def test_non_positive_days_are_rejected(foreign):
    with pytest.raises(ValueError, match="positive"):
        _surface(foreign, Method.DoubleLinear).get_volatility(0.0, 0)


# --- step 2: strikes recomputed at the target tenor -----------------------


@pytest.mark.parametrize("days", [7, 60, 200, 730])
def test_log_moneyness_is_recomputed_from_delta_at_the_target_tenor(foreign, days):
    model = _surface(foreign, Method.DoubleLinear)._interpolation_model
    t = days / 365
    vols = model.get_smile_t(days)
    expected = [
        0.0 if d == "atm" else _closed_form_k(d, v, t, np.exp(-Q * t))
        for d, v in zip(DELTAS, vols)
    ]
    np.testing.assert_allclose(
        model.get_smile_log_moneynesses(days), expected, rtol=0, atol=1e-14
    )


def test_recomputed_strikes_differ_from_interpolated_pillar_strikes(foreign):
    model = _surface(foreign, Method.DoubleLinear)._interpolation_model
    pillar_k_interp = [np.interp(60, DAYS, model.log_moneyness[:, j]) for j in [0, 4]]
    recomputed = model.get_smile_log_moneynesses(60)[[0, 4]]
    assert np.abs(recomputed - pillar_k_interp).max() > 1e-4


@pytest.mark.parametrize("t", [5 / 365, 0.5, 2.0])
def test_foreign_log_df_is_exact_for_a_flat_rate(foreign, t):
    model = _surface(foreign, Method.DoubleLinear)._interpolation_model
    assert model._foreign_log_df(t) == pytest.approx(-Q * t, abs=1e-13)


# --- step 3: smiles --------------------------------------------------------


@pytest.mark.parametrize("method", LOCAL_SMILES)
def test_local_smiles_reproduce_every_quote(foreign, method):
    surface = _surface(foreign, method)
    for i, days in enumerate(DAYS):
        vols = surface.get_volatility(surface.log_moneyness[i], days)
        np.testing.assert_allclose(vols, VOLS[i], rtol=0, atol=1e-14)


@pytest.mark.parametrize("method", LOCAL_SMILES)
def test_local_smiles_are_flat_beyond_the_outer_nodes(foreign, method):
    surface = _surface(foreign, method)
    k_nodes, vol_nodes = surface._interpolation_model.get_smile_nodes(120)
    assert surface.get_volatility(k_nodes[0] - 1.0, 120) == vol_nodes[0]
    assert surface.get_volatility(k_nodes[-1] + 1.0, 120) == vol_nodes[-1]


def test_monotone_cubic_does_not_overshoot_between_nodes(foreign):
    surface = _surface(foreign, Method.DoubleMonotoneCubic)
    k_nodes, vol_nodes = surface._interpolation_model.get_smile_nodes(90)
    for a in range(len(k_nodes) - 1):
        k = np.linspace(k_nodes[a], k_nodes[a + 1], 50)
        vols = surface.get_volatility(k, 90)
        lo, hi = sorted(vol_nodes[a : a + 2])
        assert (vols >= lo - 1e-15).all() and (vols <= hi + 1e-15).all()


@pytest.mark.parametrize(
    "method, atol",
    [
        (Method.DoubleLinear, 1e-15),
        (Method.DoubleMonotoneCubic, 1e-15),
        (Method.SABR, 1e-6),
    ],
)
def test_flat_quotes_give_a_flat_surface(foreign, method, atol):
    # A flat smile is degenerate for SABR (vol depends on nu^2 near nu = 0), so the
    # fit stops at a tiny nu: ~1e-9 at the nodes, up to ~2e-7 far in a 10-day wing.
    surface = _surface(foreign, method, vols=np.full((3, 5), 0.15))
    for days in (10, 60, 365, 900):
        vols = surface.get_volatility(np.linspace(-0.3, 0.3, 7), days)
        np.testing.assert_allclose(vols, 0.15, rtol=0, atol=atol)


# --- SABR -------------------------------------------------------------------


def test_sabr_smile_reproduces_three_quotes_at_every_tenor(foreign):
    surface = _surface(foreign, Method.SABR, THREE_DELTAS, THREE_VOLS)
    for i, days in enumerate(DAYS):
        vols = surface.get_volatility(surface.log_moneyness[i], days)
        np.testing.assert_allclose(vols, THREE_VOLS[i], rtol=0, atol=1e-10)
    k_nodes, vol_nodes = surface._interpolation_model.get_smile_nodes(200)
    vols = surface.get_volatility(k_nodes, 200)
    np.testing.assert_allclose(vols, vol_nodes, rtol=0, atol=1e-10)


def _sabr_quotes(params, t, df_q, deltas=(10, 25, 75, 90)):
    """Vols that a SABR smile gives at the strikes of `deltas` (fixed point)."""
    vols = np.full(len(deltas), params.alpha)
    for _ in range(200):
        k = np.array([_closed_form_k(d, v, t, df_q) for d, v in zip(deltas, vols)])
        vols = sabr_volatility(k, t, params)
    k = np.array([_closed_form_k(d, v, t, df_q) for d, v in zip(deltas, vols)])
    return np.concatenate((k, [0.0])), np.concatenate(
        (vols, [sabr_volatility(0.0, t, params)])
    )


def test_calibrate_sabr_recovers_the_parameters_of_a_sabr_smile():
    truth = SABRParameters(alpha=0.12, rho=0.3, nu=0.8)
    t = 0.5
    k, vols = _sabr_quotes(truth, t, np.exp(-Q * t))
    order = np.argsort(k)
    fitted = calibrate_sabr(k[order], vols[order], t)
    assert fitted.alpha == pytest.approx(truth.alpha, abs=1e-8)
    assert fitted.rho == pytest.approx(truth.rho, abs=1e-6)
    assert fitted.nu == pytest.approx(truth.nu, abs=1e-6)


def test_sabr_positive_rho_lifts_the_high_strike_side():
    params = SABRParameters(alpha=0.12, rho=0.4, nu=0.6)
    assert sabr_volatility(0.1, 0.5, params) > sabr_volatility(-0.1, 0.5, params)


def test_sabr_volatility_is_continuous_at_the_money():
    params = SABRParameters(alpha=0.12, rho=0.3, nu=0.8)
    at_k0 = sabr_volatility(0.0, 0.5, params)
    expected = 0.12 * (1 + (0.3 * 0.8 * 0.12 / 4 + (2 - 3 * 0.09) * 0.64 / 24) * 0.5)
    assert at_k0 == pytest.approx(expected, abs=1e-15)
    assert sabr_volatility(1e-9, 0.5, params) == pytest.approx(at_k0, abs=1e-9)
    assert sabr_volatility(-1e-9, 0.5, params) == pytest.approx(at_k0, abs=1e-9)


def test_sabr_parameters_are_exposed_per_tenor(foreign):
    model = _surface(foreign, Method.SABR)._interpolation_model
    params = model.get_sabr_parameters(90)
    assert params.alpha > 0 and -1 < params.rho < 1 and params.nu > 0


def test_sabr_smiles_are_cached_by_days(foreign):
    surface = _surface(foreign, Method.SABR)
    surface.get_volatility(0.0, 120)
    surface.get_volatility(0.05, 120)
    assert list(surface._interpolation_model._smiles) == [120]


# --- eSSVI on ACT/365 ----------------------------------------------------------


def test_essvi_uses_act_365(foreign, capsys):
    flat = np.full((3, 3), 0.15)
    surface = _surface(foreign, Method.eSSVI, THREE_DELTAS, flat)
    model = surface._interpolation_model
    assert isinstance(model, eSSVI)
    for days in DAYS:
        assert surface.get_volatility(0.0, days) == pytest.approx(0.15, abs=1e-3)
    capsys.readouterr()  # calibration prints its progress


# --- conventions shared with the option pricer ------------------------------


def _put(strike, maturity):
    return Put(1.0, strike, maturity, currency_pair=USDCLP)


@pytest.mark.parametrize("row", range(len(DAYS)))
@pytest.mark.parametrize("column", [0, 1, 3, 4])
def test_pillar_strike_round_trips_through_the_option_pricer(
    curve_date, domestic, foreign, row, column
):
    """The pillar k is ln(K/F) of the pricer's strike for that spot put delta, and
    the pricer's delta at that strike is the quoted delta."""
    surface = _surface(foreign, Method.DoubleLinear)
    delta, vol = DELTAS[column] / 100, VOLS[row, column]
    maturity = curve_date + timedelta(days=DAYS[row])
    strike = Calculator.get_put_strike_from_delta(
        -delta, SPOT, vol, domestic, foreign, maturity
    )
    put = _put(strike, maturity)
    k = Calculator.get_option_log_moneyness(put, SPOT, domestic, foreign)
    assert k == pytest.approx(surface.log_moneyness[row, column], abs=1e-12)
    pricer_delta = Calculator.get_option_delta(
        put, curve_date, SPOT, vol, domestic, foreign
    )
    assert pricer_delta == pytest.approx(-delta, abs=1e-12)


def test_atm_column_is_the_option_pricer_forward(curve_date, domestic, foreign):
    maturity = curve_date + timedelta(days=90)
    forward = SPOT * foreign.get_df(maturity) / domestic.get_df(maturity)
    k = Calculator.get_option_log_moneyness(
        _put(forward, maturity), SPOT, domestic, foreign
    )
    assert k == pytest.approx(0.0, abs=1e-15)
    assert (_surface(foreign, Method.DoubleLinear).log_moneyness[:, 2] == 0).all()


@pytest.mark.parametrize("days", DAYS)
def test_surface_time_is_the_option_pricer_year_fraction(curve_date, days):
    put = _put(SPOT, curve_date + timedelta(days=days))
    assert Calculator._get_option_yf(put, curve_date) == days / DAYS_IN_YEAR
