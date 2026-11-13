"""VolatilitySurface: two-step interpolation, vanna-volga and SVI smiles, and the
conventions shared with the option pricer (Calculator /
derivatives.calculator._options)."""

from datetime import timedelta

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.options.options import Put
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair
from fintoolsom.market.volatility_surface import (
    DAYS_IN_YEAR,
    DoubleInterpolationLinear,
    DoubleInterpolationMonotoneCubic,
    DoubleInterpolationSVI,
    DoubleInterpolationVannaVolga,
    SVIParameters,
    VolatilitySurface,
    VolatilitySurfaceInterpolationMethod as Method,
    _vanna_volga_anchors,
    calibrate_svi,
    eSSVI,
    svi_total_variance,
    svi_volatility,
    vanna_volga_volatility,
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

TWO_STEP = [
    Method.DoubleLinear,
    Method.DoubleMonotoneCubic,
    Method.VannaVolga,
    Method.SVI,
]
INTERPOLATING = [Method.DoubleLinear, Method.DoubleMonotoneCubic, Method.VannaVolga]


@pytest.fixture
def foreign(curve_date):
    return flat_zero_coupon_curve(curve_date, Q)


@pytest.fixture
def domestic(curve_date):
    return flat_zero_coupon_curve(curve_date, R)


def _surface(foreign, method=Method.VannaVolga, deltas=DELTAS, vols=VOLS, days=DAYS):
    return VolatilitySurface(days, deltas, vols, foreign, method)


def _closed_form_k(delta_pct, vol, t, df_q):
    return ndtri(delta_pct / 100 / df_q) * vol * np.sqrt(t) + vol**2 * t / 2


# --- method selection ----------------------------------------------------


def test_vanna_volga_is_the_default_method(foreign):
    surface = VolatilitySurface(DAYS, DELTAS, VOLS, foreign)
    assert surface.interpolation_method is Method.VannaVolga
    assert isinstance(surface._interpolation_model, DoubleInterpolationVannaVolga)


@pytest.mark.parametrize(
    "method, model",
    [
        (Method.DoubleLinear, DoubleInterpolationLinear),
        (Method.DoubleMonotoneCubic, DoubleInterpolationMonotoneCubic),
        (Method.VannaVolga, DoubleInterpolationVannaVolga),
        (Method.SVI, DoubleInterpolationSVI),
    ],
)
def test_each_method_builds_its_model(foreign, method, model):
    assert isinstance(_surface(foreign, method)._interpolation_model, model)


@pytest.mark.parametrize(
    "method, deltas, vols",
    [
        (Method.VannaVolga, [25, 75], VOLS[:, [1, 3]]),
        (Method.SVI, [10, 25, 75, 90], VOLS[:, [0, 1, 3, 4]]),
        (Method.DoubleMonotoneCubic, [25], VOLS[:, [1]]),
    ],
    ids=["vanna-volga-two-columns", "svi-four-columns", "monotone-cubic-one-column"],
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


@pytest.mark.parametrize("deltas, vols", [(DELTAS, VOLS), (THREE_DELTAS, THREE_VOLS)])
@pytest.mark.parametrize("method", INTERPOLATING)
def test_interpolating_smiles_reproduce_every_quote(foreign, method, deltas, vols):
    surface = _surface(foreign, method, deltas, vols)
    for i, days in enumerate(DAYS):
        got = surface.get_volatility(surface.log_moneyness[i], days)
        np.testing.assert_allclose(got, vols[i], rtol=0, atol=1e-14)
    k_nodes, vol_nodes = surface._interpolation_model.get_smile_nodes(200)
    np.testing.assert_allclose(
        surface.get_volatility(k_nodes, 200), vol_nodes, rtol=0, atol=1e-14
    )


@pytest.mark.parametrize("method", INTERPOLATING)
def test_interpolating_smiles_are_flat_beyond_the_outer_nodes(foreign, method):
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
        (Method.VannaVolga, 1e-15),
        (Method.SVI, 1e-8),
    ],
)
def test_flat_quotes_give_a_flat_surface(foreign, method, atol):
    # SVI reaches a flat smile at its slope bound b = 0, to ~1e-9.
    surface = _surface(foreign, method, vols=np.full((3, 5), 0.15))
    for days in (10, 60, 365, 900):
        vols = surface.get_volatility(np.linspace(-0.3, 0.3, 7), days)
        np.testing.assert_allclose(vols, 0.15, rtol=0, atol=atol)


@pytest.mark.parametrize("method", TWO_STEP)
def test_smiles_are_cached_by_days(foreign, method):
    surface = _surface(foreign, method)
    surface.get_volatility(0.0, 120)
    surface.get_volatility(0.05, 120)
    assert list(surface._interpolation_model._smiles) == [120]


# --- vanna-volga ----------------------------------------------------------------

ANCHOR_K = np.array([-0.06, 0.0, 0.055])
ANCHOR_VOLS = np.array([0.130, 0.120, 0.125])


def _black_call(k, vol, t):
    """Undiscounted Black call on a unit forward, at log-moneyness k."""
    std = vol * np.sqrt(t)
    d1 = (-k + std**2 / 2) / std
    return ndtr(d1) - np.exp(k) * ndtr(d1 - std)


def _black_vega(k, vol, t):
    std = vol * np.sqrt(t)
    d1 = (-k + std**2 / 2) / std
    return np.sqrt(t) * np.exp(-(d1**2) / 2) / np.sqrt(2 * np.pi)


def _vanna_volga_price_vol(k, t, ka, va):
    """Implied vol of the vanna-volga price (Castagna and Mercurio, 2007): Black at
    the ATM vol plus each anchor's smile premium, weighted by vega(K) / vega(K_i)
    times the log-strike Lagrange weight y_i(K)."""
    k1, k2, k3 = ka
    y = (
        (k2 - k) * (k3 - k) / ((k2 - k1) * (k3 - k1)),
        (k - k1) * (k3 - k) / ((k2 - k1) * (k3 - k2)),
        (k - k1) * (k - k2) / ((k3 - k1) * (k3 - k2)),
    )
    atm = va[1]
    price = _black_call(k, atm, t) + sum(
        _black_vega(k, atm, t)
        / _black_vega(k_i, atm, t)
        * y_i
        * (_black_call(k_i, v_i, t) - _black_call(k_i, atm, t))
        for y_i, k_i, v_i in zip(y, ka, va)
    )
    return brentq(lambda vol: _black_call(k, vol, t) - price, 1e-4, 3.0, xtol=1e-15)


def test_vanna_volga_formula_matches_the_vanna_volga_price():
    # The closed form is the second-order expansion of the price's implied vol:
    # exact at the anchors, within ~5e-5 between and just beyond them.
    t = 0.25
    k = np.linspace(-0.08, 0.08, 17)
    expected = [_vanna_volga_price_vol(x, t, ANCHOR_K, ANCHOR_VOLS) for x in k]
    got = vanna_volga_volatility(k, t, ANCHOR_K, ANCHOR_VOLS)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-4)
    at_anchors = vanna_volga_volatility(ANCHOR_K, t, ANCHOR_K, ANCHOR_VOLS)
    np.testing.assert_allclose(at_anchors, ANCHOR_VOLS, rtol=0, atol=1e-15)


def test_vanna_volga_does_not_depend_on_the_anchor_order():
    k = np.linspace(-0.1, 0.1, 9)
    base = vanna_volga_volatility(k, 0.25, ANCHOR_K, ANCHOR_VOLS, pivot=1)
    order = [2, 0, 1]  # the ATM anchor moves to position 2
    permuted = vanna_volga_volatility(
        k, 0.25, ANCHOR_K[order], ANCHOR_VOLS[order], pivot=2
    )
    np.testing.assert_allclose(permuted, base, rtol=0, atol=1e-15)


def test_vanna_volga_is_continuous_where_d1_d2_vanishes():
    t = 0.25
    k0 = 0.12**2 * t / 2  # d1 = 0 at the ATM vol
    at_k0 = vanna_volga_volatility(k0, t, ANCHOR_K, ANCHOR_VOLS)
    for dk in (-1e-9, 1e-9):
        near = vanna_volga_volatility(k0 + dk, t, ANCHOR_K, ANCHOR_VOLS)
        assert near == pytest.approx(at_k0, abs=1e-10)


@pytest.mark.parametrize(
    "pivot, n_nodes, expected",
    [
        (2, 5, [[0, 1, 2], [1, 2, 3], [1, 2, 3], [2, 3, 4]]),
        (1, 3, [[0, 1, 2], [0, 1, 2]]),
        (0, 3, [[0, 1, 2], [0, 1, 2]]),
        (0, 4, [[0, 1, 2], [0, 1, 2], [0, 2, 3]]),
        (3, 4, [[0, 1, 3], [1, 2, 3], [1, 2, 3]]),
    ],
)
def test_vanna_volga_anchors_always_include_the_pivot(pivot, n_nodes, expected):
    anchors = [_vanna_volga_anchors(j, pivot, n_nodes) for j in range(n_nodes - 1)]
    assert anchors == expected


def test_vanna_volga_inner_smile_uses_the_25_delta_pair_only(foreign):
    five = _surface(foreign, Method.VannaVolga)
    three = _surface(foreign, Method.VannaVolga, THREE_DELTAS, THREE_VOLS)
    k_nodes, _ = five._interpolation_model.get_smile_nodes(120)
    k = np.linspace(k_nodes[1], k_nodes[3], 25)
    np.testing.assert_allclose(
        five.get_volatility(k, 120), three.get_volatility(k, 120), rtol=0, atol=1e-15
    )


def test_vanna_volga_five_quote_smile_is_continuous_at_the_nodes(foreign):
    surface = _surface(foreign, Method.VannaVolga)
    k_nodes, vol_nodes = surface._interpolation_model.get_smile_nodes(90)
    for k, vol in zip(k_nodes[1:-1], vol_nodes[1:-1]):
        sides = surface.get_volatility(np.array([k - 1e-10, k + 1e-10]), 90)
        np.testing.assert_allclose(sides, vol, rtol=0, atol=1e-9)


# --- SVI -----------------------------------------------------------------------

SVI_TRUTH = SVIParameters(a=0.002, b=0.04, rho=-0.3, m=0.01, sigma=0.08)


def _svi_truth(t):
    """An SVI smile with ~12.4% ATM vol whose total variance grows with t."""
    return SVIParameters(
        a=0.011 * t, b=0.03 * np.sqrt(t), rho=-0.25, m=0.0, sigma=0.15 * np.sqrt(t)
    )


def _svi_quotes(truth_by_days):
    """DELTAS quotes lying on the given SVI smile at each tenor (fixed point on k)."""
    rows = []
    for days, params in truth_by_days.items():
        t = days / 365
        df_q = np.exp(-Q * t)
        vols = np.full(len(DELTAS), 0.12)
        for _ in range(200):
            k = np.array(
                [
                    0.0 if d == "atm" else _closed_form_k(d, v, t, df_q)
                    for d, v in zip(DELTAS, vols)
                ]
            )
            vols = svi_volatility(k, t, params)
        rows.append(vols)
    return np.array(rows)


def test_calibrate_svi_recovers_the_parameters_of_an_svi_smile():
    t = 0.5
    k = np.array([-0.15, -0.06, 0.0, 0.05, 0.12])
    fitted = calibrate_svi(k, svi_volatility(k, t, SVI_TRUTH), t)
    for name in ("a", "b", "rho", "m", "sigma"):
        assert getattr(fitted, name) == pytest.approx(
            getattr(SVI_TRUTH, name), abs=1e-8
        )


def test_svi_smile_reproduces_svi_shaped_quotes_at_every_tenor(foreign):
    truth = {days: _svi_truth(days / 365) for days in DAYS}
    vols = _svi_quotes(truth)
    surface = _surface(foreign, Method.SVI, vols=vols)
    k = np.linspace(-0.3, 0.3, 13)
    for i, days in enumerate(DAYS):
        got = surface.get_volatility(surface.log_moneyness[i], days)
        np.testing.assert_allclose(got, vols[i], rtol=0, atol=1e-10)
        expected = svi_volatility(k, days / 365, truth[days])
        np.testing.assert_allclose(
            surface.get_volatility(k, days), expected, rtol=0, atol=1e-8
        )


def test_svi_fits_non_svi_quotes_in_the_least_squares_sense(foreign):
    # The 30d and 90d rows have a curvature that dips at the money, which no raw
    # SVI smile has: the fit misses them by a few vol basis points.
    surface = _surface(foreign, Method.SVI)
    for i, days in enumerate(DAYS):
        got = surface.get_volatility(surface.log_moneyness[i], days)
        np.testing.assert_allclose(got, VOLS[i], rtol=0, atol=5e-4)


@pytest.mark.parametrize("days", [10, 30, 90, 200, 365, 900])
def test_svi_fit_respects_lee_bound_and_non_negative_variance(foreign, days):
    p = _surface(foreign, Method.SVI)._interpolation_model.get_svi_parameters(days)
    assert p.b >= 0 and -1 <= p.rho <= 1 and p.sigma > 0
    assert p.b * (1 + abs(p.rho)) <= 2 + 1e-12
    assert p.a + p.b * p.sigma * np.sqrt(1 - p.rho**2) >= -1e-15


def test_svi_wings_are_linear_in_total_variance():
    def slope(k):
        h = 1e-3
        return (
            svi_total_variance(k + h, SVI_TRUTH) - svi_total_variance(k, SVI_TRUTH)
        ) / h

    b, rho = SVI_TRUTH.b, SVI_TRUTH.rho
    assert slope(50.0) == pytest.approx(b * (1 + rho), rel=1e-5)
    assert slope(-50.0) == pytest.approx(-b * (1 - rho), rel=1e-5)


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
