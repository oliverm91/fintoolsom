from datetime import date

import numpy as np
import pytest
from scipy.special import ndtri

from fintoolsom.derivatives.options.options import Call
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.market.volatility_surface import (
    VolatilitySurface,
    VolatilitySurfaceInterpolationMethod as Method,
)
from fintoolsom.market.currencies import Currency, CurrencyPair
from tests.factories.curve_factory import flat_curve


USDCLP = CurrencyPair(Currency.USD, Currency.CLP)

DAYS = [30, 90]
DELTAS = [25, 50, 75]
FLAT_VOLS = [[0.15, 0.15, 0.15], [0.15, 0.15, 0.15]]


def _surface(curve, days=DAYS, deltas=DELTAS, vols=FLAT_VOLS, **kwargs):
    return VolatilitySurface(days, deltas, vols, curve, **kwargs)


def test_volatility_surface_flat_smile_returns_flat_vol(sample_curve):
    vs = _surface(
        sample_curve,
        interpolation_method=Method.DoubleMonotoneCubic,
    )
    vol = vs.get_volatility(0.1, 60)
    assert vol == pytest.approx(0.15, abs=1e-15)


def test_call_mtm_is_positive_for_in_the_money_option(sample_curve):
    spot = 940
    strike = 900
    maturity = date(2024, 10, 12)
    t_val = date(2024, 8, 12)

    vs = _surface(
        sample_curve,
        interpolation_method=Method.DoubleMonotoneCubic,
    )
    call = Call(1000, strike, maturity, currency_pair=USDCLP)
    log_moneyness = Calculator.get_option_log_moneyness(
        call, spot, sample_curve, sample_curve
    )
    vol = vs.get_volatility(log_moneyness, (maturity - t_val).days)
    mtm = Calculator.get_option_mtm(call, t_val, spot, vol, sample_curve, sample_curve)
    assert mtm > 0


def test_rows_are_sorted_by_days_without_touching_the_inputs(sample_curve):
    days = np.array([90, 30])
    vols = np.array([[0.16, 0.15, 0.14], [0.12, 0.11, 0.10]])
    vs = _surface(sample_curve, days=days, vols=vols)
    assert vs.days.tolist() == [30, 90]
    assert vs.vols.tolist() == [[0.12, 0.11, 0.10], [0.16, 0.15, 0.14]]
    assert days.tolist() == [90, 30]  # caller's arrays are not reordered
    assert vols[0].tolist() == [0.16, 0.15, 0.14]


def test_atm_column_has_zero_log_moneyness(sample_curve):
    vs = _surface(
        sample_curve,
        deltas=[25, "atm", 75],
        interpolation_method=Method.DoubleLinear,
    )
    assert vs._interpolation_model.log_moneyness[:, 1].tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(days=[30.0, 90.0]), TypeError),
        (dict(days=[30, 30]), ValueError),
        (dict(days=[0, 90]), ValueError),
        (dict(deltas=[25, "x", 75]), ValueError),
        (dict(deltas=[0.25, "atm", 0.75]), ValueError),
        (dict(vols=[[0.15, 0.15], [0.15, 0.15]]), ValueError),
        (dict(vols=[[0.15, np.nan, 0.15], [0.15, 0.15, 0.15]]), ValueError),
    ],
    ids=[
        "float-days",
        "repeated-days",
        "zero-days",
        "bad-delta",
        "fraction-delta",
        "shape",
        "nan",
    ],
)
def test_invalid_grid_is_rejected(sample_curve, kwargs, error):
    with pytest.raises(error):
        _surface(sample_curve, **kwargs)


def test_pillar_log_moneyness_closed_form(curve_date):
    rate = 0.04
    foreign = flat_curve(curve_date, rate)
    days, deltas = [30, 365, 730], [10, 25, 75]
    vols = np.array([[0.13, 0.12, 0.115], [0.14, 0.13, 0.12], [0.15, 0.14, 0.13]])
    vs = VolatilitySurface(days, deltas, vols, foreign)
    t = np.array(days)[:, None] / 365
    expected = (
        ndtri(np.array(deltas) / 100 / np.exp(-rate * t)) * vols * np.sqrt(t)
        + vols**2 * t / 2
    )
    np.testing.assert_allclose(vs.log_moneyness, expected, rtol=0, atol=1e-14)


def test_put_delta_above_the_foreign_discount_factor_is_rejected(curve_date):
    # 90-delta put at 3Y with an 8% foreign rate: df_q ~ 0.787 < 0.90.
    foreign = flat_curve(curve_date, 0.08)
    with pytest.raises(ValueError, match="unreachable"):
        VolatilitySurface([30, 1095], [25, 90], [[0.1, 0.1], [0.1, 0.1]], foreign)


# ── Calculator.valuate on options (B12) ─────────────────────────────────────


def _option_market(t, domestic, foreign, spot=940.0):
    from fintoolsom.market import FX_Rate, Market, OvernightInterestPriceIndex

    icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
    market = Market(
        t=t,
        risk_free_index=icp,
        curves={icp: domestic, Currency.USD: foreign},
        volatility_surfaces={USDCLP: _surface(foreign)},
    )
    market.add_fx_rate(t, FX_Rate(USDCLP, spot))
    return market


def test_valuate_reports_an_option_in_the_requested_currency(curve_date, sample_curve):
    foreign = flat_curve(curve_date, 0.04)
    market = _option_market(curve_date, sample_curve, foreign)
    call = Call(1000, 930, date(2024, 9, 6), currency_pair=USDCLP)
    in_clp = Calculator.valuate(call, market, Currency.CLP)
    in_usd = Calculator.valuate(call, market, Currency.USD)
    assert in_clp > 0
    assert in_usd == pytest.approx(in_clp / 940.0, rel=1e-12)
    assert (
        Calculator.get_mtm_option(call, market) == in_clp
    )  # quote currency by default


def test_an_option_is_worth_its_intrinsic_value_on_its_maturity_date(sample_curve):
    maturity = date(2024, 9, 6)
    nan = float("nan")  # no volatility is read
    itm, otm = Call(1000, 930, maturity, USDCLP), Call(1000, 950, maturity, USDCLP)
    args = (maturity, 940.0, nan, sample_curve, sample_curve)
    assert Calculator.get_option_mtm(itm, *args) == pytest.approx(10_000.0)
    assert Calculator.get_option_mtm(otm, *args) == 0.0
    assert Calculator.get_option_delta(itm, *args) == 1000
    assert Calculator.get_option_delta(otm, *args) == 0.0
    assert Calculator.get_option_gamma(itm, *args) == 0.0
    assert Calculator.get_option_vega(itm, *args) == 0.0


def test_an_expired_option_is_worth_nothing(curve_date, sample_curve):
    foreign = flat_curve(curve_date, 0.04)
    call = Call(1000, 900, date(2024, 7, 5), currency_pair=USDCLP)  # before t
    market = _option_market(curve_date, sample_curve, foreign)
    assert Calculator.valuate(call, market, Currency.CLP) == 0.0


def test_option_rates_are_continuous_act365_of_the_curves_dfs(curve_date, sample_curve):
    foreign = flat_curve(curve_date, 0.04)
    call = Call(1000, 930, date(2025, 1, 8), currency_pair=USDCLP)
    r, q = Calculator._get_option_rates(call, curve_date, sample_curve, foreign)
    yf = (call.maturity - curve_date).days / 365
    assert np.exp(-r * yf) == pytest.approx(
        sample_curve.get_df(call.maturity), rel=1e-14
    )
    assert np.exp(-q * yf) == pytest.approx(foreign.get_df(call.maturity), rel=1e-14)


def test_options_no_longer_carry_curve_indexes():
    with pytest.raises(TypeError):
        Call(1000, 930, date(2025, 1, 8), USDCLP, domestic_index=None)
