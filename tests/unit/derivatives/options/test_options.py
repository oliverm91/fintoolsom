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
from tests.factories.curve_factory import flat_zero_coupon_curve


USDCLP = CurrencyPair(Currency.USD, Currency.CLP)

DAYS = [30, 90]
DELTAS = [25, 50, 75]
FLAT_VOLS = [[0.15, 0.15, 0.15], [0.15, 0.15, 0.15]]


def _surface(curve, days=DAYS, deltas=DELTAS, vols=FLAT_VOLS, **kwargs):
    return VolatilitySurface(days, deltas, vols, curve, **kwargs)


def test_volatility_surface_flat_smile_returns_flat_vol(sample_zero_coupon_curve):
    vs = _surface(
        sample_zero_coupon_curve,
        interpolation_method=Method.DoubleMonotoneCubic,
    )
    vol = vs.get_volatility(0.1, 60)
    assert vol == pytest.approx(0.15, abs=1e-15)


def test_call_mtm_is_positive_for_in_the_money_option(sample_zero_coupon_curve):
    spot = 940
    strike = 900
    maturity = date(2024, 10, 12)
    t_val = date(2024, 8, 12)

    vs = _surface(
        sample_zero_coupon_curve,
        interpolation_method=Method.DoubleMonotoneCubic,
    )
    call = Call(1000, strike, maturity, currency_pair=USDCLP)
    log_moneyness = Calculator.get_option_log_moneyness(
        call, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    vol = vs.get_volatility(log_moneyness, (maturity - t_val).days)
    mtm = Calculator.get_option_mtm(
        call, t_val, spot, vol, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert mtm > 0


def test_rows_are_sorted_by_days_without_touching_the_inputs(sample_zero_coupon_curve):
    days = np.array([90, 30])
    vols = np.array([[0.16, 0.15, 0.14], [0.12, 0.11, 0.10]])
    vs = _surface(sample_zero_coupon_curve, days=days, vols=vols)
    assert vs.days.tolist() == [30, 90]
    assert vs.vols.tolist() == [[0.12, 0.11, 0.10], [0.16, 0.15, 0.14]]
    assert days.tolist() == [90, 30]  # caller's arrays are not reordered
    assert vols[0].tolist() == [0.16, 0.15, 0.14]


def test_atm_column_has_zero_log_moneyness(sample_zero_coupon_curve):
    vs = _surface(
        sample_zero_coupon_curve,
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
def test_invalid_grid_is_rejected(sample_zero_coupon_curve, kwargs, error):
    with pytest.raises(error):
        _surface(sample_zero_coupon_curve, **kwargs)


def test_pillar_log_moneyness_closed_form(curve_date):
    rate = 0.04
    foreign = flat_zero_coupon_curve(curve_date, rate)
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
    foreign = flat_zero_coupon_curve(curve_date, 0.08)
    with pytest.raises(ValueError, match="unreachable"):
        VolatilitySurface([30, 1095], [25, 90], [[0.1, 0.1], [0.1, 0.1]], foreign)
