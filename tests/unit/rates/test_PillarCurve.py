import warnings
from dataclasses import fields
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.rates import (
    Curve,
    PillarCurve,
    InterpolationMethod,
    ModelCurve,
)
from fintoolsom.rates.PillarCurve import _warn_piecewise_linear
from tests.factories.curve_factory import curve_from_dfs

T = date(2026, 6, 29)
PILLARS = [T + timedelta(days=d) for d in (1, 32, 95, 185, 370, 735, 1830, 3655)]
ZEROS = np.array([0.051, 0.0505, 0.05, 0.049, 0.047, 0.044, 0.042, 0.041])
TAU = np.array([(d - T).days / 365 for d in PILLARS])
DFS = np.exp(-ZEROS * TAU)
FORWARDS = np.diff(np.concatenate(([0.0], ZEROS * TAU))) / np.diff(
    np.concatenate(([0.0], TAU))
)
# Inside the pillars and past the last one.
PROBE = [T + timedelta(days=d) for d in range(0, 5000, 7)]


def test_pillar_curve_is_a_concrete_curve():
    assert issubclass(PillarCurve, Curve) and not PillarCurve.__abstractmethods__


def test_model_curve_is_a_curve_and_stays_abstract():
    assert issubclass(ModelCurve, Curve)
    with pytest.raises(TypeError):
        ModelCurve()  # type: ignore[abstract]


def test_the_interpolation_belongs_to_pillar_curves():
    assert not hasattr(Curve, "allowed_methods")
    assert hasattr(PillarCurve, "allowed_methods")


def test_constructor_arguments_keep_their_order():
    """The pillar state, then the optional spot rate; no identity: the Market key
    gives the curve its role."""
    assert [f.name for f in fields(PillarCurve)] == [
        "curve_date",
        "knot_dates",
        "forward_rates",
        "interpolation_method",
        "spot_rate",
        "precalculate",
    ]
    assert next(f for f in fields(PillarCurve) if f.name == "precalculate").kw_only


def test_last_date_is_the_last_knot_of_either_role():
    discount = curve_from_dfs(T, PILLARS, DFS)
    index = PillarCurve(
        T,
        [T, *PILLARS],
        FORWARDS,
        InterpolationMethod.PiecewiseConstant,
    )
    assert discount.last_date == PILLARS[-1]
    assert index.last_date == PILLARS[-1]


@pytest.mark.parametrize(
    "method",
    [
        InterpolationMethod.PiecewiseConstant,
        InterpolationMethod.HermiteCubicSpline,
        InterpolationMethod.CubicSpline,
    ],
)
def test_discount_curve_returns_its_pillars(method):
    curve = curve_from_dfs(T, PILLARS, DFS, method)
    np.testing.assert_allclose(curve.get_dfs(PILLARS), DFS, rtol=1e-14)


def test_a_method_without_an_integral_raises(monkeypatch):
    monkeypatch.setattr(
        PillarCurve, "allowed_methods", frozenset({InterpolationMethod.LogLinear})
    )
    with pytest.raises(ValueError, match="No forward integral defined"):
        PillarCurve(
            T,
            [T, *PILLARS],
            FORWARDS,
            InterpolationMethod.LogLinear,
        )


EVERY_METHOD = sorted(PillarCurve.allowed_methods, key=lambda m: m.name)


def test_linear_forward_is_continuous_and_keeps_the_segment_averages():
    # Forward at the knots: 4%, 4%, 6%. The second segment lasts two years.
    knots = [T, T + timedelta(days=365), T + timedelta(days=1095)]
    curve = PillarCurve(
        T,
        knots,
        [0.04, 0.05],
        InterpolationMethod.PiecewiseLinear,
    )
    assert curve.get_df(T + timedelta(days=730)) == pytest.approx(
        np.exp(-(0.04 + 0.045)), rel=1e-13
    )
    assert curve.get_df(knots[-1]) == pytest.approx(np.exp(-(0.04 + 0.10)), rel=1e-13)


def test_piecewise_linear_warns_once_pointing_to_hagan_and_west():
    _warn_piecewise_linear.cache_clear()
    method = InterpolationMethod.PiecewiseLinear
    with pytest.warns(UserWarning, match="Hagan & West") as record:
        curve = PillarCurve(T, [T, *PILLARS], FORWARDS, method)
        PillarCurve(T, [T, *PILLARS], FORWARDS, method)
        curve.get_aged_curve(T + timedelta(days=200))
    assert len(record) == 1


@pytest.mark.parametrize(
    "method",
    [m for m in EVERY_METHOD if m is not InterpolationMethod.PiecewiseLinear],
)
def test_other_methods_do_not_warn(method):
    _warn_piecewise_linear.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        PillarCurve(T, [T, *PILLARS], FORWARDS, method)
