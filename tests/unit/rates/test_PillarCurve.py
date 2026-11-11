import warnings
from dataclasses import fields
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.market import Currency
from fintoolsom.rates import (
    Curve,
    DiscountCurve,
    IndexCurve,
    InterpolationMethod,
    ModelCurve,
    PillarCurve,
)
from fintoolsom.rates.PillarCurve import _warn_piecewise_linear
from tests.factories.quote_factory import ICP, SOFR
from tests.factories.curve_factory import discount_curve_from_dfs

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


@pytest.mark.parametrize("curve", [IndexCurve, DiscountCurve])
def test_roles_are_pillar_curves(curve):
    assert issubclass(curve, PillarCurve) and issubclass(curve, Curve)


def test_model_curve_is_a_curve_and_stays_abstract():
    assert issubclass(ModelCurve, Curve)
    with pytest.raises(TypeError):
        ModelCurve()  # type: ignore[abstract]


def test_the_interpolation_belongs_to_pillar_curves():
    assert not hasattr(Curve, "allowed_methods")
    assert hasattr(PillarCurve, "allowed_methods")


def test_constructor_arguments_keep_their_order():
    assert [f.name for f in fields(IndexCurve)] == [
        "curve_date",
        "precalculate",
        "index",
        "knot_dates",
        "forward_rates",
        "spot_rate",
        "interpolation_method",
    ]
    assert [f.name for f in fields(DiscountCurve)] == [
        "curve_date",
        "precalculate",
        "currency",
        "risk_free_index",
        "knot_dates",
        "forward_rates",
        "interpolation_method",
    ]
    assert next(f for f in fields(IndexCurve) if f.name == "precalculate").kw_only


def test_last_date_is_the_last_knot_of_either_role():
    discount = discount_curve_from_dfs(T, Currency.USD, ICP, PILLARS, DFS)
    index = IndexCurve(T, SOFR, [T, *PILLARS], FORWARDS)
    assert discount.last_date == PILLARS[-1]
    assert index.last_date == PILLARS[-1]


@pytest.mark.parametrize(
    "build",
    [
        lambda m: DiscountCurve(T, Currency.USD, ICP, [T, *PILLARS], FORWARDS, m),
        lambda m: IndexCurve(T, SOFR, [T, *PILLARS], FORWARDS, None, m),
    ],
)
def test_log_linear_is_only_for_the_legacy_curve(build):
    with pytest.raises(ValueError, match="does not support interpolation 'LogLinear'"):
        build(InterpolationMethod.LogLinear)


@pytest.mark.parametrize(
    "discount_method, index_method",
    [
        (InterpolationMethod.PiecewiseConstant, InterpolationMethod.PiecewiseConstant),
        (InterpolationMethod.CubicSpline, InterpolationMethod.CubicSpline),
    ],
)
def test_discount_and_index_curves_on_the_same_knots_agree(
    discount_method, index_method
):
    discount = discount_curve_from_dfs(
        T, Currency.USD, ICP, PILLARS, DFS, discount_method
    )
    index = IndexCurve(T, SOFR, [T, *PILLARS], FORWARDS, None, index_method)
    np.testing.assert_allclose(
        discount.get_dfs(PROBE), index.get_dfs(PROBE), rtol=1e-13
    )


@pytest.mark.parametrize(
    "method",
    [
        InterpolationMethod.PiecewiseConstant,
        InterpolationMethod.HermiteCubicSpline,
        InterpolationMethod.CubicSpline,
    ],
)
def test_discount_curve_returns_its_pillars(method):
    curve = discount_curve_from_dfs(T, Currency.USD, ICP, PILLARS, DFS, method)
    np.testing.assert_allclose(curve.get_dfs(PILLARS), DFS, rtol=1e-14)


def test_a_method_without_an_integral_raises(monkeypatch):
    monkeypatch.setattr(
        IndexCurve, "allowed_methods", frozenset({InterpolationMethod.LogLinear})
    )
    with pytest.raises(ValueError, match="No forward integral defined"):
        IndexCurve(
            T, SOFR, [T, *PILLARS], FORWARDS, None, InterpolationMethod.LogLinear
        )


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize(
    "build",
    [
        lambda: DiscountCurve(T, Currency.USD, ICP, [T, *PILLARS], FORWARDS),
        lambda: IndexCurve(T, SOFR, [T, *PILLARS], FORWARDS),
    ],
)
def test_aging_needs_two_segments_left(build, offset):
    second_to_last = PILLARS[-2] + timedelta(days=offset)
    with pytest.raises(ValueError, match="second-to-last knot"):
        build().get_aged_curve(second_to_last)
    build().get_aged_curve(PILLARS[-2] - timedelta(days=1))


EVERY_METHOD = sorted(PillarCurve.allowed_methods, key=lambda m: m.name)


@pytest.mark.parametrize("method", EVERY_METHOD)
def test_both_roles_accept_every_method_and_agree(method):
    knots = [T, *PILLARS]
    discount = DiscountCurve(T, Currency.USD, ICP, knots, FORWARDS, method)
    index = IndexCurve(T, SOFR, knots, FORWARDS, None, method)
    np.testing.assert_allclose(
        discount.get_dfs(PROBE), index.get_dfs(PROBE), rtol=1e-13
    )


@pytest.mark.parametrize("method", EVERY_METHOD)
def test_the_dfs_at_the_knots_do_not_depend_on_the_method(method):
    curve = IndexCurve(T, SOFR, [T, *PILLARS], FORWARDS, None, method)
    np.testing.assert_allclose(curve.get_dfs(PILLARS), DFS, rtol=1e-13)


def test_linear_forward_is_continuous_and_keeps_the_segment_averages():
    # Forward at the knots: 4%, 4%, 6%. The second segment lasts two years.
    knots = [T, T + timedelta(days=365), T + timedelta(days=1095)]
    curve = IndexCurve(
        T, SOFR, knots, [0.04, 0.05], None, InterpolationMethod.PiecewiseLinear
    )
    assert curve.get_df(T + timedelta(days=730)) == pytest.approx(
        np.exp(-(0.04 + 0.045)), rel=1e-13
    )
    assert curve.get_df(knots[-1]) == pytest.approx(np.exp(-(0.04 + 0.10)), rel=1e-13)


def test_piecewise_linear_warns_once_pointing_to_hagan_and_west():
    _warn_piecewise_linear.cache_clear()
    method = InterpolationMethod.PiecewiseLinear
    with pytest.warns(UserWarning, match="Hagan & West") as record:
        curve = IndexCurve(T, SOFR, [T, *PILLARS], FORWARDS, None, method)
        DiscountCurve(T, Currency.USD, ICP, [T, *PILLARS], FORWARDS, method)
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
        IndexCurve(T, SOFR, [T, *PILLARS], FORWARDS, None, method)


@pytest.mark.parametrize("method", EVERY_METHOD)
@pytest.mark.parametrize(
    "build",
    [
        lambda m: DiscountCurve(T, Currency.USD, ICP, [T, *PILLARS], FORWARDS, m),
        lambda m: IndexCurve(T, SOFR, [T, *PILLARS], FORWARDS, None, m),
    ],
)
def test_aged_dfs_at_the_knots_are_forward_dfs_for_every_method(build, method):
    curve = build(method)
    age = T + timedelta(days=200)
    aged = curve.get_aged_curve(age)
    kept = [p for p in PILLARS if p > age]
    assert type(aged) is type(curve)
    assert aged.curve_date == age and aged.knot_dates == [age, *kept]
    np.testing.assert_allclose(
        aged.get_dfs(kept), curve.get_dfs(kept) / curve.get_df(age), rtol=1e-12
    )
