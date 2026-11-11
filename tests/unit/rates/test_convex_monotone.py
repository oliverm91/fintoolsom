from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.rates import DiscountCurve, IndexCurve, InterpolationMethod
from fintoolsom.market import Currency
from fintoolsom.rates._convex_monotone import (
    _segment_polynomials,
    convex_monotone_forward,
)
from tests.factories.quote_factory import ICP, SOFR

T = date(2026, 6, 29)
METHOD = InterpolationMethod.ConvexMonotone

# From QuantLib 1.44, PiecewiseConvexMonotoneForward with
# ql.ConvexMonotone(0.3, 0.7, False): knot days, segment averages, {day: df}.
QUANTLIB = {
    "jagged": (
        [100, 200, 300, 400, 500, 600, 700],
        [0.05, 0.02, 0.09, 0.01, 0.07, 0.03, 0.08],
        {
            37: 0.9942916691727072,
            150: 0.9837922730827826,
            249: 0.969260737762024,
            333: 0.9555115560836563,
            450: 0.9454136959703826,
            555: 0.9322910189368669,
            690: 0.9108659040026653,
        },
    ),
    "hump": (
        [60, 120, 240, 480, 960, 1920],
        [0.03, 0.05, 0.08, 0.075, 0.04, 0.035],
        {
            21: 0.9985274893188943,
            90: 0.991406995748924,
            201: 0.9696541910102342,
            333: 0.9422707853451178,
            702: 0.8911645611272256,
            1500: 0.8230803827418011,
        },
    ),
    "negative": (
        [90, 180, 360, 720, 1440],
        [-0.01, -0.004, 0.002, 0.01, 0.015],
        {
            45: 1.0013725163821128,
            135: 1.0031178641576504,
            270: 1.003375547385455,
            540: 0.9983388059703054,
            1000: 0.9825094653771413,
        },
    ),
}
DAYS = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
TAU = np.concatenate(([0.0], DAYS / 365))
AVERAGES = np.array([0.05, 0.02, 0.09, 0.01, 0.07, 0.03, 0.08, 0.04, 0.06])


def _index_curve(days, averages):
    knots = [T, *(T + timedelta(days=d) for d in days)]
    return IndexCurve(T, SOFR, knots, averages, None, METHOD)


@pytest.mark.parametrize("case", QUANTLIB)
def test_discount_factors_match_quantlib(case):
    days, averages, expected = QUANTLIB[case]
    curve = _index_curve(days, averages)
    dates = [T + timedelta(days=d) for d in expected]
    np.testing.assert_allclose(
        curve.get_dfs(dates), list(expected.values()), atol=1e-11
    )


def test_the_method_works_on_a_discount_curve():
    days, averages, expected = QUANTLIB["hump"]
    knots = [T, *(T + timedelta(days=d) for d in days)]
    curve = DiscountCurve(T, Currency.USD, ICP, knots, averages, METHOD)
    dates = [T + timedelta(days=d) for d in expected]
    np.testing.assert_allclose(
        curve.get_dfs(dates), list(expected.values()), atol=1e-11
    )


def test_each_segment_keeps_its_average():
    integral = convex_monotone_forward(TAU, AVERAGES).antiderivative()
    np.testing.assert_allclose(
        np.diff(integral(TAU)) / np.diff(TAU), AVERAGES, rtol=1e-12
    )


def test_the_forward_is_continuous_at_the_knots():
    forward = convex_monotone_forward(TAU, AVERAGES)
    inner = TAU[1:-1]
    np.testing.assert_allclose(forward(inner - 1e-9), forward(inner + 1e-9), atol=1e-6)


def test_a_segment_only_moves_the_segments_next_to_it():
    probe = np.linspace(TAU[0], TAU[-1], 2000)
    bumped = AVERAGES.copy()
    bumped[4] += 0.0001
    moved = (
        convex_monotone_forward(TAU, bumped)(probe)
        - convex_monotone_forward(TAU, AVERAGES)(probe)
    ) != 0
    assert moved[(probe > TAU[3]) & (probe < TAU[6])].any()
    assert not moved[(probe < TAU[3]) | (probe > TAU[6])].any()


def test_equal_averages_give_a_flat_forward():
    forward = convex_monotone_forward(TAU, np.full(len(AVERAGES), 0.05))
    np.testing.assert_allclose(forward(np.linspace(0, TAU[-1], 100)), 0.05, atol=1e-14)


def test_past_the_last_knot_the_forward_continues_from_its_last_value():
    curve = _index_curve(DAYS.tolist(), AVERAGES)
    last_forward = float(convex_monotone_forward(TAU, AVERAGES)(TAU[-1]))
    year_later = curve.last_date + timedelta(days=365)
    assert curve.get_df(year_later) == pytest.approx(
        curve.get_df(curve.last_date) * np.exp(-last_forward), rel=1e-13
    )


@pytest.mark.parametrize(
    "g0, g1",
    [
        (1.0, np.nextafter(-2.0, -3.0)),  # The inner break lands on the start.
        (1.0, np.nextafter(-0.5, 0.0)),  # The inner break lands on the end.
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
    ],
)
def test_degenerate_shapes_stay_finite(g0, g1):
    eta, left, right = _segment_polynomials(g0, g1)
    assert np.isfinite([eta, *left, *right]).all()
