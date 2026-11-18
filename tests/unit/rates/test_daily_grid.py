"""Daily discount-factor grid of PillarCurve."""

from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.rates import (
    PillarCurve,
    InterpolationMethod,
    Rate,
    RateConvention,
)
from tests.factories.curve_factory import curve_from_dfs

T = date(2026, 6, 29)
PILLARS = [T + timedelta(days=d) for d in (30, 91, 365, 730, 1825)]
FORWARDS = [0.05, 0.048, 0.046, 0.045]
DFS = [0.996, 0.988, 0.955, 0.915, 0.80]
METHOD = InterpolationMethod


def _index(method, spot=None, **kwargs):
    knots = [T + timedelta(days=1) if spot else T, *PILLARS[:-1]]
    return PillarCurve(T, knots, FORWARDS, method, spot, **kwargs)


def _discount(method, **kwargs):
    return curve_from_dfs(T, PILLARS, DFS, method, **kwargs)


SPOT = Rate(RateConvention(), 0.05)
MAKERS = {
    "index_piecewise": lambda **k: _index(METHOD.PiecewiseConstant, **k),
    "index_linear": lambda **k: _index(METHOD.PiecewiseLinear, **k),
    "index_cubic": lambda **k: _index(METHOD.CubicSpline, **k),
    "index_spot": lambda **k: _index(METHOD.PiecewiseConstant, SPOT, **k),
    "discount_piecewise": lambda **k: _discount(METHOD.PiecewiseConstant, **k),
    "discount_hermite": lambda **k: _discount(METHOD.HermiteCubicSpline, **k),
    "discount_cubic": lambda **k: _discount(METHOD.CubicSpline, **k),
}


@pytest.fixture(params=list(MAKERS.values()), ids=list(MAKERS))
def make(request):
    return request.param


def test_grid_gives_the_formula_inside_and_outside_its_range(make):
    with_grid, formula = make(), make(precalculate=False)
    last = (with_grid.last_date - T).days
    days = [T + timedelta(days=n) for n in range(-5, last + 40)]
    np.testing.assert_allclose(
        with_grid.get_dfs(days), formula.get_dfs(days), rtol=1e-14, atol=0
    )
    for d in (days[0], days[20], days[-1]):
        assert with_grid.get_df(d) == pytest.approx(formula.get_df(d), rel=1e-14)
        assert with_grid.get_df_fwd(d, days[30]) == pytest.approx(
            formula.get_df_fwd(d, days[30]), rel=1e-14
        )


def test_grid_is_created_by_default_and_skipped_on_request(make):
    assert make()._grid is not None
    assert make(precalculate=False)._grid is None


def test_precalculate_dfs_builds_the_grid_of_a_curve_without_one(make):
    curve = make(precalculate=False)
    curve.precalculate_dfs()
    assert curve.precalculate is True
    assert len(curve._grid) == (curve.last_date - T).days + 1


def test_precalculate_dfs_recomputes_an_existing_grid(make):
    curve = make()
    first = curve._grid
    curve.precalculate_dfs()
    assert curve._grid is not first


@pytest.mark.parametrize("flag", [True, False])
def test_copy_and_bump_keep_the_flag(make, flag):
    curve = make(precalculate=flag)
    for derived in (curve.copy(), curve.parallel_bump_bps(1.0)):
        assert derived.precalculate is flag
        assert (derived._grid is not None) is flag


def test_writing_to_the_returned_dfs_leaves_the_curve_alone(make):
    curve = make()
    d = T + timedelta(days=45)
    before = curve.get_df(d)
    curve.get_dfs([d])[0] = 0.0
    assert curve.get_df(d) == before
