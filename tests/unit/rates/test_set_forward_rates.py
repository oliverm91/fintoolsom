"""Forwards updated in place give the curve built on them, and the solver uses it.

``PillarCurve._set_forward_rates`` is how the curve builder's solver moves its curves
on every iteration instead of constructing new ones: the curve after it must be,
to the bit, the one ``__post_init__`` builds on the new forwards, on both backends,
for every method, with or without a spot rate or a daily grid.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.curve_builder import _solver, build_curves
from fintoolsom.curve_builder._seeds import _FixedSegments
from fintoolsom.market import Currency
from fintoolsom.rates import (
    DiscountCurve,
    IndexCurve,
    InterpolationMethod,
    Rate,
    RateConvention,
)
from tests.factories import quote_factory
from tests.factories.quote_factory import ICP

T = date(2026, 6, 29)
KNOTS = [T + timedelta(days=d) for d in (0, 30, 91, 365, 730, 1825)]
OLD = [0.05, 0.048, 0.046, 0.045, 0.043]
NEW = [0.031, 0.052, 0.049, 0.038, 0.047]
SPOT = Rate(RateConvention(), 0.05)
PROBE = [T + timedelta(days=d) for d in range(-3, 2200, 3)]


def _index(method, forwards, spot=None, **kw):
    knots = [T + timedelta(days=2), *KNOTS[1:]] if spot else KNOTS
    return IndexCurve(T, ICP, knots, forwards, spot, method, **kw)


def _discount(method, forwards, **kw):
    return DiscountCurve(T, Currency.USD, ICP, KNOTS, forwards, method, **kw)


MAKERS = {
    "index": _index,
    "index_spot": lambda m, f, **kw: _index(m, f, SPOT, **kw),
    "discount": _discount,
}


@pytest.fixture(autouse=True)
def _native(request):
    if "cpp" in request.node.name:
        pytest.importorskip("fintoolsom._native")


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("make", list(MAKERS.values()), ids=list(MAKERS))
@pytest.mark.parametrize("method", list(InterpolationMethod), ids=lambda m: m.name)
@pytest.mark.parametrize("precalculate", [True, False])
def test_updated_curve_is_the_curve_built_on_the_new_forwards(
    backend, make, method, precalculate
):
    if method not in IndexCurve.allowed_methods:
        pytest.skip("not a pillar curve method")
    with settings.use_backend(backend, "rates"):
        updated = make(method, OLD, precalculate=precalculate)
        updated._set_forward_rates(list(NEW))
        built = make(method, NEW, precalculate=precalculate)
        assert np.array_equal(updated.forward_rates, built.forward_rates)
        assert updated._log_df_effective == built._log_df_effective
        assert (updated._grid is None) is (built._grid is None)
        if built._grid is not None:
            assert np.array_equal(updated._grid, built._grid)
        assert np.array_equal(updated.get_dfs(PROBE), built.get_dfs(PROBE))
        updated._set_forward_rates(list(OLD))  # and back
        assert np.array_equal(
            updated.get_dfs(PROBE),
            make(method, OLD, precalculate=precalculate).get_dfs(PROBE),
        )


def test_the_number_of_forwards_is_checked():
    curve = _index(InterpolationMethod.PiecewiseConstant, OLD)
    with pytest.raises(ValueError, match="one entry per segment"):
        curve._set_forward_rates(OLD[:-1])


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "method", [InterpolationMethod.PiecewiseConstant, InterpolationMethod.CubicSpline]
)
def test_the_solver_builds_each_curve_once_per_solve(monkeypatch, backend, method):
    built, updated = [], []
    original_build = {
        "IndexCurve": _solver.IndexCurve,
        "DiscountCurve": _solver.DiscountCurve,
    }
    for name, cls in original_build.items():
        monkeypatch.setattr(
            _solver, name, lambda *a, cls=cls, **k: built.append(1) or cls(*a, **k)
        )
    set_forwards = IndexCurve._set_forward_rates
    monkeypatch.setattr(
        IndexCurve,
        "_set_forward_rates",
        lambda self, f: updated.append(1) or set_forwards(self, f),
    )
    monkeypatch.setattr(
        DiscountCurve,
        "_set_forward_rates",
        lambda self, f: updated.append(1) or set_forwards(self, f),
    )
    solves = []
    init = _solver._GroupSolve.__post_init__

    def counting_init(self):
        init(self)
        solves.append(len(self.curve_slice))

    monkeypatch.setattr(_solver._GroupSolve, "__post_init__", counting_init)
    # The Jacobian bumps private copies, which never reach the market.
    private = []
    private_curve = _solver._GroupSolve.private_curve
    monkeypatch.setattr(
        _solver._GroupSolve,
        "private_curve",
        lambda self, *a: private.append(1) or private_curve(self, *a),
    )
    with settings.use_backend(backend, "rates"):
        build_curves(
            quote_factory.load_quotes(),
            quote_factory.quote_market(),
            interp_method=method,
        )
    # One construction per curve of each solve; every other iteration in place.
    assert len(built) - len(private) == sum(solves)
    assert len(updated) > 3 * len(built)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_a_rolled_back_solve_leaves_the_previous_curve_untouched(backend):
    with settings.use_backend(backend, "rates"):
        market = quote_factory.quote_market()
        previous = _index(InterpolationMethod.PiecewiseConstant, OLD)
        market.set_curve(previous)
        before = previous.get_dfs(PROBE)
        fixed = _FixedSegments(KNOTS[:2], OLD[:1], None)
        solve = _solver._GroupSolve.from_free_pillars(
            market,
            T,
            InterpolationMethod.PiecewiseConstant,
            {ICP: fixed},
            {ICP: KNOTS[2:]},
            {ICP: NEW[1:]},
        )
        for y in ([0.01] * 4, [0.02] * 4, [0.03] * 4):
            solve.apply(np.array(y))
        current = market.get_index_curve(ICP)
        assert current is not previous
        solve.restore()
        assert market.get_index_curve(ICP) is previous
        assert np.array_equal(previous.get_dfs(PROBE), before)
        solve.apply(np.array([0.04] * 4))  # after a rollback: a new curve again
        assert market.get_index_curve(ICP) is not previous
        assert market.get_index_curve(ICP) is not current
