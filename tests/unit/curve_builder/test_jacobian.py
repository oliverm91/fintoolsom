"""The solver's analytic Jacobian: against central differences of the residuals, and
the curves it solves against hybr's own differencing."""

import numpy as np
import pytest

from fintoolsom.curve_builder import _solver, build_curves
from fintoolsom.derivatives.calculator import ValuationBatch
from fintoolsom.market.currencies import Currency
from fintoolsom.rates import InterpolationMethod
from tests.factories import quote_factory
from tests.factories.swap_factory import build_market

# Joint solves (non-local interpolation), where the solver hands hybr the Jacobian.
JOINT = [
    InterpolationMethod.CubicSpline,
    InterpolationMethod.HermiteCubicSpline,
    InterpolationMethod.ConvexMonotone,
]


def _build(method):
    market = quote_factory.quote_market()
    build_curves(quote_factory.load_quotes(), market, interp_method=method)
    return market


@pytest.mark.parametrize("method", JOINT)
def test_jacobian_matches_central_differences(monkeypatch, method):
    call = _solver._Jacobian.__call__
    checked = []

    def check(self, y):
        jac = call(self, y)
        y = np.asarray(y, dtype=float)
        h = 1e-6
        fd = np.empty_like(jac)
        for j, k in enumerate(self.columns):
            up, down = y.copy(), y.copy()
            up[k] += h
            down[k] -= h
            self.solve.apply(up)
            r_up = self.plan.values()
            self.solve.apply(down)
            fd[:, j] = (r_up - self.plan.values()) / (2 * h)
        self.solve.apply(y)
        # Monotone methods difference their curve part forward: ~1e-6 relative.
        assert np.abs(jac - fd).max() <= 1e-5 * np.abs(fd).max()
        checked.append(jac.shape)
        return jac

    monkeypatch.setattr(_solver._Jacobian, "__call__", check)
    _build(method)
    assert checked and max(r * c for r, c in checked) > 1


@pytest.mark.parametrize("method", JOINT)
def test_analytic_and_differenced_jacobians_solve_the_same_curves(monkeypatch, method):
    analytic = _build(method)
    monkeypatch.setattr(_solver, "_ANALYTIC_JACOBIAN", False)
    differenced = _build(method)
    assert analytic.curves.keys() == differenced.curves.keys()
    for cid, curve in analytic.curves.items():
        other = differenced.curves[cid]
        np.testing.assert_allclose(
            curve.forward_rates, other.forward_rates, rtol=0, atol=1e-12
        )


def test_hybr_gets_the_jacobian_on_joint_solves(monkeypatch):
    jacs = []
    root = _solver.root

    def spy(*args, **kwargs):
        jacs.append(kwargs.get("jac"))
        return root(*args, **kwargs)

    monkeypatch.setattr(_solver, "root", spy)
    _build(InterpolationMethod.CubicSpline)
    assert jacs and all(isinstance(j, _solver._Jacobian) for j in jacs)


def test_an_uncompiled_instrument_leaves_the_differencing_to_hybr():
    """A batch with an instrument it leaves to the calculator has no gradient."""
    from fintoolsom.derivatives.swaps import SimplifiedSwap
    from tests.unit.derivatives.test_valuation_batch import _instruments

    market = build_market()
    simplified = [i for i in _instruments() if isinstance(i, SimplifiedSwap)]
    assert simplified
    batch = ValuationBatch(simplified, market, Currency.CLP)
    assert not batch.differentiable
