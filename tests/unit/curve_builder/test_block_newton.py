"""The sweep blocks' Newton solver, and its fallback to hybr."""

import math

import numpy as np
import pytest

from fintoolsom.curve_builder import _solver, build_curves
from fintoolsom.rates import InterpolationMethod
from tests.factories import quote_factory

LOCAL = [InterpolationMethod.PiecewiseConstant, InterpolationMethod.PiecewiseLinear]


def _build(method):
    market = quote_factory.quote_market()
    build_curves(quote_factory.load_quotes(), market, interp_method=method)
    return market


@pytest.mark.parametrize("n", [1, 2, 3])
def test_solve_small_matches_numpy(n):
    rng = np.random.default_rng(n)
    a = rng.normal(size=(n, n)) + 3 * np.eye(n)
    b = rng.normal(size=n)
    x = _solver._solve_small(a.tolist(), b.tolist())
    np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-13, atol=1e-15)


def test_solve_small_reports_a_singular_matrix():
    assert _solver._solve_small([[0.0]], [1.0]) is None
    assert _solver._solve_small([[1.0, 2.0], [2.0, 4.0]], [1.0, 1.0]) is None


def test_newton_solves_a_nearly_linear_system_in_few_evaluations():
    calls = []

    def residual(x):
        calls.append(1)
        # Discount-factor-like: exp terms, nearly linear around the root.
        return [
            100 * (math.exp(-2 * x[0]) - math.exp(-0.06)),
            100 * (math.exp(-3 * x[1] - x[0]) - math.exp(-0.12)),
        ]

    solved = _solver._newton(residual, [0.02, 0.02])
    assert solved is not None
    x, jac = solved
    np.testing.assert_allclose(x, [0.03, 0.03], rtol=0, atol=1e-13)
    assert len(calls) <= 8  # 1 + 2 differences + a few steps
    calls.clear()
    # A reused Jacobian needs no differences: the solution's own residual and done.
    again = _solver._newton(residual, x, jac)
    assert again is not None and len(calls) == 1


def test_newton_gives_up_when_the_steps_do_not_lower_the_residuals():
    # A root Newton overshoots from this seed: |x|^(1/3) has no usable derivative.
    assert (
        _solver._newton(lambda x: [math.copysign(abs(x[0]) ** (1 / 3), x[0])], [1.0])
        is None
    )


def test_newton_gives_up_on_non_finite_residuals():
    assert _solver._newton(lambda x: [math.nan], [0.0]) is None


@pytest.mark.parametrize("method", LOCAL)
def test_sweeps_solve_by_newton_and_match_hybr(monkeypatch, method):
    roots = []
    solve = _solver._solve_or_rollback

    def spy(*args, **kwargs):
        roots.append(len(args[1]))
        return solve(*args, **kwargs)

    monkeypatch.setattr(_solver, "_solve_or_rollback", spy)
    newton = _build(method)
    newton_roots = list(roots)
    monkeypatch.setattr(_solver, "_BLOCK_NEWTON", False)
    roots.clear()
    hybr = _build(method)
    assert len(newton_roots) < len(roots)  # the blocks no longer go to hybr
    for cid, curve in newton.curves.items():
        np.testing.assert_allclose(
            curve.forward_rates, hybr.curves[cid].forward_rates, rtol=0, atol=1e-10
        )


@pytest.mark.parametrize("method", LOCAL)
def test_a_block_newton_gives_up_on_goes_to_hybr(monkeypatch, method):
    expected = _build(method)
    monkeypatch.setattr(_solver, "_newton", lambda *a, **k: None)
    fallback = _build(method)
    for cid, curve in expected.curves.items():
        np.testing.assert_allclose(
            curve.forward_rates, fallback.curves[cid].forward_rates, rtol=0, atol=1e-10
        )
