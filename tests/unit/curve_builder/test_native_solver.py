"""``fintoolsom._native.GroupSolver``, the C++ group solve of the C++ curve builder:
its residuals and Jacobian against the Python solver's, the sweeps bit-identical to
the Python ones, and every way back to the Python solve."""

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.curve_builder import _solver, build_curves
from fintoolsom.derivatives.calculator import ValuationBatch
from fintoolsom.rates import InterpolationMethod
from tests.factories import quote_factory

LOCAL = [InterpolationMethod.PiecewiseConstant, InterpolationMethod.PiecewiseLinear]
JOINT = [
    InterpolationMethod.CubicSpline,
    InterpolationMethod.HermiteCubicSpline,
    InterpolationMethod.ConvexMonotone,
]


def _build(method, builder="cpp"):
    with settings.use_backend("cpp"), settings.use_backend(builder, "curve_builder"):
        market = quote_factory.quote_market()
        build_curves(quote_factory.load_quotes(), market, interp_method=method)
    return market


def _forwards(market) -> dict:
    return {k: np.asarray(c.forward_rates) for k, c in market.curves.items()}


def _spy_solvers(monkeypatch) -> list:
    """(solve, instruments, report currency, GroupSolver) of every C++ solve."""
    made = []
    native_solver = _solver._native_solver

    def spy(solve, instruments, report_currency):
        native = native_solver(solve, instruments, report_currency)
        made.append((solve, instruments, report_currency, native))
        return native

    monkeypatch.setattr(_solver, "_native_solver", spy)
    return made


@pytest.mark.parametrize("method", LOCAL + JOINT)
def test_every_solve_runs_in_cpp_without_the_python_solver(monkeypatch, method):
    """No Python Newton, and hybr only on the few one-to-two-unknown blocks Newton gives
    up on (one on these quotes, as in the Python sweep)."""
    made = _spy_solvers(monkeypatch)

    def python_newton(*args, **kwargs):
        raise AssertionError("the Python Newton ran")

    roots = []
    root = _solver.root

    def spy(fun, x0, *args, **kwargs):
        roots.append(len(x0))
        return root(fun, x0, *args, **kwargs)

    monkeypatch.setattr(_solver, "_newton", python_newton)
    monkeypatch.setattr(_solver, "root", spy)
    _build(method)
    assert made and all(native is not None for *_, native in made)
    assert len(roots) <= 2 and all(n <= 2 for n in roots)


@pytest.mark.parametrize("method", LOCAL)
def test_cpp_sweeps_are_bit_identical_to_the_python_ones(method):
    py, cpp = _forwards(_build(method, "python")), _forwards(_build(method))
    assert py.keys() == cpp.keys()
    for key, forwards in py.items():
        assert np.array_equal(forwards, cpp[key]), key


@pytest.mark.parametrize("method", JOINT)
def test_the_cpp_joint_solve_finds_the_python_root(method):
    py, cpp = _forwards(_build(method, "python")), _forwards(_build(method))
    for key, forwards in py.items():
        np.testing.assert_allclose(cpp[key], forwards, rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", LOCAL + JOINT)
def test_residuals_are_the_batch_values(monkeypatch, method):
    checked = []
    native_solver = _solver._native_solver

    def check(solve, instruments, currency):
        native = native_solver(solve, instruments, currency)
        y = np.asarray(solve.y0, dtype=float) * 1.01
        cpp = native.residuals(y)
        solve.apply(y)
        expected = ValuationBatch(instruments, solve.market, currency).values()
        assert cpp.tolist() == expected.tolist()
        checked.append(len(cpp))
        return native

    monkeypatch.setattr(_solver, "_native_solver", check)
    _build(method)
    assert checked


@pytest.mark.parametrize("method", JOINT)
def test_the_cpp_jacobian_is_the_python_one(monkeypatch, method):
    made = _spy_solvers(monkeypatch)
    checked = []
    native_joint = _solver._native_joint

    def check(native, solve, instruments, y):
        currency = next(c for s, _, c, n in made if n is native)
        linear = all(m in _solver._LINEAR_METHODS for m in solve.methods.values())
        bump = _solver._LINEAR_BUMP if linear else _solver._NONLINEAR_BUMP
        cpp = native.jacobian(np.asarray(y, dtype=float), linear, bump)
        batch = ValuationBatch(instruments, solve.market, currency, compile_all=True)
        python = _solver._Jacobian(solve, batch, np.arange(solve.n_unknowns))(y)
        assert cpp.shape == python.shape
        scale = np.abs(python).max()
        assert np.abs(cpp - python).max() <= 1e-9 * scale
        checked.append(cpp.shape)
        return native_joint(native, solve, instruments, y)

    monkeypatch.setattr(_solver, "_native_joint", check)
    _build(method)
    assert checked and max(r * c for r, c in checked) > 1000


class _Wrapped:
    """A GroupSolver whose sweeps give up as `sweep` says."""

    def __init__(self, native, sweep):
        self.native, self._sweep = native, sweep

    def __getattr__(self, name):
        return getattr(self.native, name)

    def sweep(self, *args):
        return self._sweep(self.native, *args)


def _wrap_solvers(monkeypatch, sweep, joint=None):
    native_solver = _solver._native_solver

    def wrapped(*args):
        native = native_solver(*args)
        if native is None:
            return None
        out = _Wrapped(native, sweep)
        if joint is not None:
            out.joint = joint
        return out

    monkeypatch.setattr(_solver, "_native_solver", wrapped)


@pytest.mark.parametrize("method", [InterpolationMethod.PiecewiseConstant, *JOINT])
def test_a_cpp_solve_that_raises_is_redone_in_python(monkeypatch, method):
    """Status 3 (a valuation raised) and a joint solve stopping short: Python redoes
    them, so the Python builder's curves."""
    _wrap_solvers(monkeypatch, lambda native, y, *a: (3, y, 0, 0), lambda *a: None)
    cpp = _forwards(_build(method))
    py = _forwards(_build(method, "python"))
    for key, forwards in py.items():
        assert np.array_equal(forwards, cpp[key]), key


def test_a_block_newton_gives_up_on_is_solved_by_hybr_and_the_sweep_resumes(
    monkeypatch,
):
    """Status 2 on the first block of every solve: hybr solves it in Python, then the
    C++ sweep goes on from the next block."""
    resumed = []

    def first_block_fails(native, y, max_sweeps, tol, sweep, block):
        if sweep == 0 and block == 0:
            return 2, y, 0, 0
        resumed.append(block)
        return native.sweep(y, max_sweeps, tol, sweep, block)

    roots = []
    solve_or_rollback = _solver._solve_or_rollback

    def spy(residual, y0, *args, **kwargs):
        roots.append(len(y0))
        return solve_or_rollback(residual, y0, *args, **kwargs)

    _wrap_solvers(monkeypatch, first_block_fails)
    monkeypatch.setattr(_solver, "_solve_or_rollback", spy)
    method = InterpolationMethod.PiecewiseConstant
    cpp = _forwards(_build(method))
    monkeypatch.undo()
    py = _forwards(_build(method, "python"))
    assert roots and 1 in resumed
    for key, forwards in py.items():
        np.testing.assert_allclose(cpp[key], forwards, rtol=0, atol=1e-12)


def test_sweeps_above_the_tolerance_go_to_the_cpp_joint_solve(monkeypatch):
    monkeypatch.setattr(_solver, "_SWEEP_TOL", -1.0)  # never met
    joints = []
    native_joint = _solver._native_joint

    def spy(*args):
        out = native_joint(*args)
        joints.append(out is not None)
        return out

    monkeypatch.setattr(_solver, "_native_joint", spy)
    method = InterpolationMethod.PiecewiseConstant
    cpp = _forwards(_build(method))
    py = _forwards(_build(method, "python"))
    assert joints and all(joints)
    for key, forwards in py.items():
        np.testing.assert_allclose(cpp[key], forwards, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "switch", ["builder", "_BLOCK_NEWTON", "_ANALYTIC_JACOBIAN", "rates"]
)
def test_the_python_solve_runs_when_the_cpp_one_does_not_apply(monkeypatch, switch):
    made = _spy_solvers(monkeypatch)
    method = InterpolationMethod.PiecewiseConstant
    if switch == "builder":
        _build(method, "python")
    elif switch == "rates":  # Python curves have no C++ core
        with (
            settings.use_backend("cpp"),
            settings.use_backend("python", "rates"),
        ):
            build_curves(
                quote_factory.load_quotes(),
                quote_factory.quote_market(),
                interp_method=method,
            )
    else:
        monkeypatch.setattr(_solver, switch, False)
        _build(method)
    assert made and all(native is None for *_, native in made)
