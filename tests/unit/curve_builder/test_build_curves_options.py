"""build_curves' report_currency, reprice_tol and sweep_tol keyword arguments."""

import math
import re
import warnings

import numpy as np
import pytest

from fintoolsom.curve_builder import (
    RedundantQuoteWarning,
    RepricingError,
    _solver,
    build_curves,
)
from fintoolsom.curve_builder._keys import _instrument_of
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.market import Currency
from fintoolsom.rates import InterpolationMethod
from tests.factories import quote_factory


def _build(**kwargs):
    market = quote_factory.quote_market()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_curves(quote_factory.load_quotes(), market, **kwargs)
    ditched = [
        w.message for w in caught if issubclass(w.category, RedundantQuoteWarning)
    ]
    return market, ditched


def _forwards(market) -> dict:
    return {k: np.asarray(c.forward_rates) for k, c in market.curves.items()}


def test_the_defaults_are_the_documented_values():
    default, _ = _build()
    explicit, _ = _build(
        report_currency=Currency.CLP, reprice_tol=1e-2, sweep_tol=1e-4, group_tol=1.0
    )
    a, b = _forwards(default), _forwards(explicit)
    assert a.keys() == b.keys()
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize(
    "method", [InterpolationMethod.PiecewiseConstant, InterpolationMethod.CubicSpline]
)
def test_another_report_currency_solves_the_same_curves(method):
    clp, _ = _build(interp_method=method)
    usd, ditched = _build(interp_method=method, report_currency=Currency.USD)
    a, b = _forwards(clp), _forwards(usd)
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_allclose(b[k], a[k], rtol=0, atol=1e-12)
    # The ditched quotes are reported in that currency.
    assert len(ditched) == 1
    warning = ditched[0]
    assert isinstance(warning, RedundantQuoteWarning)
    lines = str(warning).splitlines()[1:]
    assert lines and all(re.search(r"MTM -?[\d,.]+ USD\b", line) for line in lines)
    assert warning.ditched
    for quote, mtm, _, _ in warning.ditched:
        instrument = _instrument_of(quote, usd)
        assert mtm == Calculator.valuate(instrument, usd, Currency.USD)


def test_a_tighter_reprice_tol_raises():
    with pytest.raises(RepricingError, match="no longer reprice"):
        _build(reprice_tol=1e-300)


def test_sweeps_above_sweep_tol_go_to_the_joint_solve(monkeypatch):
    joint = []
    solve = _solver._solve_unknowns

    def spy(*args, sweep_tol, **kwargs):
        joint.append(sweep_tol)
        return solve(*args, sweep_tol=sweep_tol, **kwargs)

    roots = []
    root = _solver._solve_or_rollback

    def root_spy(residual, y0, *args, **kwargs):
        roots.append(len(y0))
        return root(residual, y0, *args, **kwargs)

    monkeypatch.setattr(_solver, "_solve_or_rollback", root_spy)
    from fintoolsom.curve_builder import builder

    monkeypatch.setattr(builder, "_solve_unknowns", spy)
    _build()
    default_roots = list(roots)
    roots.clear()
    # Unreachable: every sweep is followed by a joint solve of its group.
    _build(sweep_tol=1e-300)
    assert set(joint) == {1e-4, 1e-300}
    assert len(roots) > len(default_roots)
    assert max(roots) > 3  # a joint solve, not a sweep block


class _FlatCalculator:
    """Every MTM the builder checks is `mtm` (the solve itself values elsewhere)."""

    def __init__(self, mtm):
        self.mtm = mtm

    def valuate(self, instrument, market, currency):
        return self.mtm


def test_a_group_whose_quotes_add_up_past_group_tol_raises(monkeypatch):
    """Each quote within reprice_tol, a group of three or more past group_tol = 1."""
    from fintoolsom.curve_builder import builder

    monkeypatch.setattr(builder, "Calculator", _FlatCalculator(0.4))
    with pytest.raises(RepricingError, match=r"group\(s\).*above group_tol = 1\."):
        _build(reprice_tol=0.5)
    _build(reprice_tol=0.5, group_tol=1e6)  # loose enough: builds


def test_reprice_tol_is_checked_before_group_tol(monkeypatch):
    from fintoolsom.curve_builder import builder

    monkeypatch.setattr(builder, "Calculator", _FlatCalculator(0.4))
    with pytest.raises(RepricingError, match="no longer reprice"):
        _build(reprice_tol=0.3, group_tol=1e-300)


def test_the_fitted_quotes_reprice_well_within_the_defaults():
    from fintoolsom.curve_builder import builder

    market = quote_factory.quote_market()
    cb = builder._CurveBuilder(quote_factory.load_quotes(), market)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RedundantQuoteWarning)
        cb.build()
    mtms = [abs(Calculator.valuate(i, market, c)) for i, c in cb._pinning]
    assert len(mtms) == len(cb._pinned_groups) > 0
    assert max(mtms) < 1e-8  # the defaults leave a wide margin


@pytest.mark.parametrize("name", ["reprice_tol", "sweep_tol", "group_tol"])
@pytest.mark.parametrize("value", [0, -1e-6, math.nan, math.inf, "1e-4", None])
def test_a_tolerance_must_be_positive_and_finite(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive finite number"):
        build_curves(
            quote_factory.load_quotes(), quote_factory.quote_market(), **{name: value}
        )
