"""Unknown layout, atomic root, and how the solver creates and updates curves."""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.curve_builder import _solver, build_curves
from fintoolsom.curve_builder._keys import _built
from fintoolsom.curve_builder._seeds import _FixedSegments
from fintoolsom.curve_builder._solver import _GroupSolve, _solve_or_rollback
from fintoolsom.dates import ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.market.index import TermRateIndex
from fintoolsom.rates import PillarCurve, InterpolationMethod
from tests.factories import quote_factory
from tests.factories.quote_factory import ICP

T = date(2026, 6, 29)
SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)
CLP = Currency.CLP  # a PillarCurve key under SOFR
TERM3M = TermRateIndex(
    "TERM3M",
    currency=Currency.USD,
    term=Term(3, TermUnit.M, ModifiedFollowingConvention("US")),
)


def _days(n: int) -> timedelta:
    return timedelta(days=n)


def test_group_solve_applies_and_rolls_back_an_index_curve():
    market = Market(T, risk_free_index=SOFR)
    knots = [T + _days(2), T + _days(94)]
    solve = _GroupSolve.from_free_pillars(
        market,
        T,
        InterpolationMethod.PiecewiseConstant,
        fixed_pillars={TERM3M: _FixedSegments(knots, [0.05])},
        free_dates={TERM3M: [T + _days(400)]},
        free_seeds={TERM3M: [0.05]},
    )
    assert solve.y0 == [0.05]
    solve.apply([0.06])
    curve = market.get_index_curve(TERM3M)
    assert list(curve.knot_dates) == [*knots, T + _days(400)]
    assert list(curve.forward_rates) == pytest.approx([0.05, 0.06])
    solve.restore()
    assert TERM3M not in market.curves


def test_group_solve_lays_out_discount_unknowns_as_forwards():
    market = Market(T, risk_free_index=SOFR)
    dates = [T + _days(365), T + _days(730)]
    solve = _GroupSolve.from_free_pillars(
        market,
        T,
        InterpolationMethod.PiecewiseConstant,
        fixed_pillars={CLP: _FixedSegments([T], [])},
        free_dates={CLP: dates},
        free_seeds={CLP: [0.05, 0.04]},
    )
    assert solve.n_unknowns == 2 and solve.y0 == pytest.approx([0.05, 0.04])
    solve.apply(solve.y0)
    curve = market.get_discount_curve(CLP)
    assert curve.knot_dates == [T, *dates]
    assert curve.get_dfs(dates) == pytest.approx([math.exp(-0.05), math.exp(-0.09)])
    solve.restore()
    assert Currency.CLP not in market.curves


def test_failed_root_rolls_the_market_back():
    restored = []
    with pytest.raises(ValueError, match="failed to converge|non-finite"):
        _solve_or_rollback(
            lambda y: [y[0] ** 2 + 1.0], [0.5], lambda: restored.append(True), "toy"
        )
    assert restored == [True]


# ── How the solver creates and updates curves ───────────────────────────────

PC, CS = InterpolationMethod.PiecewiseConstant, InterpolationMethod.CubicSpline


@pytest.mark.parametrize(
    "method, precalculated", [(PC, False), (CS, True)], ids=["local", "cubic"]
)
def test_the_daily_grid_is_skipped_only_while_solving_local_curves(
    monkeypatch, method, precalculated
):
    flags = []
    apply = _solver._GroupSolve.apply

    def recording_apply(self, y):
        apply(self, y)
        flags.extend(_built(self.market, c).precalculate for c in self.curve_slice)

    monkeypatch.setattr(_solver._GroupSolve, "apply", recording_apply)
    market = quote_factory.quote_market()
    build_curves(quote_factory.load_quotes(), market, interp_method=method)
    assert flags and all(bool(flag) is precalculated for flag in flags)
    assert all(c._grid is not None for c in market.curves.values())


@pytest.mark.parametrize("method", [PC, CS])
def test_the_solver_builds_each_curve_once_per_solve(monkeypatch, method):
    built, updated = [], []
    cls = _solver.PillarCurve
    monkeypatch.setattr(
        _solver, "PillarCurve", lambda *a, **k: built.append(1) or cls(*a, **k)
    )
    set_forwards = PillarCurve._set_forward_rates
    monkeypatch.setattr(
        PillarCurve,
        "_set_forward_rates",
        lambda self, f: updated.append(1) or set_forwards(self, f),
    )
    monkeypatch.setattr(
        PillarCurve,
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
    build_curves(
        quote_factory.load_quotes(),
        quote_factory.quote_market(),
        interp_method=method,
    )
    # One construction per curve of each solve; every other iteration in place.
    assert len(built) - len(private) == sum(solves)
    assert len(updated) > 3 * len(built)


def test_a_rolled_back_solve_leaves_the_previous_curve_untouched():
    knots = [T + _days(d) for d in (0, 30, 91, 365, 730, 1825)]
    forwards = [0.05, 0.048, 0.046, 0.045, 0.043]
    probe = [T + _days(d) for d in range(-3, 2200, 3)]
    market = quote_factory.quote_market()
    previous = PillarCurve(T, knots, forwards, PC)
    market.set_curve(ICP, previous)
    before = previous.get_dfs(probe)
    solve = _GroupSolve.from_free_pillars(
        market,
        T,
        PC,
        {ICP: _FixedSegments(knots[:2], forwards[:1], None)},
        {ICP: knots[2:]},
        {ICP: [0.052, 0.049, 0.038, 0.047]},
    )
    for y in ([0.01] * 4, [0.02] * 4, [0.03] * 4):
        solve.apply(np.array(y))
    current = market.get_index_curve(ICP)
    assert current is not previous
    solve.restore()
    assert market.get_index_curve(ICP) is previous
    assert np.array_equal(previous.get_dfs(probe), before)
    solve.apply(np.array([0.04] * 4))  # after a rollback: a new curve again
    assert market.get_index_curve(ICP) is not previous
    assert market.get_index_curve(ICP) is not current
