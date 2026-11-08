"""Unknown layout, atomic root and the date-by-date bootstrap (_solver)."""
import math
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.curve_builder._keys import _Proj
from fintoolsom.curve_builder._solver import _GroupSolve, _solve_or_rollback, _solve_unknowns
from fintoolsom.dates import ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.market.index import TermRateIndex
from fintoolsom.rates import InterpolationMethod

T = date(2026, 6, 29)
SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)
TERM3M = TermRateIndex("TERM3M", currency=Currency.USD, term=Term(3, TermUnit.M, ModifiedFollowingConvention("US")))


def _days(n: int) -> timedelta:
    return timedelta(days=n)


def test_group_solve_applies_and_rolls_back_a_projection_curve():
    market = Market(T, riskless_index=SOFR)
    key = _Proj(TERM3M)
    knots = [T + _days(2), T + _days(94)]
    solve = _GroupSolve.from_free_pillars(
        market, T, InterpolationMethod.LogLinear,
        fixed_pillars={key: (knots, [0.05])}, free_dates={key: [T + _days(400)]}, free_seed_dfs={key: [0.05]},
    )
    assert solve.y0 == [0.05]  # projection unknowns are forward rates, not log(df)
    solve.apply([0.06])
    curve = market.projection_curves[TERM3M]
    assert list(curve.knot_dates) == [*knots, T + _days(400)]
    assert list(curve.forward_rates) == pytest.approx([0.05, 0.06])
    solve.restore()
    assert TERM3M not in market.projection_curves


def test_group_solve_lays_out_discount_unknowns_in_log_df():
    market = Market(T, riskless_index=SOFR)
    dates = [T + _days(365), T + _days(730)]
    solve = _GroupSolve.from_free_pillars(
        market, T, InterpolationMethod.LogLinear, fixed_pillars={SOFR: []}, free_dates={SOFR: dates},
        free_seed_dfs={SOFR: [0.95, 0.90]},
    )
    assert solve.n_unknowns == 2 and solve.y0 == pytest.approx([math.log(0.95), math.log(0.90)])
    solve.apply(solve.y0)
    assert market.curves[SOFR].get_dfs(dates) == pytest.approx([0.95, 0.90])
    solve.restore()
    assert SOFR not in market.curves


def test_failed_root_rolls_the_market_back():
    restored = []
    with pytest.raises(ValueError, match="failed to converge|non-finite"):
        _solve_or_rollback(lambda y: [y[0] ** 2 + 1.0], [0.5], lambda: restored.append(True), "toy")
    assert restored == [True]


class _ToyInstrument:
    """Residual = df(pillar) - target on one curve: a one-pillar 'quote'."""

    def __init__(self, d, target):
        self.d, self.target = d, target


def test_date_by_date_and_joint_solves_agree(monkeypatch):
    from fintoolsom.curve_builder import _solver

    market = Market(T, riskless_index=SOFR)
    dates = [T + _days(365), T + _days(730), T + _days(1095)]
    targets = [0.96, 0.91, 0.87]
    instruments = [_ToyInstrument(d, x) for d, x in zip(dates, targets)]
    monkeypatch.setattr(
        _solver.Calculator, "valuate", staticmethod(lambda i, m, ccy: m.curves[SOFR].get_df(i.d) - i.target)
    )

    def solve(date_by_date):
        layout = _GroupSolve.from_free_pillars(
            market, T, InterpolationMethod.LogLinear, fixed_pillars={SOFR: []}, free_dates={SOFR: dates},
            free_seed_dfs={SOFR: [0.99, 0.99, 0.99]},
        )
        return np.exp(_solve_unknowns(layout, instruments, dates, Currency.USD, "toy", date_by_date=date_by_date))

    assert solve(True) == pytest.approx(targets, rel=1e-10)
    assert solve(False) == pytest.approx(targets, rel=1e-10)
