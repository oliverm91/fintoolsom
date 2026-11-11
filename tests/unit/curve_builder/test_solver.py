"""Unknown layout, atomic root and the date-by-date bootstrap (_solver)."""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.curve_builder._seeds import _FixedSegments
from fintoolsom.curve_builder._solver import (
    _GroupSolve,
    _solve_or_rollback,
    _solve_unknowns,
)
from fintoolsom.dates import ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.market.index import TermRateIndex
from fintoolsom.rates import InterpolationMethod

T = date(2026, 6, 29)
SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)
CLP = Currency.CLP  # a DiscountCurve key under SOFR
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
    assert "TERM3M" not in market.curves


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
    assert "CLP_SOFR" not in market.curves


def test_failed_root_rolls_the_market_back():
    restored = []
    with pytest.raises(ValueError, match="failed to converge|non-finite"):
        _solve_or_rollback(
            lambda y: [y[0] ** 2 + 1.0], [0.5], lambda: restored.append(True), "toy"
        )
    assert restored == [True]


class _ToyInstrument:
    """Residual = df(pillar) - target on one curve: a one-pillar 'quote'."""

    def __init__(self, d, target):
        self.d, self.target = d, target


def test_date_by_date_and_joint_solves_agree(monkeypatch):
    from fintoolsom.curve_builder import _solver

    market = Market(T, risk_free_index=SOFR)
    dates = [T + _days(365), T + _days(730), T + _days(1095)]
    targets = [0.96, 0.91, 0.87]
    instruments = [_ToyInstrument(d, x) for d, x in zip(dates, targets)]
    monkeypatch.setattr(
        _solver.Calculator,
        "valuate",
        staticmethod(
            lambda i, m, ccy: m.get_discount_curve(CLP).get_df(i.d) - i.target
        ),
    )

    def solve(date_by_date):
        layout = _GroupSolve.from_free_pillars(
            market,
            T,
            InterpolationMethod.PiecewiseConstant,
            fixed_pillars={CLP: _FixedSegments([T], [])},
            free_dates={CLP: dates},
            free_seeds={CLP: [0.01, 0.01, 0.01]},
        )
        forwards = _solve_unknowns(
            layout,
            instruments,
            dates,
            Currency.USD,
            "toy",
            date_by_date=date_by_date,
        )
        return np.exp(-np.cumsum(forwards))  # segments of exactly one year

    assert solve(True) == pytest.approx(targets, rel=1e-10)
    assert solve(False) == pytest.approx(targets, rel=1e-10)
