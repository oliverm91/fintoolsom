"""Solving a planned group: the unknown layout, the atomic root, the bootstrap.

:class:`_GroupSolve` maps the free pillars of a group to an unknown vector and
rebuilds its curves from it; :func:`_solve_unknowns` bootstraps them date by date
(square blocks, ascending) and falls back to one joint ``root``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np
from scipy.optimize import root

from ..derivatives.calculator import Calculator
from ..market.currencies import Currency
from ..market.index import Index
from ..market.market import Market
from ..rates import DiscountCurve, IndexCurve, InterpolationMethod
from ._keys import CurveKey, _built, _curve_id

# A date-by-date bootstrap is accepted when every instrument reprices to this
# (report currency); otherwise the group falls back to one joint solve.
_SWEEP_TOL = 1e-6
# Interpolations where a pillar only moves the curve between its neighbours.
_LOCAL_METHODS = frozenset({InterpolationMethod.LogLinear, InterpolationMethod.PiecewiseConstant})
_MAX_SWEEPS = 4


def _solve_or_rollback(
    residual: Callable[[list[float]], list[float]],
    y0: list[float],
    restore: Callable[[], None],
    what: str,
):
    """Solve ``residual(y) = 0`` with ``scipy.optimize.root`` (hybr), atomically.

    A failed solve must not leave partial curves behind, so the market is rolled
    back through `restore` before raising.

    Parameters
    ----------
    residual : callable
        Residuals as a function of the unknowns.
    y0 : list of float
        Initial guess.
    restore : callable
        Rolls the market back to its state before the solve.
    what : str
        Description used in error messages.

    Returns
    -------
    scipy.optimize.OptimizeResult

    Raises
    ------
    ValueError
        On non-convergence or non-finite residuals.
    """
    try:
        result = root(residual, y0, method="hybr")
    except ValueError as exc:
        # scipy raises on non-finite residuals: the instruments are nearly insensitive
        # to these pillars, so the solver wandered where valuation blows up.
        restore()
        raise ValueError(
            f"{what} did not converge: valuation produced non-finite residuals, so it "
            f"is ill-posed for the pillars it is solving ({exc})."
        ) from exc
    if not result.success:
        restore()
        raise ValueError(f"{what} failed to converge: {result.message}")
    return result


@dataclass
class _GroupSolve:
    """Free-pillar layout of one curve-group solve.

    Holds the free dates of each curve, their slice of the unknown vector, the
    fixed part, and an atomic apply / rollback over ``market.curves``. A currency's
    discount curve is solved in ``log(df)`` at its new pillars (which removes the
    collinearity between near-1 short dfs and small long dfs); an index curve in the
    forward rate of each new segment.

    Parameters
    ----------
    market : Market
        Market whose curves are solved.
    t : date
        Curve date.
    method : InterpolationMethod
        Interpolation of new discount curves. A curve already in the market keeps
        its own method when extended, so its existing part still interpolates as
        when it was solved.
    fixed_pillars : dict
        Curve key to its fixed part: ``(date, df)`` pillars (currency), or an
        :class:`_IndexFixed` (index: knots, forwards and spot rate).
    free_dates : dict
        Curve key to free pillar dates.
    curve_slice : dict
        Curve key to its ``(start, end)`` slice of the unknown vector.
    y0 : list of float
        Initial guesses: ``log(df)``, or segment forwards for an index curve.
    index_method : InterpolationMethod, optional
        Interpolation of new index curves; an existing one keeps its own.
    """

    market: Market
    t: date
    method: InterpolationMethod
    fixed_pillars: dict
    free_dates: dict
    curve_slice: dict
    y0: list[float]
    index_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant
    _snapshot: dict = field(init=False, default=None)
    methods: dict = field(init=False, default=None)

    @classmethod
    def from_free_pillars(
        cls,
        market: Market,
        t: date,
        method: InterpolationMethod,
        fixed_pillars: dict,
        free_dates: dict,
        free_seeds: dict,
        index_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant,
    ) -> _GroupSolve:
        """Lay out the unknown vector from the free dates and their seeds.

        Parameters
        ----------
        market, t, method, fixed_pillars, free_dates, index_method
            As in the class.
        free_seeds : dict
            Curve key to initial values of its free dates: dfs (currency) or
            segment forwards (index).

        Returns
        -------
        _GroupSolve
        """
        curve_slice: dict[CurveKey, tuple[int, int]] = {}
        y0: list[float] = []
        start = 0
        for curve, dates in free_dates.items():
            curve_slice[curve] = (start, start + len(dates))
            seeds = free_seeds[curve]
            y0 += list(seeds) if isinstance(curve, Index) else [math.log(df) for df in seeds]
            start += len(dates)
        return cls(market, t, method, fixed_pillars, free_dates, curve_slice, y0, index_method)

    def __post_init__(self):
        self._snapshot = {c: _built(self.market, c) for c in self.curve_slice}
        # Interpolation of each rebuilt curve: an existing curve keeps its own.
        self.methods = {
            c: prev.interpolation_method if prev is not None
            else (self.index_method if isinstance(c, Index) else self.method)
            for c, prev in self._snapshot.items()
        }

    @property
    def local(self) -> bool:
        """Whether every curve solved interpolates locally (log-linear dfs,
        piecewise-constant forwards): an instrument then depends only on pillars up
        to its own maturity."""
        return all(m in _LOCAL_METHODS for m in self.methods.values())

    @property
    def n_unknowns(self) -> int:
        """Number of free pillars."""
        return len(self.y0)

    def apply(self, y) -> None:
        """Rebuild every group curve from its fixed part plus the unknowns `y`."""
        for curve, (s, e) in self.curve_slice.items():
            fixed = self.fixed_pillars[curve]
            if isinstance(curve, Index):
                built = IndexCurve(
                    self.t, curve, [*fixed.knots, *self.free_dates[curve]],
                    [*fixed.forwards, *(float(v) for v in y[s:e])], fixed.spot_rate, self.methods[curve],
                )
            else:
                dates = [d for d, _ in fixed] + list(self.free_dates[curve])
                dfs = [df for _, df in fixed] + [math.exp(v) for v in y[s:e]]
                built = DiscountCurve(
                    self.t, curve, self.market.risk_free_index, dates, dfs, self.methods[curve]
                )
            self.market.set_curve(built)

    def restore(self) -> None:
        """Put back the curves as they were before the solve."""
        for curve, previous in self._snapshot.items():
            if previous is None:
                self.market.curves.pop(_curve_id(curve, self.market.risk_free_index), None)
            else:
                self.market.set_curve(previous)


def _solve_unknowns(
    solve: _GroupSolve, instruments, pillar_dates, report_currency: Currency, what: str, *, date_by_date: bool = True
) -> np.ndarray:
    """Bootstrap the free pillars date by date, falling back to one joint solve.

    With local interpolation (log-linear dfs, piecewise-constant forwards) an
    instrument depends only on pillars up to its own last free pillar, so
    ascending dates give small square blocks, one per pillar date. The joint
    solve runs, warm-started from the sweep, when the interpolation is not local,
    a block is not square, or the sweep leaves a residual above ``_SWEEP_TOL``.

    Parameters
    ----------
    solve : _GroupSolve
        Layout of the unknowns.
    instruments : list
        Residual instruments, one per unknown.
    pillar_dates : list of date
        Date of the pillar each instrument owns (same order as `instruments`).
    report_currency : Currency
        Currency the residuals are valued in.
    what : str
        Description used in error messages.
    date_by_date : bool, optional
        False always runs the joint solve.

    Returns
    -------
    numpy.ndarray
        Solved unknowns.
    """
    def values(instrs, y) -> list[float]:
        solve.apply(y)
        return [Calculator.valuate(i, solve.market, report_currency) for i in instrs]

    # Each instrument owns one pillar (its primary curve, at its maturity): blocks of
    # same-date pillars are square by construction.
    unknowns_at: dict[date, list[int]] = defaultdict(list)
    for curve, (start, _) in solve.curve_slice.items():
        for k, d in enumerate(solve.free_dates[curve]):
            unknowns_at[d].append(start + k)
    instruments_at: dict[date, list] = defaultdict(list)
    for instr, d in zip(instruments, pillar_dates):
        instruments_at[d].append(instr)

    y = np.array(solve.y0, dtype=float)
    if date_by_date and solve.local:
        # Ascending sweeps (Gauss-Seidel): exact in one pass when every quote's
        # curves have pillars at its maturity; a quote reading another curve between
        # pillars depends on a later one, fixed by the next sweep.
        for _ in range(_MAX_SWEEPS):
            for d in sorted(unknowns_at):
                pos, block = unknowns_at[d], instruments_at[d]

                def block_residual(x, pos=pos, block=block):
                    y[pos] = x
                    return values(block, y)

                y[pos] = _solve_or_rollback(block_residual, list(y[pos]), solve.restore, what).x
            if max(abs(v) for v in values(instruments, y)) <= _SWEEP_TOL:
                return y
    return _solve_or_rollback(lambda v: values(instruments, v), list(y), solve.restore, what).x
