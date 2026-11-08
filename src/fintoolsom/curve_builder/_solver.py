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
from ..market.market import CurveKey, Market
from ..rates import InterpolationMethod, ProjectionCurve, ProjectionInterpolationMethod, ZeroCouponCurve
from ._keys import _Proj
from ._seeds import _DAY_BASE

# A date-by-date bootstrap is accepted when every instrument reprices to this
# (report currency); otherwise the group falls back to one joint solve.
_SWEEP_TOL = 1e-6
_MAX_SWEEPS = 4


def _set_or_pop(store: dict, key, previous) -> None:
    """Restore ``store[key]`` to `previous`; None means the key was absent."""
    if previous is None:
        store.pop(key, None)
    else:
        store[key] = previous


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
    fixed pillars, and an atomic apply / rollback over ``market.curves`` and
    ``market.projection_curves``. Discount curves are solved in ``log(df)``, which
    removes the collinearity between near-1 short dfs and small long dfs;
    projection curves (``_Proj`` keys) in the forward rate of each new segment.

    Parameters
    ----------
    market : Market
        Market whose curves are solved.
    t : date
        Curve date.
    method : InterpolationMethod
        Interpolation of the rebuilt curves.
    fixed_pillars : dict
        Curve key to fixed ``(date, df)`` pillars; for a ``_Proj`` key, its fixed
        ``(knot_dates, forward_rates)`` (spot anchor and already solved segments).
    free_dates : dict
        Curve key to free pillar dates.
    curve_slice : dict
        Curve key to its ``(start, end)`` slice of the unknown vector.
    y0 : list of float
        Initial guesses: ``log(df)``, or forward rates for ``_Proj`` keys.
    projection_method : ProjectionInterpolationMethod
        Interpolation of the rebuilt projection curves.
    """

    market: Market
    t: date
    method: InterpolationMethod
    fixed_pillars: dict
    free_dates: dict
    curve_slice: dict
    y0: list[float]
    projection_method: ProjectionInterpolationMethod = ProjectionInterpolationMethod.PiecewiseConstant
    _snapshot: dict = field(init=False, default=None)

    @classmethod
    def from_free_pillars(
        cls,
        market: Market,
        t: date,
        method: InterpolationMethod,
        fixed_pillars: dict,
        free_dates: dict,
        free_seed_dfs: dict,
        projection_method: ProjectionInterpolationMethod = ProjectionInterpolationMethod.PiecewiseConstant,
    ) -> _GroupSolve:
        """Lay out the unknown vector from the free dates and their seed dfs.

        Parameters
        ----------
        market, t, method, fixed_pillars, free_dates
            As in the class.
        free_seed_dfs : dict
            Curve key to initial dfs of its free dates (forward rates for ``_Proj``).
        projection_method : ProjectionInterpolationMethod
            As in the class.

        Returns
        -------
        _GroupSolve
        """
        curve_slice: dict[CurveKey, tuple[int, int]] = {}
        y0: list[float] = []
        start = 0
        for curve, dates in free_dates.items():
            curve_slice[curve] = (start, start + len(dates))
            # Discount curves: log(df) unknowns; projection curves: segment forwards.
            seeds = free_seed_dfs[curve]
            y0 += list(seeds) if isinstance(curve, _Proj) else [math.log(df) for df in seeds]
            start += len(dates)
        return cls(market, t, method, fixed_pillars, free_dates, curve_slice, y0, projection_method)

    def _store(self, curve) -> dict:
        """Market dict holding `curve`."""
        return self.market.projection_curves if isinstance(curve, _Proj) else self.market.curves

    @staticmethod
    def _key(curve):
        """Key of `curve` in its market dict."""
        return curve.index if isinstance(curve, _Proj) else curve

    def __post_init__(self):
        self._snapshot = {c: self._store(c).get(self._key(c)) for c in self.curve_slice}

    @property
    def n_unknowns(self) -> int:
        """Number of free pillars."""
        return len(self.y0)

    def apply(self, y) -> None:
        """Rebuild every group curve from its fixed pillars plus the unknowns `y`."""
        for curve, (s, e) in self.curve_slice.items():
            if isinstance(curve, _Proj):
                knots, forwards = self.fixed_pillars[curve]
                self.market.projection_curves[curve.index] = ProjectionCurve(
                    self.t, [*knots, *self.free_dates[curve]], [*forwards, *(float(v) for v in y[s:e])],
                    self.projection_method, _DAY_BASE,
                )
                continue
            free = zip(self.free_dates[curve], (math.exp(v) for v in y[s:e]))
            date_dfs = sorted([*self.fixed_pillars[curve], *free], key=lambda p: p[0])
            self.market.curves[curve] = ZeroCouponCurve(
                self.t, date_dfs=date_dfs, df_interpolation_method=self.method
            )

    def restore(self) -> None:
        """Put back the curves as they were before the solve."""
        for curve, previous in self._snapshot.items():
            _set_or_pop(self._store(curve), self._key(curve), previous)


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
    local = solve.method is InterpolationMethod.LogLinear  # forwards: piecewise-constant
    if date_by_date and local:
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
