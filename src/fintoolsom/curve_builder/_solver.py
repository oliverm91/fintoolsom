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

from .. import settings
from ..derivatives.calculator import ValuationBatch
from ..market.currencies import Currency
from ..market.index import Index
from ..market.market import Market
from ..rates import DiscountCurve, IndexCurve, InterpolationMethod
from ._keys import CurveKey, _built, _curve_id

# A date-by-date bootstrap is accepted when every instrument reprices to this
# (report currency); otherwise the group falls back to one joint solve.
_SWEEP_TOL = 1e-6
# Interpolations where a pillar only moves the curve between its neighbours.
_LOCAL_METHODS = frozenset(
    {
        InterpolationMethod.PiecewiseConstant,
        InterpolationMethod.PiecewiseLinear,
    }
)
_MAX_SWEEPS = 4
# Interpolations where log(df) is affine in the forwards: the curve part of the
# Jacobian is then constant over a solve.
_LINEAR_METHODS = _LOCAL_METHODS | {InterpolationMethod.CubicSpline}
# Give hybr the Jacobian of the residuals instead of letting it difference them.
_ANALYTIC_JACOBIAN = True
# Forward bump that reads the curve part of the Jacobian off the curves: exact
# (up to rounding) on affine curves, a forward difference on the others.
_LINEAR_BUMP, _NONLINEAR_BUMP = 1e-2, 1e-7
# Solve the date blocks of a sweep by Newton (Broyden-updated), falling back to hybr.
_BLOCK_NEWTON = True
# Newton: forward-difference bump of the first Jacobian; a step of the forwards below
# _NEWTON_XTOL is the last (taken without evaluating it); one below _NEWTON_NOISE that
# no longer lowers the residuals has reached their rounding; at most _NEWTON_STEPS.
_NEWTON_BUMP, _NEWTON_XTOL, _NEWTON_NOISE, _NEWTON_STEPS = 1e-7, 1e-12, 1e-10, 8


def _newton(
    residual: Callable[[list[float]], list[float]],
    x0,
    jac: list[list[float]] | None = None,
) -> tuple[list[float], list[list[float]]] | None:
    """Solve a small square ``residual(x) = 0`` by Newton steps, Broyden-updated.

    The Jacobian is `jac` (one from an earlier solve of the same block) or a forward
    difference at `x0`; each step then updates it by Broyden's rank-one formula, so a
    step costs one evaluation. A sweep's blocks are nearly linear and well seeded:
    this takes 2 to 4 evaluations where hybr takes 5 to 7. Blocks have 1 to 3
    unknowns, so the algebra is on Python floats (numpy's per-call cost would exceed
    an evaluation's).

    Returns
    -------
    tuple or None
        The solution and the last Jacobian; None when the steps stop lowering the
        residuals before converging, or a residual is not finite (the caller falls
        back to hybr).
    """
    n = len(x0)
    x = [float(v) for v in x0]
    r = residual(x)
    if not all(map(math.isfinite, r)):
        return None
    if jac is None:
        jac = [[0.0] * n for _ in range(n)]
        for k in range(n):
            bumped = x.copy()
            bumped[k] += _NEWTON_BUMP
            for i, (up, base) in enumerate(zip(residual(bumped), r)):
                jac[i][k] = (up - base) / _NEWTON_BUMP
    else:
        jac = [row.copy() for row in jac]
    norm = max(map(abs, r))
    for _ in range(_NEWTON_STEPS):
        dx = _solve_small(jac, [-v for v in r])
        if dx is None:
            return None
        step = max(map(abs, dx))
        if not math.isfinite(step):
            return None
        if step <= _NEWTON_XTOL:
            return [a + b for a, b in zip(x, dx)], jac
        x_new = [a + b for a, b in zip(x, dx)]
        r_new = residual(x_new)
        norm_new = max(map(abs, r_new))
        if not math.isfinite(norm_new):
            return None
        if norm_new >= norm:
            # Not lower: converged to the residuals' rounding, or diverging.
            return (x, jac) if step <= _NEWTON_NOISE else None
        # Broyden: J += (dr - J dx) dx^T / (dx . dx)
        dd = math.fsum(v * v for v in dx)
        for i in range(n):
            row = jac[i]
            u = (r_new[i] - r[i] - math.fsum(row[k] * dx[k] for k in range(n))) / dd
            for k in range(n):
                row[k] += u * dx[k]
        x, r, norm = x_new, r_new, norm_new
    return None


def _solve_small(a: list[list[float]], b: list[float]) -> list[float] | None:
    """``a x = b`` by Gaussian elimination with partial pivoting; None if singular."""
    n = len(b)
    if n == 1:
        return [b[0] / a[0][0]] if a[0][0] != 0.0 else None
    m = [[*row, v] for row, v in zip(a, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(m[i][c]))
        if m[p][c] == 0.0:
            return None
        m[c], m[p] = m[p], m[c]
        pivot = m[c]
        for i in range(c + 1, n):
            f = m[i][c] / pivot[c]
            if f:
                row = m[i]
                for k in range(c, n + 1):
                    row[k] -= f * pivot[k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        row = m[i]
        x[i] = (row[n] - math.fsum(row[k] * x[k] for k in range(i + 1, n))) / row[i]
    return x


def _solve_or_rollback(
    residual: Callable[[list[float]], list[float]],
    y0: list[float],
    restore: Callable[[], None],
    what: str,
    jac: Callable[[list[float]], np.ndarray] | None = None,
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
    jac : callable, optional
        Jacobian of `residual`; None lets hybr difference it.

    Returns
    -------
    scipy.optimize.OptimizeResult

    Raises
    ------
    ValueError
        On non-convergence or non-finite residuals.
    """
    try:
        result = root(residual, y0, method="hybr", jac=jac)
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
    fixed part, and an atomic apply / rollback over ``market.curves``. Every curve is
    solved in the forward rate of each new segment.

    Parameters
    ----------
    market : Market
        Market whose curves are solved.
    t : date
        Curve date.
    method : InterpolationMethod
        Interpolation of new curves, discount and index alike. A curve already in the market keeps
        its own method when extended, so its existing part still interpolates as
        when it was solved.
    fixed_pillars : dict
        Curve key to its :class:`_FixedSegments` (knots, forwards and spot rate).
    free_dates : dict
        Curve key to free pillar dates.
    curve_slice : dict
        Curve key to its ``(start, end)`` slice of the unknown vector.
    y0 : list of float
        Initial guesses: segment forwards.
    """

    market: Market
    t: date
    method: InterpolationMethod
    fixed_pillars: dict
    free_dates: dict
    curve_slice: dict
    y0: list[float]
    _snapshot: dict = field(init=False, default=None)
    methods: dict = field(init=False, default=None)
    # Curves this solve built and still holds in the market: later iterations update
    # their forwards in place instead of building new curves.
    _owned: dict = field(init=False, default=None)

    @classmethod
    def from_free_pillars(
        cls,
        market: Market,
        t: date,
        method: InterpolationMethod,
        fixed_pillars: dict,
        free_dates: dict,
        free_seeds: dict,
    ) -> _GroupSolve:
        """Lay out the unknown vector from the free dates and their seeds.

        Parameters
        ----------
        market, t, method, fixed_pillars, free_dates
            As in the class.
        free_seeds : dict
            Curve key to the initial segment forwards of its free dates.

        Returns
        -------
        _GroupSolve
        """
        curve_slice: dict[CurveKey, tuple[int, int]] = {}
        y0: list[float] = []
        start = 0
        for curve, dates in free_dates.items():
            curve_slice[curve] = (start, start + len(dates))
            y0 += list(free_seeds[curve])
            start += len(dates)
        return cls(market, t, method, fixed_pillars, free_dates, curve_slice, y0)

    def __post_init__(self):
        self._snapshot = {c: _built(self.market, c) for c in self.curve_slice}
        self._owned = {}
        # Interpolation of each rebuilt curve: an existing curve keeps its own.
        self.methods = {
            c: prev.interpolation_method if prev is not None else self.method
            for c, prev in self._snapshot.items()
        }

    @property
    def local(self) -> bool:
        """Whether every curve solved interpolates locally (piecewise-constant or
        linear forwards): an instrument then depends only on pillars up to its own
        maturity."""
        return all(m in _LOCAL_METHODS for m in self.methods.values())

    @property
    def n_unknowns(self) -> int:
        """Number of free pillars."""
        return len(self.y0)

    def apply(self, y) -> None:
        """Rebuild every group curve from its fixed part plus the unknowns `y`.

        Local curves are read at few dates per rebuild, so a daily grid would cost more
        than it saves. A Python spline curve is read everywhere and pays for its grid
        (scipy's evaluation is slow per call); a C++ one evaluates its formula as fast
        as it reads a grid, so filling one per rebuild is wasted (about 25% of the
        build). The builder adds the grid to the final curves either way.

        The first call builds the curves; later ones set the new forwards on them in
        place (:meth:`PillarCurve._set_forward_rates`), which gives the same curves
        without constructing and checking them again. Curves this solve did not build
        (the snapshot, or one replaced in the market since) are never changed.
        """
        for curve, (s, e) in self.curve_slice.items():
            self.set_curve(curve, y[s:e])

    def set_curve(self, curve: CurveKey, values) -> None:
        """Rebuild one group curve from its fixed part plus its unknowns `values`."""
        fixed = self.fixed_pillars[curve]
        forwards = [*fixed.forwards, *(float(v) for v in values)]
        owned = self._owned.get(curve)
        if owned is not None and _built(self.market, curve) is owned:
            owned._set_forward_rates(forwards)
            return
        precalculate = not self.local and not settings.is_cpp("rates")
        built = self._new_curve(curve, forwards, precalculate)
        self.market.set_curve(built)
        self._owned[curve] = built

    def private_curve(self, curve: CurveKey, values) -> DiscountCurve | IndexCurve:
        """A new curve as :meth:`set_curve` would build it from `values`, without a
        grid and outside the market (to bump freely)."""
        fixed = self.fixed_pillars[curve]
        return self._new_curve(
            curve, [*fixed.forwards, *(float(v) for v in values)], False
        )

    def _new_curve(
        self, curve: CurveKey, forwards, precalculate: bool
    ) -> DiscountCurve | IndexCurve:
        fixed = self.fixed_pillars[curve]
        knots = [*fixed.knots, *self.free_dates[curve]]
        if isinstance(curve, Index):
            return IndexCurve(
                self.t,
                curve,
                knots,
                forwards,
                fixed.spot_rate,
                self.methods[curve],
                precalculate=precalculate,
            )
        return DiscountCurve(
            self.t,
            curve,
            self.market.risk_free_index,
            knots,
            forwards,
            self.methods[curve],
            precalculate=precalculate,
        )

    def restore(self) -> None:
        """Put back the curves as they were before the solve."""
        self._owned.clear()
        for curve, previous in self._snapshot.items():
            if previous is None:
                self.market.curves.pop(
                    _curve_id(curve, self.market.risk_free_index), None
                )
            else:
                self.market.set_curve(previous)


class _Jacobian:
    """``d residuals / d y[columns]`` of one batch, for hybr.

    By the chain rule through the values ``v`` the batch reads off the market:
    ``d r / d y = (d r / d v) (d v / d y)``. The first factor is the batch's analytic
    :meth:`~ValuationBatch.gradient`. For the second, every value is a product of
    curve dfs to the powers +-1, so ``g = log(v + shift)`` is a signed sum of
    ``log df`` of the curves solved (:meth:`_source_terms`), and
    ``d v / d y = (v + shift) d g / d y``.

    ``d log df / d y`` of each curve solved is read off a private copy of it, bumping
    one forward at a time, at the dates the batch reads. It is constant over the solve
    on piecewise-constant/linear and spline curves, where ``log df`` is affine in the
    forwards (the bump is then exact up to rounding), and a forward difference
    recomputed at each call on the monotone methods. A C++ batch, valued in one
    ``market_valuate_many`` call, is compiled again in Python for its gradient.

    Parameters
    ----------
    solve : _GroupSolve
        Layout of the unknowns, and the curves they rebuild.
    batch : ValuationBatch
        Batch of the residual instruments.
    columns : array of int
        Unknowns the Jacobian is taken with respect to.
    """

    def __init__(self, solve: _GroupSolve, batch: ValuationBatch, columns):
        self.solve = solve
        self.columns = np.asarray(columns, dtype=np.intp)
        self.plan = (
            batch
            if batch.differentiable
            else ValuationBatch(
                batch.instruments, batch.market, batch.currency, compile_all=True
            )
        )
        self.ok = self.plan.differentiable
        self.linear = all(m in _LINEAR_METHODS for m in solve.methods.values())
        self._terms = None
        self._dg = None

    def __call__(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        self.solve.apply(y)
        v = self.plan.source_values()
        if self._terms is None:
            self._terms = self._source_terms()
        dg = self._dg if self._dg is not None else self._curve_part(y, len(v))
        if self.linear:
            self._dg = dg
        return self.plan.gradient(v) @ ((v + self.plan.log_shift)[:, None] * dg)

    def _source_terms(self) -> dict:
        """Curve solved to ``(ordinals, [(value rows, ordinal rows, sign)])``: the
        ``log df`` terms of ``log(v + shift)``, as the market computes each source."""
        market = self.solve.market
        key_of = {id(c): key for key, c in self.solve._owned.items()}
        terms: dict = defaultdict(list)

        def add(curve, rows, ordinals, sign):
            key = key_of.get(id(curve))
            if key is not None and len(rows):
                terms[key].append((rows, np.asarray(ordinals, dtype=np.int64), sign))

        offset = 0
        for source in self.plan._sources_list:
            rows = offset + np.arange(source.size)
            offset += source.size
            kind, args, ords = source.kind, source.args, source.ordinals
            if kind == "discount":  # Market._dfs_at_ordinals
                currency, collateral = args
                if collateral is None:
                    add(market.get_discount_curve(currency), rows, ords[0], 1.0)
                elif collateral.currency == currency:
                    add(market.get_index_curve(collateral), rows, ords[0], 1.0)
                else:
                    add(market.get_discount_curve(currency), rows, ords[0], 1.0)
                    add(market.get_index_curve(collateral), rows, ords[0], 1.0)
                    add(
                        market.get_discount_curve(collateral.currency),
                        rows,
                        ords[0],
                        -1.0,
                    )
            elif kind == "index":
                add(market.get_index_curve(args[0]), rows, ords[0], 1.0)
            elif kind == "accrual":  # 1 + accrual = df(start) / df(end)
                curve = market.get_index_curve(args[0])
                add(curve, rows, ords[0], 1.0)
                add(curve, rows, ords[1], -1.0)
            else:  # Market._future_fx_rates_at_ordinals: anchor * df ratios
                pair = args[0]
                base, quote = pair.base_currency, pair.quote_currency
                if base == quote:
                    continue
                last = market.get_fx_known_until(pair).toordinal()
                future = ords[0] > last
                rows, dates = rows[future], ords[0][future]
                anchor = np.full(len(dates), last)
                for currency, sign in ((base, 1.0), (quote, -1.0)):
                    curve = market.get_discount_curve(currency)
                    add(curve, rows, dates, sign)
                    add(curve, rows, anchor, -sign)
        out = {}
        for key, entries in terms.items():
            ordinals, inverse = np.unique(
                np.concatenate([o for _, o, _ in entries]), return_inverse=True
            )
            split = np.cumsum([len(o) for _, o, _ in entries])[:-1]
            out[key] = (
                ordinals,
                [
                    (rows, at, sign)
                    for (rows, _, sign), at in zip(entries, np.split(inverse, split))
                ],
            )
        return out

    def _curve_part(self, y: np.ndarray, n_values: int) -> np.ndarray:
        """``d log(v + shift) / d y[columns]``."""
        solve = self.solve
        h = _LINEAR_BUMP if self.linear else _NONLINEAR_BUMP
        position = {int(k): j for j, k in enumerate(self.columns)}
        dg = np.zeros((n_values, len(self.columns)))
        for key, (ordinals, entries) in self._terms.items():
            s, e = solve.curve_slice[key]
            free = [k for k in range(s, e) if k in position]
            if not free:
                continue
            curve = solve.private_curve(key, y[s:e])
            forwards = np.array(curve.forward_rates, dtype=float)
            first = len(forwards) - (e - s)  # the free forwards come last
            base = np.log(curve._dfs_at_ordinals(ordinals))
            dlog = np.empty((len(ordinals), len(free)))
            for j, k in enumerate(free):
                bumped = forwards.copy()
                bumped[first + k - s] += h
                curve._set_forward_rates(bumped)
                dlog[:, j] = (np.log(curve._dfs_at_ordinals(ordinals)) - base) / h
            cols = [position[k] for k in free]
            for rows, at, sign in entries:
                dg[np.ix_(rows, cols)] += sign * dlog[at]
        return dg


def _solve_unknowns(
    solve: _GroupSolve,
    instruments,
    pillar_dates,
    report_currency: Currency,
    what: str,
    *,
    date_by_date: bool = True,
) -> np.ndarray:
    """Bootstrap the free pillars date by date, falling back to one joint solve.

    With local interpolation (piecewise-constant or linear forwards) an instrument depends only on pillars up to its own last free pillar, so
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

    batches: dict[int, ValuationBatch] = {}

    def batch_of(instrs) -> ValuationBatch:
        """The batch of `instrs` (one list each: the batch is compiled once)."""
        batch = batches.get(id(instrs))
        if batch is None:
            batch = batches[id(instrs)] = ValuationBatch(
                instrs, solve.market, report_currency
            )
        return batch

    def values(instrs, y) -> list[float]:
        """The residuals of `instrs`."""
        solve.apply(y)
        return batch_of(instrs).values().tolist()

    def jacobian(instrs) -> _Jacobian | None:
        """The Jacobian of `instrs` in every unknown, None to let hybr difference it."""
        if not _ANALYTIC_JACOBIAN:
            return None
        jac = _Jacobian(solve, batch_of(instrs), np.arange(solve.n_unknowns))
        return jac if jac.ok else None

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
    block_jacs: dict[date, list] = {}  # Newton's last Jacobian of each block
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

                # A block has 1-3 unknowns: its Jacobian is differenced (hybr's, or
                # Newton's first), cheaper than the analytic one's setup.
                x0 = y[pos].copy()
                solved = None
                if _BLOCK_NEWTON:
                    try:
                        solved = _newton(block_residual, x0, block_jacs.get(d))
                    except Exception:  # hybr below raises it as it would
                        solved = None
                if solved is not None:
                    y[pos], block_jacs[d] = solved
                else:
                    y[pos] = x0
                    y[pos] = _solve_or_rollback(
                        block_residual, list(x0), solve.restore, what
                    ).x
            if max(abs(v) for v in values(instruments, y)) <= _SWEEP_TOL:
                return y
    jac = jacobian(instruments)
    return _solve_or_rollback(
        lambda v: values(instruments, v), list(y), solve.restore, what, jac
    ).x
