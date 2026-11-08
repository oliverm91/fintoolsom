from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np
from scipy.optimize import root

from fintoolsom.rates.Rates import RateConvention  # type: ignore[import-untyped]

from ..market.index import Index
from ..market.currencies import Currency, CurrencyName
from ..derivatives.calculator import Calculator
from ..market.market import CurveKey, Market, funding_key
from ..market.index_history import OvernightHistory
from ..market.quotes import InstrumentQuote, CrossCurrencyFloatFloatQuote, ForwardPointsQuote
from ..derivatives.swaps import Swap, FixedLeg, FloatingLeg
from ..derivatives.forwards import Forward, NDF
from ..rates import (
    ZeroCouponCurve,
    InterpolationMethod,
    ProjectionCurve,
    ProjectionInterpolationMethod,
)


# Curve-solve residuals are valued in one reporting currency; a par/MTM==0 residual
# is zero in any currency, so the choice is arbitrary — CLP by convention.
_REPORT_CURRENCY = Currency(CurrencyName.CLP)
_DAY_BASE = 365
# A date-by-date bootstrap is accepted when every instrument reprices to this
# (report currency); otherwise the group falls back to one joint solve.
_SWEEP_TOL = 1e-6
# Quote types whose instrument needs the market FX to fully materialise: XCCY (to size
# matched leg notionals) and forward-points (to build the outright strike off the spot).
_FX_DEPENDENT = (CrossCurrencyFloatFloatQuote, ForwardPointsQuote)


class InsufficientQuotesError(Exception):
    """A curve group is under-determined: more free pillars than instruments."""


@dataclass(frozen=True)
class _Proj:
    """Solve key of the independent forward curve of a projection-only index.

    Discount curves are keyed by Index or Currency (``market.curves``); a
    projection-only index (e.g. a term index such as TAB 3M) is solved under this
    key and stored in ``market.projection_curves``. Both kinds go through the same
    scheduler, so a projection curve can depend on, and be needed by, other curves.
    """

    index: Index

    @property
    def currency(self) -> Currency:
        return self.index.currency


def _instrument_of(quote: InstrumentQuote, market: Market):
    """Instrument described by `quote`.

    FX-dependent quotes receive `market` so their notionals or strike resolve off
    the spot; the others do not take it.

    Parameters
    ----------
    quote : InstrumentQuote
        Quote to materialise.
    market : Market
        Market at the quote date.

    Returns
    -------
    Forward, NDF or Swap
    """
    return quote.get_instrument(market) if isinstance(quote, _FX_DEPENDENT) else quote.get_instrument()


def _set_or_pop(store: dict, key, previous) -> None:
    """Restore ``store[key]`` to `previous`; None means the key was absent."""
    if previous is None:
        store.pop(key, None)
    else:
        store[key] = previous


# ── Curve dependency analysis ───────────────────────────────────────────────

def _discount_indexes(quote: InstrumentQuote, riskless_index: Index, extra: frozenset[Index]) -> set[Index]:
    """Indexes whose own curve discounts for `quote`.

    Parameters
    ----------
    quote : InstrumentQuote
        Quote considered.
    riskless_index : Index
        Market funding index.
    extra : frozenset of Index
        Other discount indexes (other quotes' collateral, index curves in market).

    Returns
    -------
    set of Index
        The riskless index, the quote's collateral and `extra`.
    """
    collateral: Index | None = getattr(quote, "collateral_index", None)
    return {riskless_index, *extra, *([collateral] if collateral is not None else [])}


def _curve_keys(
    quote: InstrumentQuote,
    riskless_index: Index,
    *,
    cancel_collateral: bool,
    discount_indexes: frozenset[Index] = frozenset(),
) -> frozenset[CurveKey]:
    """Curve keys `quote` relates to.

    Each leg discounts in its settlement currency (the swap payment currency for a
    non-deliverable leg in another currency, else its own). Under collateral C the
    discount factor is ``funding(ccy) * curve(C) / funding(C.ccy)``; when
    ``ccy == C.ccy`` the funding factors cancel. A non-deliverable leg also reads
    the funding curves of both currencies to project its FX fixings. A floating
    index is its own key when its curve discounts, else ``_Proj(index)``: a
    projection-only forward curve.

    Parameters
    ----------
    quote : InstrumentQuote
        Quote considered.
    riskless_index : Index
        Market funding index.
    cancel_collateral : bool
        True for the curves the quote pins (collateral cancellation applied,
        :func:`_curves_needed`); False for every curve that must exist to value
        it (:func:`_curves_touched`), used to order the build.
    discount_indexes : frozenset of Index, optional
        Other discount indexes, see :func:`_discount_indexes`.

    Returns
    -------
    frozenset of Index or Currency
    """
    instrument = quote.get_instrument()
    collateral: Index | None = getattr(quote, "collateral_index", None)
    discounting = _discount_indexes(quote, riskless_index, discount_indexes)
    keys: set[CurveKey] = set()

    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            pay_ccy = _settlement_currency(instrument, leg)
            if collateral is None:
                keys.add(funding_key(riskless_index, pay_ccy))
            else:
                if not (cancel_collateral and pay_ccy == collateral.currency):
                    keys.add(funding_key(riskless_index, pay_ccy))
                    keys.add(funding_key(riskless_index, collateral.currency))
                keys.add(collateral)
            if pay_ccy != leg.currency:  # FX fixings projected on both funding curves
                keys.add(funding_key(riskless_index, leg.currency))
                keys.add(funding_key(riskless_index, pay_ccy))
            if isinstance(leg, FloatingLeg):
                keys.add(leg.index if leg.index in discounting else _Proj(leg.index))
    else:  # Forward / NDF (a UF forward is an NDF on CLF/CLP: CLP and CLF funding)
        pair = instrument.currency_pair
        for currency in (pair.base_currency, pair.quote_currency):
            keys.add(funding_key(riskless_index, currency))

    return frozenset(keys)


def _curves_needed(
    quote: InstrumentQuote, riskless_index: Index, discount_indexes: frozenset[Index] = frozenset()
) -> frozenset[CurveKey]:
    """Curves `quote` pins (collateral cancellation applied); groups instruments.

    See :func:`_curve_keys` for the parameters.
    """
    return _curve_keys(quote, riskless_index, cancel_collateral=True, discount_indexes=discount_indexes)


def _curves_touched(
    quote: InstrumentQuote, riskless_index: Index, discount_indexes: frozenset[Index] = frozenset()
) -> frozenset[CurveKey]:
    """Every curve that must exist to value `quote`; orders the build.

    A superset of :func:`_curves_needed`. See :func:`_curve_keys` for the parameters.
    """
    return _curve_keys(quote, riskless_index, cancel_collateral=False, discount_indexes=discount_indexes)


def _settlement_currency(swap: Swap, leg) -> Currency:
    """Currency `leg` is paid and discounted in.

    See :meth:`Swap.leg_settlement_currency`: a UF (CLF) leg always settles in CLP.
    """
    return swap.leg_settlement_currency(leg)


def _curve_currency(curve_key: CurveKey) -> Currency:
    """Currency of the flows `curve_key` discounts (an index curve: its currency)."""
    return curve_key if isinstance(curve_key, Currency) else curve_key.currency


def _get_maturity(curve_key: CurveKey, instrument, riskless_index: Index) -> date | None:
    """Pillar `instrument` pins on `curve_key`: its latest relevant date.

    Payment dates for the settlement-currency funding curve, FX fixing dates for
    the leg-currency funding curve of a non-deliverable leg, end dates for a
    discounting floating index; the fixing date (NDF) or payment date (forward)
    for an FX forward.

    Parameters
    ----------
    curve_key : Index or Currency
        Curve considered.
    instrument : Swap or Forward
        Instrument considered.
    riskless_index : Index
        Market funding index.

    Returns
    -------
    date or None
        None when the instrument does not touch `curve_key`.

    Raises
    ------
    TypeError
        For an unsupported instrument type.
    """
    candidates: set[date] = set()
    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            pay_ccy = _settlement_currency(instrument, leg)
            if funding_key(riskless_index, pay_ccy) == curve_key:
                candidates.add(max(leg.payment_dates))
            if pay_ccy != leg.currency and funding_key(riskless_index, leg.currency) == curve_key:
                candidates.add(max(leg.fx_fixing_dates))
            if isinstance(leg, FloatingLeg) and curve_key in (leg.index, _Proj(leg.index)):
                candidates.add(max(leg.end_dates))
    elif isinstance(instrument, Forward):
        pair = instrument.currency_pair
        for currency in (pair.base_currency, pair.quote_currency):
            if funding_key(riskless_index, currency) == curve_key:
                candidates.add(instrument.fixing_date if isinstance(instrument, NDF) else instrument.payment_date)
    else:
        raise TypeError(
            f"_get_maturity only supports Swap and Forward instruments, got {type(instrument).__name__}."
        )
    return max(candidates) if candidates else None


# ── Shared atomic solve ─────────────────────────────────────────────────────

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


# ── Orchestrator ────────────────────────────────────────────────────────────

class _CurveBuilder:
    """One curve build: every discount and projection curve, by one scheduler.

    Quotes are grouped by the curves they pin; a fixpoint scheduler solves each
    group once the curves it only reads exist. Discount curves go to
    ``market.curves``, the forward curves of projection-only indexes to
    ``market.projection_curves``. Every index ends with exactly one curve. Mutates
    `market` in place.

    Parameters
    ----------
    quotes : list of InstrumentQuote
        Quotes to reprice.
    market : Market
        Market to build into; needs a `riskless_index`.
    discount_method : InterpolationMethod
        Interpolation of the discount curves.
    projection_method : ProjectionInterpolationMethod
        Interpolation of the projection curves.

    Raises
    ------
    ValueError
        If the market has no `riskless_index`.
    """

    # Bootstrap each group date by date when its interpolation allows it; False always
    # runs one joint solve per group (the reference the date-by-date sweep must match).
    date_by_date: bool = True

    def __init__(
        self,
        quotes: list,
        market: Market,
        discount_method: InterpolationMethod,
        projection_method: ProjectionInterpolationMethod,
    ):
        if market.riskless_index is None:
            raise ValueError("Market has no riskless_index; curve building needs one.")
        self.quotes = quotes
        self.riskless_index = riskless_index = market.riskless_index
        self.market = market
        self.t = market.t
        self.discount_method = discount_method
        self.projection_method = projection_method

        # Indexes whose curve discounts: the riskless index, any quote's collateral, and
        # any index curve already in the market. Other floating indexes are projection-only.
        collaterals = {c for q in quotes if (c := getattr(q, "collateral_index", None)) is not None}
        self.discount_indexes: frozenset[Index] = frozenset(
            {riskless_index, *collaterals, *(k for k in market.curves if isinstance(k, Index))}
        )
        # pinned  (_curves_needed): the curves each quote determines — groups instruments.
        # touched (_curves_touched): every curve that must exist to value it — orders the build.
        self.pinned = {q: _curves_needed(q, riskless_index, self.discount_indexes) for q in quotes}
        self.touched = {q: _curves_touched(q, riskless_index, self.discount_indexes) for q in quotes}
        self.groups: dict[frozenset, list[InstrumentQuote]] = defaultdict(list)
        for q in quotes:
            self.groups[self.pinned[q]].append(q)

    def build(self) -> None:
        """Solve every curve group with a fixpoint scheduler.

        Each pass solves the groups that are valuable and square (smaller first),
        building the curves other groups depend on.

        Raises
        ------
        InsufficientQuotesError
            If a pass makes no progress: the remaining groups are under-determined.
        """
        pending = [g for g in self.groups if g]
        while pending:
            solved_any = False
            for group in sorted(pending, key=len):
                if self._attempt(group) in ("solved", "nothing"):
                    pending.remove(group)
                    solved_any = True
            if not solved_any:
                raise InsufficientQuotesError(
                    "Curve build stalled: the remaining groups are under-determined given "
                    f"the available quotes: {[set(g) for g in pending]}."
                )

    def _attempt(self, group: frozenset) -> str:
        """Try to solve `group` against the current market curves.

        Parameters
        ----------
        group : frozenset of Index, Currency or _Proj
            Curves the group pins.

        Returns
        -------
        str
            'solved'; 'nothing' when all its pillars exist; 'defer' when a curve it
            reads is unbuilt or the system is not square yet.
        """
        instruments = [_instrument_of(q, self.market) for q in self.groups[group]]

        # Valuable? Every curve these instruments touch but do NOT pin must already
        # exist, else valuation would hit a missing curve.
        group_touched = set().union(*(self.touched[q] for q in self.groups[group]))
        if not (group_touched - group) <= self._existing_keys():
            return "defer"
        return self._solve_group(group, instruments)

    def _existing_keys(self) -> set:
        """Keys of every curve already in the market (projection curves as _Proj)."""
        return {*self.market.curves, *(_Proj(i) for i in self.market.projection_curves)}

    def _solve_group(self, group, instruments, report_currency: Currency = _REPORT_CURRENCY) -> str:
        """Solve the free pillars of `group` so `instruments` reprice.

        Parameters
        ----------
        group : frozenset of Index, Currency or _Proj
            Curves the group pins.
        instruments : list
            Instruments of the group.
        report_currency : Currency, optional
            Currency the residuals are valued in.

        Returns
        -------
        str
            As :meth:`_attempt`.
        """
        fixed_pillars, free_dates, free_seed_dfs = self._partition_pillars(group, instruments)
        if not free_dates:
            return "nothing"  # every pillar this group needs already exists.
        if not set(group) <= self._existing_keys() | set(free_dates):
            return "defer"  # a curve it reads but gives no pillar to is not built yet.

        residual_instruments = self._residual_instruments(instruments, free_dates)
        solve = _GroupSolve.from_free_pillars(
            self.market, self.t, self.discount_method, fixed_pillars, free_dates, free_seed_dfs,
            self.projection_method,
        )
        if solve.n_unknowns != len(residual_instruments):
            # Not square yet (a curve it reads as fixed may still be unbuilt) — defer;
            # if nothing ever makes it square the scheduler raises InsufficientQuotesError.
            return "defer"

        y = self._solve_unknowns(
            solve, residual_instruments, free_dates, report_currency, f"Curve solve for group {set(free_dates)}"
        )
        solve.apply(y)  # root's last f-eval is not guaranteed to be at the solution.
        return "solved"

    def _solve_unknowns(self, solve, instruments, free_dates, report_currency, what) -> np.ndarray:
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
        free_dates : dict
            Curve key to free pillar dates.
        report_currency : Currency
            Currency the residuals are valued in.
        what : str
            Description used in error messages.

        Returns
        -------
        numpy.ndarray
            Solved unknowns.
        """
        def values(instrs, y) -> list[float]:
            solve.apply(y)
            return [Calculator.valuate(i, self.market, report_currency) for i in instrs]

        unknowns_at: dict[date, list[int]] = defaultdict(list)
        for curve, (start, _) in solve.curve_slice.items():
            for k, d in enumerate(free_dates[curve]):
                unknowns_at[d].append(start + k)
        instruments_at: dict[date, list] = defaultdict(list)
        for instr in instruments:
            last = max(
                m for c, dates in free_dates.items()
                if (m := _get_maturity(c, instr, self.riskless_index)) is not None and m in dates
            )
            instruments_at[last].append(instr)

        y = np.array(solve.y0, dtype=float)
        local = self.discount_method is InterpolationMethod.LogLinear  # forwards: piecewise-constant
        if self.date_by_date and local and all(len(instruments_at[d]) == len(pos) for d, pos in unknowns_at.items()):
            for d in sorted(unknowns_at):
                pos, block = unknowns_at[d], instruments_at[d]

                def block_residual(x, pos=pos, block=block):
                    y[pos] = x
                    return values(block, y)

                y[pos] = _solve_or_rollback(block_residual, list(y[pos]), solve.restore, what).x
            if max(abs(v) for v in values(instruments, y)) <= _SWEEP_TOL:
                return y
        return _solve_or_rollback(lambda v: values(instruments, v), list(y), solve.restore, what).x

    def _partition_pillars(self, group, instruments):
        """Split each pinned curve's maturities into fixed and free pillars.

        Fixed pillars are already solved, or a short-end anchor; free maturities are
        the new unknowns. A non-pillar curve (e.g. NSSCurve) is read as fixed.

        Parameters
        ----------
        group : frozenset of Index or Currency
            Curves the group pins.
        instruments : list
            The group's instruments.

        Returns
        -------
        fixed_pillars : dict
            Curve key to fixed ``(date, df)`` pillars.
        free_dates : dict
            Curve key to free pillar dates.
        free_seed_dfs : dict
            Curve key to initial dfs of the free pillars.
        """
        fixed_pillars: dict[CurveKey, list[tuple[date, float]]] = {}
        free_dates: dict[CurveKey, list[date]] = {}
        flat_guess: dict[CurveKey, float] = {}
        for curve in group:
            maturities = {_get_maturity(curve, instr, self.riskless_index) for instr in instruments}
            maturities.discard(None)
            if not maturities:
                continue

            if isinstance(curve, _Proj):
                built = self.market.projection_curves.get(curve.index)
                if built is None:
                    knots, forwards = self._projection_anchor(curve.index)
                else:
                    knots, forwards = list(built.knot_dates), [float(f) for f in built.forward_rates]
                fixed_pillars[curve] = (knots, forwards)
                if new_dates := sorted(m for m in maturities if m > knots[-1]):
                    free_dates[curve] = new_dates
                continue

            if curve not in self.market.curves:
                # New curve: every maturity is free, after the overnight anchor if any.
                anchor, flat_guess[curve] = self._short_end_anchor(curve)
                fixed_pillars[curve] = [anchor] if anchor is not None else []
                free_dates[curve] = sorted(maturities)
            elif isinstance(built := self.market.curves[curve], ZeroCouponCurve):
                # Existing curve: keep solved pillars, only extend past its last one;
                # maturities inside the range are read off by interpolation (no unknown).
                existing = list(zip(built.dates, built.dfs))
                last = max(d for d, _ in existing)
                new_dates = sorted(m for m in maturities if m > last)
                fixed_pillars[curve] = existing
                if new_dates:
                    free_dates[curve] = new_dates
            # else: a non-pillar curve (e.g. NSSCurve) is defined at every date and has
            # no pillars to extend — it is read as fixed, never solved.
        discount_free = {c: d for c, d in free_dates.items() if not isinstance(c, _Proj)}
        free_seed_dfs = self._seed_free_pillars(instruments, fixed_pillars, discount_free, flat_guess)
        for curve, dates in free_dates.items():
            if isinstance(curve, _Proj):  # seed new segments flat at the last known forward
                free_seed_dfs[curve] = [fixed_pillars[curve][1][-1]] * len(dates)
        return fixed_pillars, free_dates, free_seed_dfs

    # -- seeds ----------------------------------------------------------------

    def _tau(self, d: date) -> float:
        """Act/365 year fraction from the curve date to `d`."""
        return (d - self.t).days / _DAY_BASE

    def _seed_free_pillars(self, instruments, fixed_pillars, free_dates, flat_guess):
        """Initial discount factors of the free pillars, read off the quotes.

        Each free pillar gets a continuously compounded zero rate implied by an
        instrument that pins it, then the remaining ones are filled in:

        1. A fixed leg in the curve's currency: its coupons compounded from the
           curve date to the leg end (par rate ~ zero rate).
        2. An FX forward or NDF whose other currency's curve is known (built, or
           seeded in step 1): covered interest parity on its strike.
        3. A float-float (basis) swap whose other curves are known: each floating
           leg is worth about par + (z_index + spread - z_discount) * annuity, and
           the swap is at par.
        4. Other pillars: zero rate interpolated linearly in time between the
           known points of the same curve (short-end anchor, existing pillars,
           steps 1-3), flat beyond them; the flat guess of
           :meth:`_short_end_anchor` when the curve has no point at all; an existing
           curve's own extrapolation when it is being extended.

        The seed only needs to be close: the solve then reprices exactly.

        Parameters
        ----------
        instruments : list
            The group's instruments.
        fixed_pillars : dict
            Curve key to fixed ``(date, df)`` pillars.
        free_dates : dict
            Curve key to free pillar dates.
        flat_guess : dict
            New curve key to its flat fallback rate (annual compounding).

        Returns
        -------
        dict
            Curve key to initial dfs of its free pillars, in `free_dates` order.
        """
        implied: dict[CurveKey, dict[date, float]] = defaultdict(dict)

        def pinned_free_date(curve, instr) -> date | None:
            m = _get_maturity(curve, instr, self.riskless_index)
            return m if m is not None and m in free_dates[curve] and m not in implied[curve] else None

        # 1. Fixed legs.
        for instr in instruments:
            if not isinstance(instr, Swap):
                continue
            for leg in (instr.receive_leg, instr.pay_leg):
                if not isinstance(leg, FixedLeg) or (z := self._fixed_leg_zero(leg)) is None:
                    continue
                for curve in free_dates:
                    if _curve_currency(curve) == leg.currency and (m := pinned_free_date(curve, instr)):
                        implied[curve][m] = z

        # 2. FX forwards, against the other currency's known curve.
        for instr in instruments:
            if isinstance(instr, Forward):
                self._forward_zero(instr, implied, fixed_pillars, free_dates, pinned_free_date)

        # 3. Float-float (basis) swaps, against the other curves' known rates.
        for instr in instruments:
            if isinstance(instr, Swap) and all(
                isinstance(leg, FloatingLeg) for leg in (instr.receive_leg, instr.pay_leg)
            ):
                self._basis_zero(instr, implied, fixed_pillars, free_dates, pinned_free_date)

        # 4. Fill the rest.
        seeds: dict[CurveKey, list[float]] = {}
        for curve, dates in free_dates.items():
            built = self.market.curves.get(curve)
            points = self._zero_points(curve, implied, fixed_pillars)
            dfs = []
            for d in dates:
                if d in implied[curve]:
                    df = math.exp(-implied[curve][d] * self._tau(d))
                elif built is not None:
                    df = built.get_df(d)
                elif points:
                    taus, zs = zip(*points)
                    df = math.exp(-float(np.interp(self._tau(d), taus, zs)) * self._tau(d))
                else:
                    df = self._seed_df(flat_guess[curve], d)
                dfs.append(df)
            seeds[curve] = dfs
        return seeds

    def _fixed_leg_zero(self, leg: FixedLeg) -> float | None:
        """Zero rate (continuous, act/365) to the leg end, compounding its coupons.

        The first coupon's rate also covers any gap between the curve date and the
        leg start. None when the leg has no coupon left or the rate is unusable.
        """
        coupons = [c for c in leg.coupons if c.end_date > self.t]
        if not coupons:
            return None
        wf = 1.0
        if coupons[0].start_date > self.t:
            wf *= float(coupons[0].rate.get_wealth_factor(self.t, coupons[0].start_date))
        for c in coupons:
            wf *= float(c.rate.get_wealth_factor(max(c.start_date, self.t), c.end_date))
        tau = self._tau(coupons[-1].end_date)
        return math.log(wf) / tau if wf > 0 and math.isfinite(wf) and tau > 0 else None

    def _forward_zero(self, fwd: Forward, implied, fixed_pillars, free_dates, pinned_free_date) -> None:
        """Seed one funding curve of `fwd` by covered interest parity.

        With ``L`` the date the pair's FX is known until and ``T`` the pinned date,
        ``strike / FX(L) = [df_b(T)/df_b(L)] / [df_q(T)/df_q(L)]``. The unknown
        side is taken flat over ``[L, T]`` and that zero rate is extended back to
        the curve date. Skipped when neither or both sides are known.
        """
        pair = fwd.currency_pair
        keys = (
            funding_key(self.riskless_index, pair.base_currency),
            funding_key(self.riskless_index, pair.quote_currency),
        )
        known_until = self.market.get_fx_known_until(pair)
        try:
            fx_known = self.market.get_fx_rate(known_until, pair).value
        except KeyError:
            return
        for unknown, known, sign in ((keys[0], keys[1], 1.0), (keys[1], keys[0], -1.0)):
            if unknown not in free_dates or not (m := pinned_free_date(unknown, fwd)):
                continue
            span = self._tau(m) - self._tau(known_until)
            log_known = self._log_df_change(known, known_until, m, implied, fixed_pillars, free_dates)
            if span <= 0 or log_known is None or fwd.strike <= 0:
                continue
            log_ratio = math.log(fwd.strike / fx_known)
            # base unknown: -z*span = log_ratio + log_known; quote unknown: -z*span = log_known - log_ratio.
            implied[unknown][m] = -(sign * log_ratio + log_known) / span
            return

    def _basis_zero(self, swap: Swap, implied, fixed_pillars, free_dates, pinned_free_date) -> None:
        """Seed one curve of a float-float swap from its spreads.

        At the pinned date, ``z_I_r + s_r - z_D_r = z_I_p + s_p - z_D_p`` (index
        ``I`` and funding discount curve ``D`` of each leg), solved for the single
        free curve in it. Collateral is ignored: it only shifts the seed. Skipped
        when there is not exactly one unknown or another curve is not known.
        """
        coef: dict[CurveKey, float] = defaultdict(float)  # coefficient of each zero rate
        spreads = 0.0
        for leg, side in ((swap.receive_leg, 1.0), (swap.pay_leg, -1.0)):
            coef[leg.index] += side
            coef[funding_key(self.riskless_index, swap.leg_settlement_currency(leg))] -= side
            spreads += side * getattr(leg.spread, "rate", 0.0)
        unknowns = [
            (curve, m)
            for curve, k in coef.items()
            if k != 0 and curve in free_dates and (m := pinned_free_date(curve, swap))
        ]
        if len(unknowns) != 1:
            return
        unknown, m = unknowns[0]
        tau = self._tau(m)
        total = spreads
        for curve, k in coef.items():
            if curve == unknown or k == 0:
                continue
            log_df = self._log_df_change(curve, self.t, m, implied, fixed_pillars, free_dates)
            if log_df is None:
                return
            total += k * (-log_df / tau)
        implied[unknown][m] = -total / coef[unknown]

    def _log_df_change(self, curve, start, end, implied, fixed_pillars, free_dates) -> float | None:
        """``log(df(end) / df(start))`` on a curve already known, else None."""
        if curve in self.market.curves and curve not in free_dates:
            built = self.market.curves[curve]
            return math.log(built.get_df(end)) - math.log(built.get_df(start))
        if curve in free_dates and implied[curve]:  # being solved: use its step-1 seeds.
            taus, zs = zip(*self._zero_points(curve, implied, fixed_pillars))
            def log_df(d: date) -> float:
                return -float(np.interp(self._tau(d), taus, zs)) * self._tau(d)
            return log_df(end) - log_df(start)
        if curve in self.market.curves:  # extended curve with nothing implied: extrapolate.
            built = self.market.curves[curve]
            return math.log(built.get_df(end)) - math.log(built.get_df(start))
        return None

    def _zero_points(self, curve, implied, fixed_pillars) -> list[tuple[float, float]]:
        """Sorted ``(tau, zero rate)`` points known on `curve`: fixed pillars and seeds."""
        points = {
            self._tau(d): -math.log(df) / self._tau(d)
            for d, df in fixed_pillars.get(curve, ())
            if d > self.t
        }
        points.update({self._tau(d): z for d, z in implied[curve].items()})
        return sorted(points.items())

    def _short_end_anchor(self, curve: CurveKey) -> tuple[tuple[date, float] | None, float]:
        """Fixed short-end pillar and rate guess for a new curve.

        An overnight index curve is anchored at its next business day with the
        rate from :meth:`OvernightHistory.spot_overnight_rate` (today's fixing, or
        the latest realised one-day rate of a price index such as ICP).

        Parameters
        ----------
        curve : Index or Currency
            New curve.

        Returns
        -------
        anchor : tuple of (date, float) or None
            Fixed pillar; None when the curve has no overnight history.
        guess : float
            Flat rate used to seed the free pillars (4% without an anchor).
        """
        if isinstance(curve, Index):
            curve_index = curve
            try:
                history = self.market.get_index(curve_index.name)
                if isinstance(history, OvernightHistory):
                    rate = history.spot_overnight_rate(self.t).copy()
                    anchor_date = curve_index.get_maturity(self.t)  # overnight: next business day.
                    rate.convert_rate_convention(RateConvention(), self.t, anchor_date)
                    return (anchor_date, self._seed_df(rate.rate, anchor_date)), rate.rate
            except KeyError:
                pass
        return None, 0.04

    def _seed_df(self, rate: float, maturity: date) -> float:
        """Annually compounded act/365 discount factor at `rate` to `maturity`."""
        return (1 + rate) ** (-(maturity - self.t).days / _DAY_BASE)

    def _residual_instruments(self, instruments, free_dates):
        """Instruments that pin a free pillar; the others are redundant in the solve.

        Parameters
        ----------
        instruments : list
            The group's instruments.
        free_dates : dict
            Curve key to free pillar dates.

        Returns
        -------
        list
        """
        free_curve_dates = {c: set(d) for c, d in free_dates.items()}

        def pins_free_pillar(instr) -> bool:
            return any(
                (m := _get_maturity(curve, instr, self.riskless_index)) is not None and m in dates
                for curve, dates in free_curve_dates.items()
            )

        return [instr for instr in instruments if pins_free_pillar(instr)]

    # -- projection curves ----------------------------------------------------

    def _projection_anchor(self, index: Index) -> tuple[list[date], list[float]]:
        """Fixed spot segment ``[t + spot_lag, maturity]`` of a new projection curve.

        Its forward comes from the index fixing of the curve date.

        Raises
        ------
        InsufficientQuotesError
            If the index has no rate history.
        """
        anchor_start = index.calendar.add_business_days(self.t, getattr(index, "spot_lag", 0))
        anchor_end = index.get_maturity(anchor_start)
        try:
            history = self.market.get_index(index.name)
        except KeyError:
            history = None
        if not hasattr(history, "get_rate"):
            raise InsufficientQuotesError(
                f"Projection index '{index.name}' has no rate history to anchor its short end."
            )
        wf = history.get_rate(self.t).get_wealth_factor(anchor_start, anchor_end)  # type: ignore[union-attr]
        return [anchor_start, anchor_end], [math.log(wf) / ((anchor_end - anchor_start).days / _DAY_BASE)]

    def _bootstrap_projection_curve(self, index: Index, swaps: list[Swap]) -> None:
        """Solve the projection curve of `index` alone from `swaps`.

        The discount curves the swaps read must already be in the market.

        Raises
        ------
        InsufficientQuotesError
            If the swaps do not pin one distinct new segment each.
        """
        if self._solve_group(frozenset({_Proj(index)}), swaps, index.currency) != "solved":
            raise InsufficientQuotesError(
                f"Projection curve for '{index.name}' is under-determined by its swaps."
            )


def build_curves(
    quotes: list,
    market: Market,
    *,
    interpolation_method: InterpolationMethod | None = None,
) -> None:
    """Bootstrap the curves that reprice `quotes` into `market`, in place.

    Discount curves go to ``market.curves`` (keyed by Index, or by a non-riskless
    Currency) and projection-only indexes get a curve in
    ``market.projection_curves``.

    Parameters
    ----------
    quotes : list of InstrumentQuote
        Quotes sharing one quote date, equal to ``market.t``.
    market : Market
        Market as of the quote date, with the FX and fixings valuation needs.
    interpolation_method : InterpolationMethod, optional
        Discount-curve interpolation for this build; defaults to
        ``market.discount_interpolation_method``.

    Raises
    ------
    ValueError
        If no quote has a quote date, the dates differ, or they differ from
        ``market.t``.
    InsufficientQuotesError
        If the quotes under-determine a curve.
    """
    # Derive the valuation date from the quotes and cross-check market.t, since
    # Calculator.valuate reads FX (and other) market data off market.t internally.
    quote_dates = {q.quote_date for q in quotes if getattr(q, "quote_date", None) is not None}
    if not quote_dates:
        raise ValueError("No quotes with a quote_date were provided; cannot derive a valuation date.")
    if len(quote_dates) > 1:
        raise ValueError(f"All quotes must share the same quote_date; got {sorted(quote_dates)}.")
    t = next(iter(quote_dates))
    if t != market.t:
        raise ValueError(
            f"Quotes' quote_date ({t}) does not match market.t ({market.t}); market must be "
            "valued as of the same date as the quotes (Calculator.valuate reads FX and other "
            "data off market.t)."
        )

    _CurveBuilder(
        quotes,
        market,
        discount_method=interpolation_method or market.discount_interpolation_method,
        projection_method=market.projection_interpolation_method,
    ).build()
