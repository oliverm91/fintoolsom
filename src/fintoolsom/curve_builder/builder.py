from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from scipy.optimize import root

from fintoolsom.rates.Rates import RateConvention  # type: ignore[import-untyped]

from ..market.index import Index
from ..market.currencies import Currency, CurrencyName
from ..derivatives.calculator import Calculator
from ..market.market import CurveKey, Market, funding_key
from ..market.index_history import OvernightHistory
from ..market.quotes import InstrumentQuote, CrossCurrencyFloatFloatQuote, ForwardPointsQuote
from ..derivatives.swaps import Swap, FloatingLeg
from ..derivatives.forwards import Forward, NDF
from ..rates import (
    Curve,
    ZeroCouponCurve,
    InterpolationMethod,
    ProjectionCurve,
    ProjectionInterpolationMethod,
)


# Curve-solve residuals are valued in one reporting currency; a par/MTM==0 residual
# is zero in any currency, so the choice is arbitrary — CLP by convention.
_REPORT_CURRENCY = Currency(CurrencyName.CLP)
_DAY_BASE = 365
# Quote types whose instrument needs the market FX to fully materialise: XCCY (to size
# matched leg notionals) and forward-points (to build the outright strike off the spot).
_FX_DEPENDENT = (CrossCurrencyFloatFloatQuote, ForwardPointsQuote)


class InsufficientQuotesError(Exception):
    """A curve group is under-determined: more free pillars than instruments."""


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


def _projection_indexes(
    quote: InstrumentQuote, riskless_index: Index, discount_indexes: frozenset[Index] = frozenset()
) -> set[Index]:
    """Floating-leg indexes of `quote` that are projection-only.

    They get an independent ProjectionCurve, never an entry in ``market.curves``.

    Parameters
    ----------
    quote : InstrumentQuote
        Quote considered.
    riskless_index : Index
        Market funding index.
    discount_indexes : frozenset of Index, optional
        Other discount indexes, see :func:`_discount_indexes`.

    Returns
    -------
    set of Index
    """
    instrument = quote.get_instrument()
    if not isinstance(instrument, Swap):
        return set()
    discounting = _discount_indexes(quote, riskless_index, discount_indexes)
    return {
        leg.index
        for leg in (instrument.receive_leg, instrument.pay_leg)
        if isinstance(leg, FloatingLeg) and leg.index not in discounting
    }


def _curve_keys(
    quote: InstrumentQuote,
    riskless_index: Index,
    *,
    cancel_collateral: bool,
    discount_indexes: frozenset[Index] = frozenset(),
) -> frozenset[CurveKey]:
    """Discount-curve keys `quote` relates to.

    Each leg discounts in its settlement currency (the swap payment currency for a
    non-deliverable leg in another currency, else its own). Under collateral C the
    discount factor is ``funding(ccy) * curve(C) / funding(C.ccy)``; when
    ``ccy == C.ccy`` the funding factors cancel. A non-deliverable leg also reads
    the funding curves of both currencies to project its FX fixings. A floating
    index is a key only when its curve discounts (projection-only indexes are
    solved by :meth:`_CurveBuilder._bootstrap_projection_curve`).

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
            if isinstance(leg, FloatingLeg) and leg.index in discounting:
                keys.add(leg.index)
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
            if isinstance(leg, FloatingLeg) and leg.index == curve_key:
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
    """Free-pillar layout of one discount-curve group solve.

    Holds the free dates of each curve, their slice of the ``log(df)`` unknown
    vector, the fixed pillars, and an atomic apply / rollback over
    ``market.curves``. Solving in ``log(df)`` removes the collinearity between
    near-1 short dfs and small long dfs.

    Parameters
    ----------
    market : Market
        Market whose curves are solved.
    t : date
        Curve date.
    method : InterpolationMethod
        Interpolation of the rebuilt curves.
    fixed_pillars : dict
        Curve key to fixed ``(date, df)`` pillars.
    free_dates : dict
        Curve key to free pillar dates.
    curve_slice : dict
        Curve key to its ``(start, end)`` slice of the unknown vector.
    y0 : list of float
        Initial ``log(df)`` guesses.
    """

    market: Market
    t: date
    method: InterpolationMethod
    fixed_pillars: dict[CurveKey, list[tuple[date, float]]]
    free_dates: dict[CurveKey, list[date]]
    curve_slice: dict[CurveKey, tuple[int, int]]
    y0: list[float]
    _snapshot: dict[CurveKey, Curve | None] = field(init=False, default=None)

    @classmethod
    def from_free_pillars(
        cls,
        market: Market,
        t: date,
        method: InterpolationMethod,
        fixed_pillars: dict[CurveKey, list[tuple[date, float]]],
        free_dates: dict[CurveKey, list[date]],
        free_seed_dfs: dict[CurveKey, list[float]],
    ) -> _GroupSolve:
        """Lay out the unknown vector from the free dates and their seed dfs.

        Parameters
        ----------
        market, t, method, fixed_pillars, free_dates
            As in the class.
        free_seed_dfs : dict
            Curve key to initial dfs of its free dates.

        Returns
        -------
        _GroupSolve
        """
        curve_slice: dict[CurveKey, tuple[int, int]] = {}
        y0: list[float] = []
        start = 0
        for curve, dates in free_dates.items():
            curve_slice[curve] = (start, start + len(dates))
            y0 += [math.log(df) for df in free_seed_dfs[curve]]
            start += len(dates)
        return cls(market, t, method, fixed_pillars, free_dates, curve_slice, y0)

    def __post_init__(self):
        self._snapshot = {c: self.market.curves.get(c) for c in self.curve_slice}

    @property
    def n_unknowns(self) -> int:
        """Number of free pillars."""
        return len(self.y0)

    def apply(self, y) -> None:
        """Rebuild every group curve from fixed pillars plus ``exp(y)`` free dfs."""
        for curve, (s, e) in self.curve_slice.items():
            free = zip(self.free_dates[curve], (math.exp(v) for v in y[s:e]))
            date_dfs = sorted([*self.fixed_pillars[curve], *free], key=lambda p: p[0])
            self.market.curves[curve] = ZeroCouponCurve(
                self.t, date_dfs=date_dfs, df_interpolation_method=self.method
            )

    def restore(self) -> None:
        """Put back the curves as they were before the solve."""
        for curve, previous in self._snapshot.items():
            _set_or_pop(self.market.curves, curve, previous)


# ── Orchestrator ────────────────────────────────────────────────────────────

class _CurveBuilder:
    """One curve build: discount curves first, then projection curves.

    Discount curves are solved into ``market.curves`` by a fixpoint scheduler, then
    a ProjectionCurve is bootstrapped for each projection-only index. Every index
    ends with exactly one curve. Mutates `market` in place.

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
        # Quotes on a projection-only index are solved in the projection phase, after
        # every discount curve exists; all others form the discount-curve groups.
        self.projection_quotes = {
            q: _projection_indexes(q, riskless_index, self.discount_indexes) for q in quotes
        }
        discount_quotes = [q for q in quotes if not self.projection_quotes[q]]

        # pinned  (_curves_needed): the curves each quote determines — groups instruments.
        # touched (_curves_touched): every curve that must exist to value it — orders the build.
        self.pinned = {q: _curves_needed(q, riskless_index, self.discount_indexes) for q in discount_quotes}
        self.touched = {q: _curves_touched(q, riskless_index, self.discount_indexes) for q in discount_quotes}
        self.groups: dict[frozenset[CurveKey], list[InstrumentQuote]] = defaultdict(list)
        for q in discount_quotes:
            self.groups[self.pinned[q]].append(q)

    def build(self) -> None:
        """Solve the discount curves, then the projection curves."""
        self._solve_discount_curves()
        self._register_projection_curves()

    # -- discount curves ------------------------------------------------------

    def _solve_discount_curves(self) -> None:
        """Solve every discount-curve group with a fixpoint scheduler.

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

    def _attempt(self, group: frozenset[CurveKey]) -> str:
        """Try to solve `group` against the current market curves.

        Parameters
        ----------
        group : frozenset of Index or Currency
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
        if not (group_touched - group) <= set(self.market.curves):
            return "defer"

        fixed_pillars, free_dates, free_seed_dfs = self._partition_pillars(group, instruments)
        if not free_dates:
            return "nothing"  # every pillar this group needs already exists.

        residual_instruments = self._residual_instruments(instruments, free_dates)
        solve = _GroupSolve.from_free_pillars(
            self.market, self.t, self.discount_method, fixed_pillars, free_dates, free_seed_dfs
        )
        if solve.n_unknowns != len(residual_instruments):
            # Not square yet (a curve it reads as fixed may still be unbuilt) — defer;
            # if nothing ever makes it square the scheduler raises InsufficientQuotesError.
            return "defer"

        def residual(y) -> list[float]:
            solve.apply(y)
            return [
                Calculator.valuate(instr, self.market, _REPORT_CURRENCY)
                for instr in residual_instruments
            ]

        result = _solve_or_rollback(
            residual, solve.y0, solve.restore, f"Curve solve for group {set(free_dates)}"
        )
        solve.apply(result.x)  # root's last f-eval is not guaranteed to be at result.x.
        return "solved"

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
        free_seed_dfs: dict[CurveKey, list[float]] = {}
        for curve in group:
            maturities = {_get_maturity(curve, instr, self.riskless_index) for instr in instruments}
            maturities.discard(None)
            if not maturities:
                continue

            if curve not in self.market.curves:
                # New curve: every maturity is free; seed from the overnight anchor.
                anchor, guess = self._short_end_anchor(curve)
                new_dates = sorted(maturities)
                fixed_pillars[curve] = [anchor] if anchor is not None else []
                free_dates[curve] = new_dates
                free_seed_dfs[curve] = [self._seed_df(guess, m) for m in new_dates]
            elif isinstance(built := self.market.curves[curve], ZeroCouponCurve):
                # Existing curve: keep solved pillars, only extend past its last one;
                # maturities inside the range are read off by interpolation (no unknown).
                existing = list(zip(built.dates, built.dfs))
                last = max(d for d, _ in existing)
                new_dates = sorted(m for m in maturities if m > last)
                fixed_pillars[curve] = existing
                if new_dates:
                    free_dates[curve] = new_dates
                    free_seed_dfs[curve] = [built.get_df(m) for m in new_dates]
            # else: a non-pillar curve (e.g. NSSCurve) is defined at every date and has
            # no pillars to extend — it is read as fixed, never solved.
        return fixed_pillars, free_dates, free_seed_dfs

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
                    return (anchor_date, self._seed_df(rate.rate_value, anchor_date)), rate.rate_value
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

    def _register_projection_curves(self) -> None:
        """Bootstrap a ProjectionCurve for each projection-only index.

        A discount index needs none: ``Market.get_projection`` views its discount
        curve as forwards.
        """
        swaps_by_index: dict[Index, list[Swap]] = defaultdict(list)
        for q, indexes in self.projection_quotes.items():
            instrument = _instrument_of(q, self.market)
            for index in indexes:
                swaps_by_index[index].append(instrument)

        for index, swaps in swaps_by_index.items():
            self._bootstrap_projection_curve(index, swaps)

    def _bootstrap_projection_curve(self, index: Index, swaps: list[Swap]) -> None:
        """Bootstrap the independent forward curve of `index` so each swap reprices.

        Discount curves already exist, so the unknowns are the forward rates of
        `index`: one knot per swap maturity after a fixed spot anchor taken from the
        index history.

        Parameters
        ----------
        index : Index
            Projection-only index.
        swaps : list of Swap
            Swaps on `index`, one per distinct maturity.

        Raises
        ------
        InsufficientQuotesError
            If the index has no rate history, or the maturities are not distinct and
            beyond the spot anchor.
        """
        market, t = self.market, self.t
        anchor_start = index.calendar.add_business_days(t, getattr(index, "spot_lag", 0))
        anchor_end = index.get_maturity(anchor_start)

        history = market.get_index(index.name)
        if not hasattr(history, "get_rate"):
            raise InsufficientQuotesError(
                f"Projection index '{index.name}' has no rate history to anchor its short end."
            )
        wf_anchor = history.get_rate(t).get_wealth_factor(anchor_start, anchor_end)  # type: ignore[attr-defined]
        f_anchor = math.log(wf_anchor) / ((anchor_end - anchor_start).days / _DAY_BASE)

        def leg_maturity(swap: Swap) -> date:
            return max(
                max(leg.end_dates)
                for leg in (swap.receive_leg, swap.pay_leg)
                if isinstance(leg, FloatingLeg) and leg.index == index
            )

        dated = sorted(((leg_maturity(s), s) for s in swaps), key=lambda ms: ms[0])
        maturities = [m for m, _ in dated]
        ordered_swaps = [s for _, s in dated]
        # One distinct maturity beyond the spot anchor per instrument → square.
        if len(set(maturities)) != len(maturities) or any(m <= anchor_end for m in maturities):
            raise InsufficientQuotesError(
                f"Projection curve for '{index.name}' is under-determined: it needs one "
                f"distinct maturity beyond the spot end {anchor_end} per instrument; got "
                f"maturities {maturities}."
            )

        knots = [anchor_start, anchor_end, *maturities]
        previous = market.projection_curves.get(index)

        def apply(free_forwards) -> None:
            forwards = [f_anchor, *(float(v) for v in free_forwards)]
            market.set_projection(index, ProjectionCurve(t, knots, forwards, self.projection_method, _DAY_BASE))

        def residual(free_forwards) -> list[float]:
            apply(free_forwards)
            return [
                Calculator.get_swap_mtm(s, market, index.currency)
                for s in ordered_swaps
            ]

        result = _solve_or_rollback(
            residual,
            [f_anchor] * len(ordered_swaps),
            lambda: _set_or_pop(market.projection_curves, index, previous),
            f"Projection solve for '{index.name}'",
        )
        apply(result.x)


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
