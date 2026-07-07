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
from ..market.market import Market
from ..market.index_history import OvernightHistory
from ..market.quotes import InstrumentQuote, CrossCurrencyFloatFloatQuote, ForwardPointsQuote
from ..derivatives.swaps import Swap, FloatingLeg
from ..derivatives.forwards import Forward, NDF
from ..rates import (
    ZeroCouponCurve,
    InterpolationMethod,
    ProjectionCurve,
    DiscountProjectionView,
    ProjectionInterpolationMethod,
)


CurveKey = tuple[Index, Currency]

# Curve-solve residuals are valued in one reporting currency; a par/MTM==0 residual
# is zero in any currency, so the choice is arbitrary — CLP by convention.
_REPORT_CURRENCY = Currency(CurrencyName.CLP)
_DAY_BASE = 365
# Quote types whose instrument needs the market FX to fully materialise: XCCY (to size
# matched leg notionals) and forward-points (to build the outright strike off the spot).
_FX_DEPENDENT = (CrossCurrencyFloatFloatQuote, ForwardPointsQuote)


class InsufficientQuotesError(Exception):
    """Raised when a curve group is under-determined: it has more free pillars to
    solve than instruments (residuals) available to pin them down."""


def _instrument_of(quote: InstrumentQuote, market: Market):
    """The instrument a quote describes. FX-dependent quote types are passed the
    market so their notionals/strike resolve off the spot; all others ignore it."""
    return quote.get_instrument(market) if isinstance(quote, _FX_DEPENDENT) else quote.get_instrument()


def _set_or_pop(store: dict, key, previous) -> None:
    """Restore a dict entry to `previous` (None ⇒ the key was absent, so remove it)."""
    if previous is None:
        store.pop(key, None)
    else:
        store[key] = previous


# ── Curve dependency analysis ───────────────────────────────────────────────

def _curve_keys(quote: InstrumentQuote, riskless_index: Index, *, cancel_collateral: bool) -> frozenset[CurveKey]:
    """The curve keys a quote relates to.

    With ``cancel_collateral=True`` (``_curves_needed``) this is the set the quote
    actually *pins*: under collateral C the effective discount factor is
    ``base(R, leg.ccy) * proj(C, C.ccy) / base(R, C.ccy)`` (see ``Calculator._leg_pv``),
    and when ``leg.currency == C.currency`` the two ``base(R, …)`` factors are the same
    curve and cancel, so the riskless curve of that currency is not pinned.

    With ``cancel_collateral=False`` (``_curves_touched``) no cancellation is applied:
    it is the superset that must merely *exist* to value the quote, used to order the
    build so a group is solved only once every curve it reads already exists."""
    instrument = quote.get_instrument()
    collateral: Index | None = getattr(quote, "collateral_index", None)
    keys: set[CurveKey] = set()

    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            if collateral is None:
                keys.add((riskless_index, leg.currency))
            else:
                if not (cancel_collateral and leg.currency == collateral.currency):
                    keys.add((riskless_index, leg.currency))
                    keys.add((riskless_index, collateral.currency))
                keys.add((collateral, collateral.currency))
            if isinstance(leg, FloatingLeg):
                keys.add((leg.index, leg.index.currency))
    elif isinstance(instrument, NDF) and instrument.is_uf_indexed:
        # UF curve structure (CLP riskless + UF index) needs a UF Index quotes don't
        # carry; not covered by this pass (mirrors historical behavior).
        return frozenset()
    else:  # Forward / FX NDF
        pair = instrument.currency_pair
        for currency in (pair.base_currency, pair.quote_currency):
            keys.add((riskless_index, currency))

    return frozenset(keys)


def _curves_needed(quote: InstrumentQuote, riskless_index: Index) -> frozenset[CurveKey]:
    """Curves a quote pins (collateral cancellation applied). Groups instruments and
    chooses which pillars a group solves."""
    return _curve_keys(quote, riskless_index, cancel_collateral=True)


def _curves_touched(quote: InstrumentQuote, riskless_index: Index) -> frozenset[CurveKey]:
    """Every curve that must EXIST to value a quote (no cancellation) — a superset of
    ``_curves_needed`` used only to order the build."""
    return _curve_keys(quote, riskless_index, cancel_collateral=False)


def _get_maturity(curve_key: CurveKey, instrument, riskless_index: Index) -> date | None:
    """The single pillar `instrument` pins on `curve_key` (its latest relevant date),
    or None if the instrument does not touch that curve."""
    candidates: set[date] = set()
    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            if leg.currency == curve_key[1] and curve_key[0] == riskless_index:
                candidates.add(max(leg.payment_dates))
            if isinstance(leg, FloatingLeg) and (leg.index, leg.index.currency) == curve_key:
                candidates.add(max(leg.end_dates))
    elif isinstance(instrument, Forward):
        if instrument.currency_pair is None:
            return None  # UF-indexed NDF: excluded from this pass (see _curve_keys).
        pair = instrument.currency_pair
        for currency in (pair.base_currency, pair.quote_currency):
            if currency == curve_key[1] and curve_key[0] == riskless_index:
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
    """Run ``scipy.optimize.root`` (hybr) over `residual`. On non-convergence or
    non-finite residuals, roll the mutated market back via `restore` and raise — a
    failed solve must not leave partial/garbage curves behind. Returns the result."""
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
    """The free-pillar layout of one discount-curve group's solve: which dates are
    free on each curve, their flattened position in the log-df unknown vector, the
    fixed pillars held constant, and an atomic apply / rollback over ``market.curves``.

    Optimising ``log(df)`` (not df) removes the strong collinearity between near-1
    short dfs and small long dfs."""

    market: Market
    t: date
    method: InterpolationMethod
    fixed_pillars: dict[CurveKey, list[tuple[date, float]]]
    free_dates: dict[CurveKey, list[date]]
    curve_slice: dict[CurveKey, tuple[int, int]]
    y0: list[float]
    _snapshot: dict[CurveKey, ZeroCouponCurve | None] = field(init=False, default=None)

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
        return len(self.y0)

    def apply(self, y) -> None:
        for curve, (s, e) in self.curve_slice.items():
            free = zip(self.free_dates[curve], (math.exp(v) for v in y[s:e]))
            date_dfs = sorted([*self.fixed_pillars[curve], *free], key=lambda p: p[0])
            self.market.curves[curve] = ZeroCouponCurve(
                self.t, date_dfs=date_dfs, df_interpolation_method=self.method
            )

    def restore(self) -> None:
        for curve, previous in self._snapshot.items():
            _set_or_pop(self.market.curves, curve, previous)


# ── Orchestrator ────────────────────────────────────────────────────────────

class _CurveBuilder:
    """Bootstraps discount curves from `quotes` into ``market.curves`` via a fixpoint
    scheduler, then registers each floating index's projection curve. Mutates `market`
    in place. Instantiated per build; holds only that build's state."""

    def __init__(
        self,
        quotes: list,
        riskless_index: Index,
        market: Market,
        discount_method: InterpolationMethod,
        projection_method: ProjectionInterpolationMethod,
    ):
        self.quotes = quotes
        self.riskless_index = riskless_index
        self.market = market
        self.t = market.t
        self.discount_method = discount_method
        self.projection_method = projection_method

        # pinned  (_curves_needed): the curves each quote determines — groups instruments.
        # touched (_curves_touched): every curve that must exist to value it — orders the build.
        self.pinned = {q: _curves_needed(q, riskless_index) for q in quotes}
        self.touched = {q: _curves_touched(q, riskless_index) for q in quotes}
        self.groups: dict[frozenset[CurveKey], list[InstrumentQuote]] = defaultdict(list)
        for q in quotes:
            self.groups[self.pinned[q]].append(q)

    def build(self) -> None:
        self._solve_discount_curves()
        self._register_projection_curves()

    # -- discount curves ------------------------------------------------------

    def _solve_discount_curves(self) -> None:
        """Fixpoint scheduler: each pass solves every group that is currently valuable
        and square (smaller first), building the curves other groups depend on. A pass
        with no progress means the rest are under-determined."""
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
        """Try to solve `group` against the market's current curves. Returns 'solved',
        'nothing' (all its pillars already exist) or 'defer' (a curve it reads is still
        unbuilt, or it is not square yet)."""
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
                Calculator.valuate(instr, self.market, self.riskless_index, _REPORT_CURRENCY)
                for instr in residual_instruments
            ]

        result = _solve_or_rollback(
            residual, solve.y0, solve.restore, f"Curve solve for group {set(free_dates)}"
        )
        solve.apply(result.x)  # root's last f-eval is not guaranteed to be at result.x.
        return "solved"

    def _partition_pillars(self, group, instruments):
        """Split each pinned curve's maturities into FIXED pillars (already solved, or a
        short-end anchor) and FREE maturities (new unknowns this group solves). Only free
        points become unknowns, so the solver focuses on the pillars these instruments pin."""
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
            else:
                # Existing curve: keep solved pillars, only extend past its last one;
                # maturities inside the range are read off by interpolation (no unknown).
                built = self.market.curves[curve]
                existing = list(built.date_dfs)
                last = max(d for d, _ in existing)
                new_dates = sorted(m for m in maturities if m > last)
                fixed_pillars[curve] = existing
                if new_dates:
                    free_dates[curve] = new_dates
                    free_seed_dfs[curve] = [built.get_df(m) for m in new_dates]
        return fixed_pillars, free_dates, free_seed_dfs

    def _short_end_anchor(self, curve: CurveKey) -> tuple[tuple[date, float] | None, float]:
        """FIXED short-end pillar + rate guess for a brand-new curve. When the curve is
        an overnight index's own-currency curve, anchor its first fixing from the index
        history (seeded, never solved); otherwise no anchor and a flat 4% guess.

        The overnight rate comes from the history polymorphically
        (:meth:`OvernightHistory.spot_overnight_rate`): a rate index gives today's
        fixing, a price index (e.g. ICP levels) the most recent realised one-day rate."""
        curve_index, curve_currency = curve
        if curve_currency == curve_index.currency:
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
        return (1 + rate) ** (-(maturity - self.t).days / _DAY_BASE)

    def _residual_instruments(self, instruments, free_dates):
        """Only instruments that actually pin a new (free) pillar; those touching solely
        already-fixed pillars are redundant here and left out of the solve."""
        free_curve_dates = {c: set(d) for c, d in free_dates.items()}

        def pins_free_pillar(instr) -> bool:
            return any(
                (m := _get_maturity(curve, instr, self.riskless_index)) is not None and m in dates
                for curve, dates in free_curve_dates.items()
            )

        return [instr for instr in instruments if pins_free_pillar(instr)]

    # -- projection curves ----------------------------------------------------

    def _register_projection_curves(self) -> None:
        """Give every floating-leg index a projection curve. An index self-discounted in
        its own currency (the riskless index, or one used as its own collateral) aliases
        its discount curve — a forward view, one unknown not two (§7). Any other index has
        a real forwarding/discounting basis and gets an independently bootstrapped curve."""
        swaps_by_index: dict[Index, list[Swap]] = defaultdict(list)
        collateral_indices: set[Index] = set()
        for q in self.quotes:
            collateral = getattr(q, "collateral_index", None)
            if collateral is not None:
                collateral_indices.add(collateral)
            instrument = _instrument_of(q, self.market)
            if isinstance(instrument, Swap):
                for leg in (instrument.receive_leg, instrument.pay_leg):
                    if isinstance(leg, FloatingLeg) and instrument not in swaps_by_index[leg.index]:
                        swaps_by_index[leg.index].append(instrument)

        for index, swaps in swaps_by_index.items():
            aliased = index == self.riskless_index or index in collateral_indices
            discount_key = (index, index.currency)
            if aliased and discount_key in self.market.curves:
                self.market.set_projection(index, DiscountProjectionView(self.market.curves[discount_key]))
            else:
                self._bootstrap_projection_curve(index, swaps)

    def _bootstrap_projection_curve(self, index: Index, swaps: list[Swap]) -> None:
        """Bootstrap an INDEPENDENT-basis projection curve (a forwarding curve distinct
        from every discount curve — a term index, or one discounted by a different
        collateral/currency). Discount curves are already built, so the only unknowns are
        `index`'s per-segment forward rates: one knot per instrument maturity (square), a
        fixed spot anchor seeded from the index-history fixing (§5), solved so each swap
        reprices (``get_swap_mtm == 0``)."""
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
                Calculator.get_swap_mtm(s, market, self.riskless_index, index.currency)
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
    riskless_index: Index,
    market: Market,
    *,
    interpolation_method: InterpolationMethod | None = None,
) -> None:
    """Bootstrap ZeroCouponCurve objects from `quotes` and store them in
    ``market.curves`` (keyed by (Index, Currency)), then register each floating index's
    projection curve in ``market.projection_curves``. Mutates `market` in place; returns
    nothing. `market` must already be valued as of the quotes' quote_date and carry
    whatever FX/other data ``Calculator.valuate`` needs at that date (e.g. spot rates).

    `interpolation_method` overrides the discount-curve interpolation for this build;
    when omitted, ``market.discount_interpolation_method`` (LogLinear by default) is
    used — consistent with the log-df solve."""
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
        riskless_index,
        market,
        discount_method=interpolation_method or market.discount_interpolation_method,
        projection_method=market.projection_interpolation_method,
    ).build()
