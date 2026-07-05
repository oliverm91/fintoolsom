from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from scipy.optimize import root

from fintoolsom.rates.Rates import RateConvention  # type: ignore[import-untyped]

from ..market.index import Index
from ..market.currencies import Currency, CurrencyName, CurrencyPair
from ..derivatives.calculator import Calculator
from ..market.market import Market
from ..market.index_history import OvernightRateHistory
from ..market.quotes import InstrumentQuote, CrossCurrencyFloatFloatQuote, ForwardPointsQuote
from ..derivatives.swaps import Swap, FloatingLeg
from ..derivatives.forwards import Forward, NDF
from ..rates import ZeroCouponCurve


CurveKey = tuple[Index, Currency]


class InsufficientQuotesError(Exception):
    """Raised when a curve group is under-determined: it has more free pillars to
    solve than instruments (residuals) available to pin them down."""


def _curves_needed(quote: InstrumentQuote, riskless_index: Index) -> frozenset[CurveKey]:
    keys: set[CurveKey] = set()

    instrument = quote.get_instrument()
    # Only _SwapQuote carries collateral_index; other quote types return None.
    collateral: Index | None = getattr(quote, "collateral_index", None)
    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            # Discount curve for this leg. With collateral C the effective discount
            # factor is  base(R, leg.ccy) * proj(C, C.ccy) / base(R, C.ccy)  (see
            # Calculator._leg_pv). When leg.currency == collateral.currency, the
            # base(R, leg.ccy) and base(R, C.ccy) factors are the SAME curve and
            # cancel exactly, leaving only the collateral projection curve — so the
            # riskless curve of that currency is NOT needed.
            if collateral is None:
                keys.add((riskless_index, leg.currency))
            else:
                if leg.currency != collateral.currency:
                    keys.add((riskless_index, leg.currency))
                    keys.add((riskless_index, collateral.currency))
                keys.add((collateral, collateral.currency))
            if isinstance(leg, FloatingLeg):
                keys.add((leg.index, leg.index.currency))
    elif isinstance(instrument, NDF) and instrument.is_uf_indexed:
        # UF curve structure (CLP riskless + UF index) needs a UF Index that quotes
        # don't carry; not covered by this pass (mirrors historical behavior).
        return frozenset()
    else:  # Forward / FX NDF
        curr_pair: CurrencyPair = instrument.currency_pair
        for curr in (curr_pair.base_currency, curr_pair.quote_currency):
            keys.add((riskless_index, curr))

    return frozenset(keys)


def _curves_touched(quote: InstrumentQuote, riskless_index: Index) -> frozenset[CurveKey]:
    """Every curve that must EXIST to value `quote` — a superset of _curves_needed
    that also includes curves whose value cancels out of the discounting (the
    base / riskless-collateral pair in Calculator._leg_pv) but is still looked up.
    Used only to order the build: a group is solved once all the curves its
    instruments merely read (touch but do not pin) already exist."""
    keys: set[CurveKey] = set()

    instrument = quote.get_instrument()
    collateral: Index | None = getattr(quote, "collateral_index", None)
    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            keys.add((riskless_index, leg.currency))
            if collateral is not None:
                keys.add((riskless_index, collateral.currency))
                keys.add((collateral, collateral.currency))
            if isinstance(leg, FloatingLeg):
                keys.add((leg.index, leg.index.currency))
    elif isinstance(instrument, NDF) and instrument.is_uf_indexed:
        return frozenset()
    else:  # Forward / FX NDF
        curr_pair: CurrencyPair = instrument.currency_pair
        for curr in (curr_pair.base_currency, curr_pair.quote_currency):
            keys.add((riskless_index, curr))

    return frozenset(keys)


def _get_maturity(curve_key: CurveKey, instrument: Swap | Forward, riskless_index: Index) -> date | None:
    maturity_candidates: set[date] = set()
    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            if leg.currency == curve_key[1] and curve_key[0] == riskless_index:
                maturity_candidates.add(max(leg.payment_dates))
            if isinstance(leg, FloatingLeg):
                if leg.index == curve_key[0] and leg.index.currency == curve_key[1]:
                    maturity_candidates.add(max(leg.end_dates))
    elif isinstance(instrument, Forward):
        if instrument.currency_pair is None:
            # UF-indexed NDF: excluded from this pass (see _curves_needed).
            return None
        currencies = (instrument.currency_pair.base_currency, instrument.currency_pair.quote_currency)
        for currency in currencies:
            if currency == curve_key[1] and curve_key[0] == riskless_index:
                if isinstance(instrument, NDF):
                    maturity_candidates.add(instrument.fixing_date)
                else:
                    maturity_candidates.add(instrument.payment_date)
    else:
        raise TypeError(
            f"_get_maturity only supports Swap and Forward instruments, got {type(instrument).__name__}."
        )
    if maturity_candidates:
        return max(maturity_candidates)
    # This should only happen when an instrument not affected by curve_key was passed.
    return None

def build_curves(
    quotes: list,
    riskless_index: Index,
    market: Market,
) -> None:
    """Bootstrap ZeroCouponCurve objects from `quotes` and store them in
    `market.curves`, keyed by (Index, Currency). Mutates `market` in place;
    does not return anything. `market` must already be valued as of the
    quotes' quote_date and carry whatever FX/other data Calculator.valuate
    needs at that date (e.g. spot rates for FX forwards)."""

    # 1. Derive valuation date from quotes and cross-check against market.t, since
    # Calculator.valuate reads FX (and other) market data off market.t internally.
    all_dates = {q.quote_date for q in quotes if getattr(q, "quote_date", None) is not None}
    if not all_dates:
        raise ValueError("No quotes with a quote_date were provided; cannot derive a valuation date.")
    if len(all_dates) > 1:
        raise ValueError(f"All quotes must share the same quote_date; got {sorted(all_dates)}.")
    t: date = next(iter(all_dates))
    if t != market.t:
        raise ValueError(
            f"Quotes' quote_date ({t}) does not match market.t ({market.t}); market must be "
            "valued as of the same date as the quotes (Calculator.valuate reads FX and other "
            "data off market.t)."
        )

    # 2. Curve dependency sets per quote:
    #    * pinned  (_curves_needed): the curves an instrument actually determines. Used
    #      to group instruments and to choose which pillars a group solves.
    #    * touched (_curves_touched): every curve that must merely EXIST to value it,
    #      including ones that cancel out of the discounting. Used to order the build.
    pinned: dict[InstrumentQuote, frozenset[CurveKey]] = {q: _curves_needed(q, riskless_index) for q in quotes}
    touched: dict[InstrumentQuote, frozenset[CurveKey]] = {q: _curves_touched(q, riskless_index) for q in quotes}

    groups: defaultdict[frozenset[CurveKey], list[InstrumentQuote]] = defaultdict(list)
    for q in quotes:
        groups[pinned[q]].append(q)

    def _attempt(curve_group: frozenset[CurveKey]) -> str:
        """Try to solve `curve_group` against the market's current curves. Returns
        'solved', 'nothing' (all needed pillars already exist) or 'defer' (not yet
        ready: a curve it reads is still unbuilt, or it is not square yet)."""
        # Some quotes need the market FX to fully materialise for valuation: XCCY (to size
        # matched leg notionals) and forward-points (to build the outright strike off the
        # market spot). Other quote types ignore it, and structure-only get_instrument()
        # calls elsewhere stay FX-free (currencies/dates don't depend on it).
        _fx_dependent = (CrossCurrencyFloatFloatQuote, ForwardPointsQuote)
        group_instruments = [
            q.get_instrument(market) if isinstance(q, _fx_dependent) else q.get_instrument()
            for q in groups[curve_group]
        ]

        # Valuable? Every curve these instruments touch but do NOT pin (e.g. a riskless
        # curve that cancels in the collateral discounting) must already exist, or
        # valuation would hit a missing curve. If not, defer to a later pass.
        group_touched: set[CurveKey] = set().union(*(touched[q] for q in groups[curve_group]))
        if not (group_touched - curve_group) <= set(market.curves):
            return "defer"

        # Split each pinned curve's maturities into FIXED pillars (already solved in an
        # earlier group, or a short-end anchor — held constant) and FREE maturities (new
        # points this group must solve). Only free points become unknowns, so the solver
        # focuses solely on the pillars these instruments pin.
        fixed_pillars: dict[CurveKey, list[tuple[date, float]]] = {}
        free_dates: dict[CurveKey, list[date]] = {}
        free_seed_dfs: dict[CurveKey, list[float]] = {}
        for curve in curve_group:
            # Each instrument contributes the maturity it pins on `curve` (or None).
            curve_maturities = {_get_maturity(curve, instr, riskless_index) for instr in group_instruments}
            curve_maturities.discard(None)
            if not curve_maturities:
                continue

            if curve not in market.curves:
                # New curve: every needed maturity is free. Seed from the overnight
                # index rate when the curve is that index's own-currency curve (else a
                # flat 4% guess), and anchor the short end with the overnight point as a
                # FIXED pillar (seeded, never solved).
                guess = 0.04
                anchor: tuple[date, float] | None = None
                curve_index, curve_currency = curve
                if curve_currency == curve_index.currency:
                    try:
                        index_history = market.get_index(curve_index.name)
                        if isinstance(index_history, OvernightRateHistory):
                            r = index_history.rates[t].copy()
                            # Overnight anchor date = the index's next fixing (its accrual
                            # period from t: one business day for an overnight index).
                            anchor_date = curve_index.get_maturity(t)
                            r.convert_rate_convention(RateConvention(), t, anchor_date)
                            guess = r.rate_value
                            anchor = (anchor_date, (1 + guess) ** (-(anchor_date - t).days / 365))
                    except KeyError:
                        pass
                new_dts = sorted(curve_maturities)
                fixed_pillars[curve] = [anchor] if anchor is not None else []
                free_dates[curve] = new_dts
                free_seed_dfs[curve] = [(1 + guess) ** (-(m - t).days / 365) for m in new_dts]
            else:
                # Existing curve: keep every solved pillar fixed and only solve
                # maturities that extend it past its current last pillar. Maturities
                # inside the existing range are read off by interpolation (no unknown).
                built_curve = market.curves[curve]
                existing: list[tuple[date, float]] = list(built_curve.date_dfs)
                cur_max = max(d for d, _ in existing)
                new_dts = sorted(m for m in curve_maturities if m > cur_max)
                fixed_pillars[curve] = existing
                if new_dts:
                    free_dates[curve] = new_dts
                    free_seed_dfs[curve] = [built_curve.get_df(m) for m in new_dts]

        if not free_dates:
            # All pillars this group needs already exist: nothing left to solve.
            return "nothing"

        # Residual instruments are only those that actually pin a new (free) pillar.
        # Instruments touching solely already-fixed pillars (e.g. short XCCY overlapping
        # an already-built short end) are redundant here and left out of the solve.
        free_curve_dates: dict[CurveKey, set[date]] = {c: set(d) for c, d in free_dates.items()}

        def _introduces_free_pillar(instr) -> bool:
            for curve, dates_set in free_curve_dates.items():
                m = _get_maturity(curve, instr, riskless_index)
                if m is not None and m in dates_set:
                    return True
            return False

        residual_instruments = [instr for instr in group_instruments if _introduces_free_pillar(instr)]

        # Flatten free pillars into one vector, optimising log(df) rather than df to
        # remove the strong collinearity between near-1 short dfs and small long dfs.
        curve_slice: dict[CurveKey, tuple[int, int]] = {}
        y0: list[float] = []
        start = 0
        for curve, dts in free_dates.items():
            end = start + len(dts)
            curve_slice[curve] = (start, end)
            y0 += [math.log(df) for df in free_seed_dfs[curve]]
            start = end

        if len(y0) != len(residual_instruments):
            # Not square yet — a curve this group reads as fixed may still be unbuilt,
            # so it is (usually transiently) under-determined. Defer to a later pass; if
            # nothing ever makes it square the scheduler raises InsufficientQuotesError.
            return "defer"

        # The solve is atomic: f() overwrites market.curves on every iteration, so snapshot
        # the curves this group touches and roll back on failure — a group that cannot be
        # solved must not leave partially-extended (garbage) pillars on the earlier,
        # already-clean curves it extends.
        snapshot = {c: market.curves.get(c) for c in curve_slice}

        def _restore() -> None:
            for c, prev in snapshot.items():
                if prev is None:
                    market.curves.pop(c, None)
                else:
                    market.curves[c] = prev

        def _apply(y) -> None:
            for curve, (s, e) in curve_slice.items():
                free = list(zip(free_dates[curve], (math.exp(v) for v in y[s:e])))
                date_dfs = fixed_pillars[curve] + free
                date_dfs.sort(key=lambda p: p[0])
                market.curves[curve] = ZeroCouponCurve(t, date_dfs=date_dfs)

        clp_currency = Currency(CurrencyName.CLP)
        def f(y) -> list[float]:
            _apply(y)
            return [
                Calculator.valuate(instr, market, riskless_index, clp_currency)
                for instr in residual_instruments
            ]

        try:
            result = root(f, y0, method="hybr")
        except ValueError as exc:
            # scipy raises on non-finite residuals. This happens when the group's
            # instruments are nearly insensitive to the pillars being solved, so the
            # solver wanders into a region where valuation blows up (e.g. cross-currency
            # basis swaps priced without their notional exchanges — the joint long-end
            # solve of the collateral discount curves is ill-posed).
            _restore()
            raise ValueError(
                f"Curve solve for group {set(free_dates)} did not converge: valuation "
                f"produced non-finite residuals, so this group is ill-posed for the "
                f"pillars it is trying to solve ({exc})."
            ) from exc
        if not result.success:
            _restore()
            raise ValueError(
                f"Curve solve failed to converge for curve group {set(free_dates)}: {result.message}"
            )
        # Store the converged curves (root's last f-eval is not guaranteed at result.x).
        _apply(result.x)
        return "solved"

    # 3. Fixpoint scheduler: each pass solves every group that is currently valuable and
    #    square (smaller groups first), building the fundamental curves other groups
    #    depend on. A pass that makes no progress means the rest are under-determined.
    pending: list[frozenset[CurveKey]] = [g for g in groups if g]
    while pending:
        solved_any = False
        for curve_group in sorted(pending, key=len):
            if _attempt(curve_group) in ("solved", "nothing"):
                pending.remove(curve_group)
                solved_any = True
        if not solved_any:
            raise InsufficientQuotesError(
                "Curve build stalled: the remaining groups are under-determined given "
                f"the available quotes: {[set(g) for g in pending]}."
            )
