from __future__ import annotations

from collections import defaultdict
from datetime import date
from scipy.optimize import least_squares

from fintoolsom.rates.Rates import RateConvention  # type: ignore[import-untyped]

from ..market.index import Index
from ..market.currencies import Currency, CurrencyName, CurrencyPair
from ..derivatives.calculator import Calculator
from ..market.market import Market
from ..market.index_history import RateHistory
from ..market.quotes import InstrumentQuote
from ..derivatives.swaps import Swap, FloatingLeg
from ..derivatives.forwards import Forward, NDF
from ..rates import ZeroCouponCurve


CurveKey = tuple[Index, Currency]


def _curves_needed(quote: InstrumentQuote, riskless_index: Index) -> frozenset[CurveKey]:
    keys: set[CurveKey] = set()

    instrument = quote.get_instrument()
    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            keys.add((riskless_index, leg.currency))
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

    # Not all quote types carry collateral_index (only _SwapQuote does).
    collateral: Index | None = getattr(quote, "collateral_index", None)
    if collateral:
        keys.add((collateral, collateral.currency))
        keys.add((riskless_index, collateral.currency))

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

    # 2. Dependency graph (all quote types)
    needs: dict[InstrumentQuote, frozenset[CurveKey]] = {q: _curves_needed(q, riskless_index) for q in quotes}

    curve_key_count: defaultdict[frozenset[CurveKey], int] = defaultdict(int)
    for needed_curves in needs.values():
        curve_key_count[needed_curves] += 1

    # Solve curve groups with fewer curves first, so simpler/more fundamental
    # curves are available as x0 seeds when solving more complex groups.
    curves_groups_count: list[tuple[frozenset[CurveKey], int]] = list(curve_key_count.items())
    curves_groups_count.sort(key=lambda x: len(x[0]))
    curves_groups = [group for group, _ in curves_groups_count]

    for curve_group in curves_groups:
        if not curve_group:
            # Quotes needing no curves here (e.g. UF-indexed NDFs; see _curves_needed).
            continue

        # Loop to solve curves in group. Each solve iteration will save a different combination of discount factors in a ZeroCouponCurve
        # object in Market and solve instrument mtms with current calculator definitons.
        # Once solved, solution will be saved in market.
        group_instruments = [q.get_instrument() for q, curves in needs.items() if curves == curve_group]

        # First, let's check which curves need solving. Some might have been solved in previous iteration.
        curves_to_build: dict[CurveKey, list[tuple[date, float]]] = {}  # curve -> pillars to solve with x0 estimation.
        for curve in curve_group:
            # Curve maturities are all maturities information that instruments that have an impact on `curve` can give
            curve_maturities = {_get_maturity(curve, instr, riskless_index) for instr in group_instruments}
            curve_maturities.discard(None)

            # TODO: Number of curve points to be solved should be equal to new maturities being added in each curve only. Not the whole curve.
            if not curve_maturities:
                continue
            x0 = None
            if curve not in market.curves:
                # If not present in market, need solving. Seed with a flat 4% guess.
                # If curve index currency and curve currency are the same, look for index history for better guess
                guess = 0.04
                curve_index, curve_currency = curve
                if curve_currency==curve_index.currency:
                    try:
                        index_history = market.get_index(curve_index.name)
                        if isinstance(index_history, RateHistory):
                            r = index_history.rates[t].copy()
                            r.convert_rate_convention(RateConvention())
                            guess = r.rate_value
                            curve_maturities.add(index_history.index.term.advance(t)) # TODO: Avoid this maturity from being solved
                    except KeyError:
                        pass

                x0 = sorted(
                    (maturity, (1 + guess) ** (-(maturity - t).days / 365)) for maturity in curve_maturities
                )
            else:
                # Curve was solved before, but this group of instruments might introduce extra information from new maturities.
                # Those will be added and use previous discount factors from existing maturities as x0.
                built_curve = market.curves[curve]
                built_curve_date_dfs: list[tuple[date, float]] = list(built_curve.date_dfs)
                built_curve_dates = {d for d, _ in built_curve_date_dfs}
                for curve_maturity in curve_maturities:
                    if curve_maturity not in built_curve_dates:
                        # This curve_maturity is not in current market curve
                        x0_df = built_curve.get_df(curve_maturity)
                        built_curve_date_dfs.append((curve_maturity, x0_df))
                    else:
                        ... # TODO: Avoid this maturity from being solved as it was previously solved for other instruments.
                if len(built_curve_date_dfs) > len(built_curve.curve_points):
                    built_curve_date_dfs.sort(key=lambda pt: pt[0])
                    x0 = built_curve_date_dfs
            if x0:
                curves_to_build[curve] = x0

        if not curves_to_build:
            continue

        curves_quotes: set[InstrumentQuote] = set()
        flat_x0s: list[float] = []
        curve_key_startend_index: dict[CurveKey, tuple[int, int]] = {}
        start = 0
        # Now let's build a x0 vector with all dfs of all of the curves that need rebuilding.
        for curve_to_build, x0 in curves_to_build.items():
            end = start + len(x0)
            curve_key_startend_index[curve_to_build] = (start, end)
            curves_quotes.update({q for q, curves_needed in needs.items() if curves_needed.issubset(curves_to_build)}) # Quotes used to solve are those that uses curves to build or less.
            flat_x0s += [df for _, df in x0]
            start = end

        instruments = [q.get_instrument() for q in curves_quotes]

        def f(flat_x: list[float]) -> list[float]:
            for curve, (start, end) in curve_key_startend_index.items():
                dfs = flat_x[start:end]
                dates = [d for d, _ in curves_to_build[curve]]
                market.curves[curve] = ZeroCouponCurve(t, date_dfs=list(zip(dates, dfs)))
            return [Calculator.valuate(instrument, market, riskless_index, Currency(CurrencyName.CLP)) for instrument in instruments]

        result = least_squares(f, flat_x0s, bounds=(5e-2, 1.5))
        if not result.success:
            raise ValueError(
                f"Curve solve failed to converge for curve group {set(curves_to_build)}: {result.message}"
            )
        
        # Add solved curves to market
        for curve, (start, end) in curve_key_startend_index.items():
            solved_dfs = result.x[start:end]
            dates_dfs = curves_to_build[curve]
            dates, _ = zip(*dates_dfs)
            solved_curve = ZeroCouponCurve(t, date_dfs=list(zip(dates, solved_dfs)))
            market.curves[curve] = solved_curve
