from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from scipy.optimize import brentq, fsolve, least_squares  # type: ignore[import-untyped]

from ..dates import ActualDayCountConvention
from ..market.index import Index
from ..market.currencies import Currency
from ..market.market import Market
from ..market.quotes import IRSQuote
from ..rates import ZeroCouponCurve
from ._needs import CurveKey, _curves_needed
from ._grouping import _group_by_maturity


class InsufficientQuotesError(ValueError):
    """Raised when a maturity bucket has more unknown curves than available quotes.

    Example: two curves (SOFR + LIBOR) need pillars at 3Y but only one
    instrument is available. The system is under-determined."""


def _build_market(
    curve_pillars: dict[CurveKey, list[tuple[date, float]]],
    riskless_index: Index,
    t: date,
) -> Market:
    """Construct a Market from the currently solved curve pillars.

    Index keys that equal riskless_index are the normalised form of
    (riskless_index, riskless_index.currency) and populate both
    projection_curves and discount_curves for that currency.
    All other Index keys (floating projection indices) go into
    projection_curves only. Tuple keys go into discount_curves only."""
    projection_curves: dict[Index, ZeroCouponCurve] = {}
    discount_curves: dict[tuple[Index, Currency], ZeroCouponCurve] = {}

    for key, pillars in curve_pillars.items():
        if not pillars:
            continue
        curve = ZeroCouponCurve(curve_date=t, date_dfs=pillars)
        if isinstance(key, tuple):
            idx, ccy = key
            discount_curves[(idx, ccy)] = curve
        elif key == riskless_index:
            # Normalised riskless: serves as both projection and self-currency discount.
            projection_curves[key] = curve
            discount_curves[(riskless_index, key.currency)] = curve
        else:
            projection_curves[key] = curve

    return Market(t=t, discount_curves=discount_curves, projection_curves=projection_curves)


def _initial_df(
    unknown_key: CurveKey,
    solver_quotes: list,
    curve_pillars: dict[CurveKey, list[tuple[date, float]]],
    t: date,
    pillar_date: date,
) -> float:
    """Compute an initial DF guess for the solver.

    Strategy (in order):
    1. Log-linear extrapolation from the curve's most recent solved pillar.
    2. For IRSQuote: approximate zero rate ≈ fixed rate → DF = exp(-r * yf).
    3. Fallback: 0.99."""
    existing = curve_pillars.get(unknown_key, [])
    if existing:
        last_date, last_df = existing[-1]
        days_last = (last_date - t).days
        days_new = (pillar_date - t).days
        if days_last > 0 and days_new > days_last and last_df > 0:
            zero_rate = -np.log(last_df) / days_last * 365
            return float(np.exp(-zero_rate * days_new / 365))

    for q in solver_quotes:
        if isinstance(q, IRSQuote):
            rate_val = q.fixed_leg.rate.value
            yf = ActualDayCountConvention.get_time_fraction(t, pillar_date, 365)
            if yf > 0:
                return float(np.exp(-rate_val * yf))

    return 0.99


def _get_spot(spot_rates: dict, cp) -> float:
    """Look up a spot rate for cp, trying the inverted pair if needed."""
    v = spot_rates.get(cp)
    if v is not None:
        return float(v)
    inv = cp.invert()
    v2 = spot_rates.get(inv)
    if v2 is not None:
        return 1.0 / float(v2)
    raise ValueError(f"spot_rates has no entry for {cp} or {inv}")


def _instrument_mtm(quote, market: Market, riskless_index: Index, spot_rates: dict | None) -> float:
    """Compute par-MTM for a single InstrumentQuote given a partial Market.

    Dispatches to swap MTM or FX forward/NDF MTM based on quote type.
    For forward quotes, spot_rates must contain the relevant CurrencyPair."""
    from ..market.quotes import _SwapQuote as _SQ, _ForwardQuote as _FQ
    from ..derivatives.calculator import Calculator
    from ..derivatives.forwards.forwards import NDF

    if isinstance(quote, _SQ):
        inst = quote.get_instrument()
        return Calculator.get_swap_mtm(inst, market, riskless_index, inst.receive_leg.currency)  # type: ignore

    if isinstance(quote, _FQ):
        inst: Any = quote.get_instrument()
        cp = inst.currency_pair
        dc = market.discount_curves.get((riskless_index, cp.quote_currency))
        fc = market.discount_curves.get((riskless_index, cp.base_currency))
        if dc is None or fc is None:
            return 0.0  # curves not yet built for this pair
        if spot_rates is None:
            raise ValueError(f"spot_rates required to value {type(quote).__name__}")
        spot = _get_spot(spot_rates, cp)
        sign = 1.0 if inst.is_buy else -1.0
        if isinstance(inst, NDF):
            df_d_fix = dc.get_df(inst.fixing_date)
            df_f_fix = fc.get_df(inst.fixing_date)
            fwd = spot * df_f_fix / df_d_fix
            df_d_pmt = dc.get_df(inst.payment_date)
            return sign * inst.notional * (fwd - inst.strike) * df_d_pmt
        df_d = dc.get_df(inst.payment_date)
        df_f = fc.get_df(inst.payment_date)
        return sign * inst.notional * (spot * df_f - inst.strike * df_d)

    raise NotImplementedError(f"No MTM implementation for {type(quote).__name__}")


def build_curves(
    quotes: list,
    riskless_index: Index,
    spot_rates: dict | None = None,
) -> tuple[dict[Index, ZeroCouponCurve], dict[tuple[Index, Currency], ZeroCouponCurve]]:
    """Bootstrap ZeroCouponCurve objects from a list of InstrumentQuotes.

    Returns (projection_curves, discount_curves) where the two dicts share the same
    ZeroCouponCurve object for curves that are both a projection index and the riskless
    discount for their own currency (e.g. SOFR for USD, ICP for CLP).

    Algorithm
    ---------
    1.  Derive valuation date t from the earliest quote_date across all quotes.
    2.  Build a dependency graph: each quote → frozenset[CurveKey] of curves it needs.
    3.  Collect all unique CurveKeys across all quotes.
    4.  Sort quotes by terminal pillar date (shortest tenor first).
    5.  Iterate maturity buckets:
        a.  Identify all CurveKeys referenced by this bucket's quotes.
        b.  Raise InsufficientQuotesError if the system is under-determined
            (#quotes < #unknown_curves). Over-determined systems are solved via
            least-squares.
        c.  Build a partial Market from all pillars solved in previous iterations.
        d.  Find DF values for each curve's new pillar:
            - 1 unknown, 1 quote  → brentq (DF ∈ (1e-8, 2.0))
            - N unknowns, N quotes → fsolve
            - N unknowns, M>N quotes → least_squares (minimise residual MTMs)
        e.  Append the solved (pillar_date, df) to each curve's pillar list.
    6.  Build final ZeroCouponCurve objects from the accumulated pillar lists.
    7.  Assemble and return the two output dicts.

    Parameters
    ----------
    spot_rates:
        Required when forwards or NDFs are included in quotes. Maps CurrencyPair
        to spot rate (or its inverse — both directions are tried)."""
    # 1. Derive t from any quote that carries quote_date
    all_dates = [q.quote_date for q in quotes if getattr(q, "quote_date", None) is not None]
    if not all_dates:
        return {}, {}
    t: date = min(all_dates)  # type: ignore[type-var]

    # 2. Dependency graph (all quote types)
    needs: dict = {q: _curves_needed(q, riskless_index) for q in quotes}

    # 3. All curve keys
    all_keys: set[CurveKey] = set()
    for ks in needs.values():
        all_keys.update(ks)

    if not all_keys:
        return {}, {}

    # 4. Group by maturity (shortest first)
    maturity_groups = _group_by_maturity(quotes)

    # 5. Bootstrap
    curve_pillars: dict[CurveKey, list[tuple[date, float]]] = {k: [] for k in all_keys}

    for pillar_date, bucket_quotes in maturity_groups:
        bucket_keys: set[CurveKey] = set()
        for q in bucket_quotes:
            bucket_keys.update(needs[q])
        unknown_keys = list(bucket_keys)

        if len(bucket_quotes) < len(unknown_keys):
            raise InsufficientQuotesError(
                f"At pillar {pillar_date}: {len(unknown_keys)} curve(s) need new pillars "
                f"but only {len(bucket_quotes)} quote(s) available (under-determined)."
            )

        x0 = [_initial_df(k, bucket_quotes, curve_pillars, t, pillar_date) for k in unknown_keys]

        def _mtm(q, mkt: Market) -> float:
            return _instrument_mtm(q, mkt, riskless_index, spot_rates)

        if len(unknown_keys) == 1 and len(bucket_quotes) == 1:
            key = unknown_keys[0]
            quote = bucket_quotes[0]

            def _obj(df_val, _key=key, _quote=quote):
                test = {k: list(v) for k, v in curve_pillars.items()}
                test[_key] = curve_pillars[_key] + [(pillar_date, float(df_val))]
                return _mtm(_quote, _build_market(test, riskless_index, t))

            try:
                df_sol: float = brentq(_obj, 1e-8, 2.0, xtol=1e-10)  # type: ignore[assignment]
            except ValueError as exc:
                raise ValueError(
                    f"brentq failed at pillar {pillar_date} for curve {key}: {exc}"
                ) from exc
            curve_pillars[key].append((pillar_date, df_sol))

        elif len(bucket_quotes) == len(unknown_keys):
            # Exactly determined N-D system
            def _res_exact(dfs, _keys=unknown_keys, _quotes=bucket_quotes):
                test = {k: list(v) for k, v in curve_pillars.items()}
                for k_i, df_i in zip(_keys, dfs):
                    test[k_i] = curve_pillars[k_i] + [(pillar_date, float(df_i))]
                mkt = _build_market(test, riskless_index, t)
                return [_mtm(q, mkt) for q in _quotes]

            dfs_sol = fsolve(_res_exact, x0)  # type: ignore[assignment]
            for k_i, df_i in zip(unknown_keys, np.asarray(dfs_sol, dtype=float)):
                curve_pillars[k_i].append((pillar_date, float(df_i)))

        else:
            # Over-determined: minimise sum of squared MTMs
            bounds_lo = [1e-8] * len(unknown_keys)
            bounds_hi = [2.0] * len(unknown_keys)

            def _res_over(dfs, _keys=unknown_keys, _quotes=bucket_quotes):
                test = {k: list(v) for k, v in curve_pillars.items()}
                for k_i, df_i in zip(_keys, dfs):
                    test[k_i] = curve_pillars[k_i] + [(pillar_date, float(df_i))]
                mkt = _build_market(test, riskless_index, t)
                return [_mtm(q, mkt) for q in _quotes]

            result = least_squares(_res_over, x0, bounds=(bounds_lo, bounds_hi))
            for k_i, df_i in zip(unknown_keys, np.asarray(result.x, dtype=float)):
                curve_pillars[k_i].append((pillar_date, float(df_i)))

    # 6. Build final ZeroCouponCurve objects
    final: dict[CurveKey, ZeroCouponCurve] = {}
    for key, pillars in curve_pillars.items():
        if pillars:
            final[key] = ZeroCouponCurve(curve_date=t, date_dfs=pillars)

    # 7. Assemble output dicts (shared objects where keys are equivalent)
    projection_curves: dict[Index, ZeroCouponCurve] = {}
    discount_curves: dict[tuple[Index, Currency], ZeroCouponCurve] = {}

    for key, curve in final.items():
        if isinstance(key, tuple):
            idx, ccy = key
            discount_curves[(idx, ccy)] = curve
        elif key == riskless_index:
            projection_curves[key] = curve
            discount_curves[(riskless_index, key.currency)] = curve
        else:
            projection_curves[key] = curve

    return projection_curves, discount_curves
