from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..market.index import Index
from ..market.currencies import Currency

if TYPE_CHECKING:
    from ..market.quotes import InstrumentQuote

# A curve is identified either as a projection index or as a (riskless_index, Currency) pair.
CurveKey = Index | tuple[Index, Currency]


def _normalize(key: CurveKey) -> CurveKey:
    """Collapse (index, index.currency) → index.

    A single ZeroCouponCurve object serves both projection_curves[index] and
    discount_curves[(riskless_index, index.currency)] when the index is its own
    riskless in that currency — e.g. SOFR for USD, ICP for CLP."""
    if isinstance(key, tuple):
        idx, ccy = key
        if cast(Index, idx).currency == ccy:
            return cast(Index, idx)
    return key


def _curves_needed(quote: InstrumentQuote, riskless_index: Index) -> frozenset[CurveKey]:
    """Return the set of curve keys required to value this quote.

    Keys are normalised so that (index, index.currency) collapses to index,
    meaning projection and discount share the same ZeroCouponCurve object."""
    from ..market.quotes import (
        IRSQuote as _IRSQuote,
        IRBasisQuote as _IRBasisQuote,
        CrossCurrencyFixedFloatQuote as _XCCYFixFloat,
        CrossCurrencyFloatFloatQuote as _XCCYBasis,
        _ForwardQuote,
    )

    keys: set[CurveKey] = set()

    # InterestIndex is a standalone ABC; concrete types (RateIndex, InterestPriceIndex)
    # also extend Index at runtime. Cast is safe — the type system just doesn't see it.
    def _idx(interest_index) -> Index:
        return cast(Index, interest_index)

    if isinstance(quote, _IRSQuote):
        keys.add((riskless_index, quote.fixed_leg.currency))
        keys.add(_idx(quote.floating_leg.index))
        if quote.collateral_index is not None:
            keys.add(_idx(quote.collateral_index))
            keys.add((riskless_index, _idx(quote.collateral_index).currency))

    elif isinstance(quote, _IRBasisQuote):
        keys.add((riskless_index, quote.receive_leg.currency))
        keys.add(_idx(quote.receive_leg.index))
        keys.add(_idx(quote.pay_leg.index))
        if quote.collateral_index is not None:
            keys.add(_idx(quote.collateral_index))
            keys.add((riskless_index, _idx(quote.collateral_index).currency))

    elif isinstance(quote, _XCCYFixFloat):
        keys.add((riskless_index, quote.fixed_leg.currency))
        keys.add((riskless_index, quote.floating_leg.currency))
        keys.add(_idx(quote.floating_leg.index))
        if quote.collateral_index is not None:
            keys.add(_idx(quote.collateral_index))
            keys.add((riskless_index, _idx(quote.collateral_index).currency))

    elif isinstance(quote, _XCCYBasis):
        keys.add((riskless_index, quote.receive_leg.currency))
        keys.add((riskless_index, quote.pay_leg.currency))
        keys.add(_idx(quote.receive_leg.index))
        keys.add(_idx(quote.pay_leg.index))
        if quote.collateral_index is not None:
            keys.add(_idx(quote.collateral_index))
            keys.add((riskless_index, _idx(quote.collateral_index).currency))

    elif isinstance(quote, _ForwardQuote):
        # FX forward / NDF: domestic discount = quote_currency, foreign discount = base_currency
        cp = quote.currency_pair
        keys.add((riskless_index, cp.quote_currency))
        keys.add((riskless_index, cp.base_currency))

    # _ForwardUFQuote: UF curve structure not covered by CurveKey — skip
    return frozenset(_normalize(k) for k in keys)


def _instruments_for_curve(
    quotes: list[InstrumentQuote],
    curve_key: CurveKey,
    riskless_index: Index,
) -> list[InstrumentQuote]:
    """Return all quotes that depend on curve_key for valuation."""
    normalized = _normalize(curve_key)
    return [q for q in quotes if normalized in _curves_needed(q, riskless_index)]
