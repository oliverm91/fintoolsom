from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING

from ..market.quotes import _SwapQuote, _ForwardQuote

if TYPE_CHECKING:
    from ..market.quotes import InstrumentQuote


def _quote_pillar_date(quote: InstrumentQuote) -> date:
    """Terminal date of the instrument described by this quote.

    For swap quotes: effective maturity (respects spot-lag, start_date, maturity_date).
    For forward/NDF quotes: effective payment date."""
    if isinstance(quote, _SwapQuote):
        start = quote._effective_start()
        return quote._effective_maturity(start)
    if isinstance(quote, _ForwardQuote):
        return quote._effective_payment_date()
    raise TypeError(f"Cannot determine pillar date for {type(quote).__name__}")


def _group_by_maturity(
    quotes: list[InstrumentQuote],
) -> list[tuple[date, list[InstrumentQuote]]]:
    """Sort quotes shortest→longest by pillar date; collect quotes that share the same date.

    Quotes from different types at the same tenor but with slightly different calendar-
    adjusted dates (e.g. one lands on Mon, another on Tue) end up in separate buckets.
    In that case the shared curve will receive two closely-spaced pillars — acceptable
    for most markets. True grouping with max(t1, t2) as a merged pillar is a future
    enhancement."""
    groups: dict[date, list[InstrumentQuote]] = defaultdict(list)
    for q in quotes:
        groups[_quote_pillar_date(q)].append(q)
    return sorted(groups.items())
