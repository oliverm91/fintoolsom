from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from ...market.currencies import Currency
from ...market.index import InterestIndex
from .legs import SwapLeg


@dataclass
class SwapBase(ABC):
    """Base for all swap instruments.

    ``payment_currency`` records the intended settlement currency when unambiguous
    (IRS, IRBasis, non-deliverable XCCY). It is None for deliverable XCCY swaps.
    ``collateral_index`` is the index used for CSA discounting; None = uncollateralised."""
    payment_currency: Currency | None = field(default=None)
    collateral_index: InterestIndex | None = field(default=None)


# ── Single instrument class ───────────────────────────────────────────────

@dataclass(kw_only=True)
class Swap(SwapBase):
    """Universal swap instrument. Direction is encoded by which leg is
    ``receive_leg`` (added) and which is ``pay_leg`` (subtracted).

    Use builders from ``fintoolsom.derivatives.swaps.builders`` to construct
    individual legs, then pass them here directly."""
    receive_leg: SwapLeg
    pay_leg: SwapLeg
    is_deliverable: bool = False

    def valuate(self, market, riskless_index, currency: Currency) -> float:
        from ..calculator import Calculator
        return Calculator.get_swap_mtm(self, market, riskless_index, currency)
