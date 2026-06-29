from __future__ import annotations

from dataclasses import dataclass, field

from ...market.currencies import Currency
from ...market.index import InterestIndex
from .legs import FixedLeg, FloatingLeg, SwapLeg


@dataclass
class SwapBase:
    """Structural base for all swap instruments.

    ``payment_currency`` is the currency in which net payments are settled (may
    differ from the leg currency for compensated cross-currency swaps).
    ``collateral_index`` is the index used for CSA discounting; None means
    uncollateralised."""
    payment_currency: Currency
    collateral_index: InterestIndex | None = field(default=None)

    def valuate(self, market) -> float:
        raise NotImplementedError(
            f"{type(self).__name__}.valuate() is not yet implemented."
        )


# ── Single-currency swaps (always compensated) ────────────────────────────

@dataclass(kw_only=True)
class IRS(SwapBase):
    """Interest rate swap: one fixed leg vs one floating leg, same currency.
    Always compensated (non-deliverable)."""
    fixed_leg: FixedLeg
    floating_leg: FloatingLeg

    def __post_init__(self):
        if self.fixed_leg.currency != self.floating_leg.currency:
            raise ValueError(
                f"IRS requires both legs in the same currency. "
                f"Got {self.fixed_leg.currency} and {self.floating_leg.currency}."
            )


@dataclass(kw_only=True)
class IRBasis(SwapBase):
    """Interest rate basis swap: two floating legs, same currency.
    Always compensated (non-deliverable)."""
    leg_a: FloatingLeg
    leg_b: FloatingLeg

    def __post_init__(self):
        if self.leg_a.currency != self.leg_b.currency:
            raise ValueError(
                f"IRBasis requires both legs in the same currency. "
                f"Got {self.leg_a.currency} and {self.leg_b.currency}."
            )


# ── Cross-currency swaps (deliverable by default) ─────────────────────────

@dataclass(kw_only=True)
class CrossCurrencyFixFloat(SwapBase):
    """Cross-currency swap: fixed leg vs floating leg, different currencies."""
    fixed_leg: FixedLeg
    floating_leg: FloatingLeg
    is_deliverable: bool = True

    def __post_init__(self):
        if self.fixed_leg.currency == self.floating_leg.currency:
            raise ValueError(
                "CrossCurrencyFixFloat requires legs in different currencies."
            )


@dataclass(kw_only=True)
class CrossCurrencyBasis(SwapBase):
    """Cross-currency basis swap: two floating legs, different currencies."""
    leg_a: FloatingLeg
    leg_b: FloatingLeg
    is_deliverable: bool = True

    def __post_init__(self):
        if self.leg_a.currency == self.leg_b.currency:
            raise ValueError(
                "CrossCurrencyBasis requires legs in different currencies."
            )


@dataclass(kw_only=True)
class CrossCurrencyFixFix(SwapBase):
    """Cross-currency fixed-fixed swap: two fixed legs, different currencies."""
    leg_a: FixedLeg
    leg_b: FixedLeg
    is_deliverable: bool = True

    def __post_init__(self):
        if self.leg_a.currency == self.leg_b.currency:
            raise ValueError(
                "CrossCurrencyFixFix requires legs in different currencies."
            )


# ── Generic swap (pre-built legs) ─────────────────────────────────────────

@dataclass(kw_only=True)
class CustomSwap(SwapBase):
    """Swap assembled from any two pre-built legs. Use when none of the typed
    classes fits (e.g., UF-indexed legs, exotic schedules)."""
    receive_leg: SwapLeg
    pay_leg: SwapLeg
    is_deliverable: bool = False
