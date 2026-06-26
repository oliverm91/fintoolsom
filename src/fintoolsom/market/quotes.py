from dataclasses import dataclass, field, KW_ONLY
from enum import Enum

from .currencies import Currency, CurrencyPair, FX_Rate
from .index import Index
from ..rates import Rate


# ── Forward Quotes ─────────────────────────────────────────────────────────

class ForwardQuoteType(Enum):
    FINAL_PRICE = "FINAL_PRICE"
    FORWARD_POINTS = "FORWARD_POINTS"


@dataclass
class ForwardQuote:
    currency_pair: CurrencyPair
    value: float
    quote_type: ForwardQuoteType
    # Only for FORWARD_POINTS: forward_price = spot + value / points_divisor.
    # Common values: 1, 100, 10000 (pips for most FX pairs).
    points_divisor: int = field(default=1)

    def __post_init__(self):
        if self.quote_type == ForwardQuoteType.FORWARD_POINTS and self.points_divisor <= 0:
            raise ValueError(
                f"points_divisor must be a positive integer. Got {self.points_divisor}."
            )

    def to_outright(self, spot: FX_Rate) -> float:
        if self.quote_type == ForwardQuoteType.FINAL_PRICE:
            return self.value
        if spot.currency_pair == self.currency_pair:
            effective_spot = spot.value
        elif spot.currency_pair == self.currency_pair.invert():
            effective_spot = spot.invert().value
        else:
            raise ValueError(
                f"Spot currency pair {spot.currency_pair} is incompatible with forward currency pair {self.currency_pair}."
            )
        return effective_spot + self.value / self.points_divisor


# ── Swap Quotes ────────────────────────────────────────────────────────────

class PaymentFrequency(Enum):
    DAILY = "DAILY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMIANNUAL = "SEMIANNUAL"
    ANNUAL = "ANNUAL"


class QuotedSide(Enum):
    RECEIVE = "RECEIVE"
    PAY = "PAY"


@dataclass
class BasisPoints:
    """Spread in basis points over a floating index. Decimal rate = value / divisor."""
    value: float
    divisor: int = field(default=10_000)


@dataclass
class LegSpec:
    currency: Currency
    payment_frequency: PaymentFrequency


@dataclass
class FixedLegSpec(LegSpec):
    rate: Rate


@dataclass
class FloatingLegSpec(LegSpec):
    index_name: str
    spread: BasisPoints = field(default=None)


@dataclass
class SwapQuote:
    """Base class for swap quotes. collateral_index=None means uncollateralised."""
    quoted_side: QuotedSide
    _: KW_ONLY
    collateral_index: Index = field(default=None)


@dataclass
class IRSQuote(SwapQuote):
    """IRS or OIS: fixed vs floating leg, same currency. Fixed rate is the quoted value."""
    fixed_leg: FixedLegSpec
    floating_leg: FloatingLegSpec

    def __post_init__(self):
        if self.fixed_leg.currency != self.floating_leg.currency:
            raise ValueError(
                f"IRS requires both legs in the same currency. "
                f"Got {self.fixed_leg.currency} and {self.floating_leg.currency}."
            )


@dataclass
class IRBasisQuote(SwapQuote):
    """Float vs float basis swap, same currency. The spread on the quoted_side leg is the quoted value."""
    receive_leg: FloatingLegSpec
    pay_leg: FloatingLegSpec

    def __post_init__(self):
        if self.receive_leg.currency != self.pay_leg.currency:
            raise ValueError(
                f"IR Basis requires both legs in the same currency. "
                f"Got {self.receive_leg.currency} and {self.pay_leg.currency}."
            )


@dataclass
class CrossCurrencyFixedFloatQuote(SwapQuote):
    """Fixed vs floating cross-currency swap. Legs must be in different currencies."""
    fixed_leg: FixedLegSpec
    floating_leg: FloatingLegSpec

    def __post_init__(self):
        if self.fixed_leg.currency == self.floating_leg.currency:
            raise ValueError(
                "CrossCurrencyFixedFloat requires legs in different currencies."
            )


@dataclass
class CrossCurrencyFloatFloatQuote(SwapQuote):
    """Float vs float cross-currency swap. Legs must be in different currencies."""
    receive_leg: FloatingLegSpec
    pay_leg: FloatingLegSpec

    def __post_init__(self):
        if self.receive_leg.currency == self.pay_leg.currency:
            raise ValueError(
                "CrossCurrencyFloatFloat requires legs in different currencies."
            )
