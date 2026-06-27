from abc import abstractmethod
from dataclasses import dataclass, field, KW_ONLY
from enum import Enum

from .currencies import Currency, CurrencyPair, FX_Rate
from .localities import Locality
from .index import InterestIndex
from ..rates import Rate


# ── Forward Quotes ─────────────────────────────────────────────────────────

@dataclass
class ForwardQuote:
    currency_pair: CurrencyPair
    value: float
    locality: Locality = field(default=None)

    @abstractmethod
    def to_outright(self, spot: FX_Rate) -> float:
        pass


@dataclass
class ForwardPriceQuote(ForwardQuote):
    """Forward quoted as a final outright price (e.g. UF forwards)."""

    def to_outright(self, spot: FX_Rate) -> float:
        return self.value


@dataclass
class ForwardPointsQuote(ForwardQuote):
    """Forward quoted as points added to spot. outright = spot + value / points_divisor."""
    # Common divisors: 1, 100, 10000 (pips for most FX pairs).
    points_divisor: int = field(default=1)

    def __post_init__(self):
        if self.points_divisor <= 0:
            raise ValueError(
                f"points_divisor must be a positive integer. Got {self.points_divisor}."
            )

    def to_outright(self, spot: FX_Rate) -> float:
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
    index: InterestIndex
    spread: BasisPoints = field(default=None)


@dataclass
class SwapQuote:
    """Base class for swap quotes. collateral_index=None means uncollateralised."""
    quoted_side: QuotedSide
    locality: Locality = field(default=None)
    _: KW_ONLY
    collateral_index: InterestIndex = field(default=None)


@dataclass
class IRSQuote(SwapQuote):
    """IRS or OIS: fixed vs floating leg, same currency. Fixed rate is the quoted value."""
    _: KW_ONLY
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
    _: KW_ONLY
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
    _: KW_ONLY
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
    _: KW_ONLY
    receive_leg: FloatingLegSpec
    pay_leg: FloatingLegSpec

    def __post_init__(self):
        if self.receive_leg.currency == self.pay_leg.currency:
            raise ValueError(
                "CrossCurrencyFloatFloat requires legs in different currencies."
            )
