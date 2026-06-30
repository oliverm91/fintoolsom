from dataclasses import dataclass, field
from enum import Enum

from ..dates import AdjustmentDateConventionBase
from ..dates.time_fractions import TimeFractionBase
from .currencies import Currency
from .index import InterestIndex
from ..rates import Rate


class PaymentFrequency(Enum):
    DAILY = "DAILY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMIANNUAL = "SEMIANNUAL"
    ANNUAL = "ANNUAL"


@dataclass
class BasisPoints:
    """Spread in basis points over a floating index. Decimal rate = value / divisor."""
    value: float
    divisor: int = field(default=10_000)


@dataclass
class LegSpec:
    """Structural specification for one swap leg: how it accrues and when it pays.
    Carries enough information for get_instrument() to generate a full leg schedule."""
    currency: Currency
    payment_frequency: PaymentFrequency
    adj_convention: AdjustmentDateConventionBase
    time_fraction: TimeFractionBase


@dataclass
class FixedLegSpec(LegSpec):
    rate: Rate


@dataclass
class FloatingLegSpec(LegSpec):
    index: InterestIndex
    spread: BasisPoints | None = field(default=None)
