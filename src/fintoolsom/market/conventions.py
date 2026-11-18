from dataclasses import KW_ONLY, dataclass
from enum import Enum

from ..dates import AdjustmentDateConventionBase
from ..rates import Rate
from .currencies import Currency
from .index import InterestIndex


class PaymentFrequency(Enum):
    DAILY = "DAILY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMIANNUAL = "SEMIANNUAL"
    ANNUAL = "ANNUAL"


@dataclass
class LegSpec:
    """Structural specification of one swap leg: how it accrues and when it pays.

    Carries what a quote's ``get_instrument`` needs to generate the leg schedule.

    Parameters
    ----------
    currency : Currency
        Currency of the flows.
    payment_frequency : PaymentFrequency
        Coupon frequency.
    adj_convention : AdjustmentDateConventionBase
        Business-day adjustment of the schedule dates.
    payment_lag : int, default 0
        Business days between each period end date and its payment date
        (keyword-only).

    Notes
    -----
    The accrual basis is the day count of the leg's rates (the fixed rate, the
    spread, the index), not a field of the spec.
    """

    currency: Currency
    payment_frequency: PaymentFrequency
    adj_convention: AdjustmentDateConventionBase
    _: KW_ONLY
    payment_lag: int = 0


@dataclass
class FixedLegSpec(LegSpec):
    """Fixed leg specification.

    Parameters
    ----------
    rate : Rate
        Fixed rate.
    """

    rate: Rate


@dataclass
class FloatingLegSpec(LegSpec):
    """Floating leg specification.

    Parameters
    ----------
    index : InterestIndex
        Floating index.
    spread : Rate
        Spread over the index.
    """

    index: InterestIndex
    spread: Rate
