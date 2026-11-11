from dataclasses import KW_ONLY, dataclass
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
    time_fraction : TimeFractionBase
        Day-count basis of the accrual.
    payment_lag : int, default 0
        Business days between each period end date and its payment date
        (keyword-only).
    """

    currency: Currency
    payment_frequency: PaymentFrequency
    adj_convention: AdjustmentDateConventionBase
    time_fraction: TimeFractionBase
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
