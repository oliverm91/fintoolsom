from dataclasses import dataclass, field
from datetime import date

from fintoolsom.rates.Rates import Rate


@dataclass
class SwapCoupon:
    """Base payment unit. All fields are set at construction and treated as immutable."""
    residual: float
    amortization: float
    start_date: date
    end_date: date
    payment_date: date


@dataclass
class FixedCoupon(SwapCoupon):
    rate: Rate
    flow: float = field(init=False)  # rate * time_fraction * residual + amortization

    def __post_init__(self):
        self.flow = self.rate.get_accrued_interest(self.residual, self.start_date, self.end_date) + self.amortization


@dataclass
class FloatingCoupon(SwapCoupon):
    spread: Rate


@dataclass
class TermRateCoupon(FloatingCoupon):
    """Floating coupon for a term-rate (IBOR-style) index. fixing_date defaults to start_date."""
    fixing_date: date | None = field(default=None)

    def __post_init__(self):
        if self.fixing_date is None:
            self.fixing_date = self.start_date


@dataclass
class OvernightCoupon(FloatingCoupon):
    """Floating coupon for an overnight compounding (OIS-style) index."""


@dataclass
class XCCYCoupon(FloatingCoupon):
    """Floating coupon for a cross-currency leg. fx_fixing_date defaults to end_date."""
    fx_fixing_date: date | None = field(default=None)

    def __post_init__(self):
        if self.fx_fixing_date is None:
            self.fx_fixing_date = self.end_date
