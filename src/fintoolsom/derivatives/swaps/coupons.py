from dataclasses import dataclass, field
from datetime import date

from fintoolsom.rates.Rates import Rate


@dataclass
class SwapCoupon:
    """Base payment unit of a swap leg.

    All fields are set at construction and treated as immutable.

    Parameters
    ----------
    residual : float
        Outstanding notional the coupon accrues on.
    amortization : float
        Notional repaid with this coupon. Only the final coupon carries the
        notional, which is the final notional exchange; there is no initial one.
    start_date : date
        Accrual start.
    end_date : date
        Accrual end.
    payment_date : date
        Settlement date of the flow.
    fx_fixing_date : date, optional
        Date the FX rate converting this flow into the settlement currency is
        fixed. Only used when the leg settles non-deliverably in another
        currency. Defaults to `payment_date`.

    Raises
    ------
    ValueError
        If `fx_fixing_date` is after `payment_date`.
    """

    residual: float
    amortization: float
    start_date: date
    end_date: date
    payment_date: date
    fx_fixing_date: date | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.fx_fixing_date is None:
            self.fx_fixing_date = self.payment_date
        if self.fx_fixing_date > self.payment_date:
            raise ValueError(
                f"fx_fixing_date {self.fx_fixing_date} is after payment_date {self.payment_date}."
            )


@dataclass
class FixedCoupon(SwapCoupon):
    """Fixed-rate coupon.

    Parameters
    ----------
    rate : Rate
        Fixed rate accrued over ``[start_date, end_date]``.

    Attributes
    ----------
    flow : float
        Interest on `residual` at `rate` plus `amortization`.
    """

    rate: Rate
    flow: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        interest = self.rate.get_accrued_interest(self.residual, self.start_date, self.end_date)
        self.flow = float(interest) + self.amortization


@dataclass
class FloatingCoupon(SwapCoupon):
    """Floating-rate coupon.

    Parameters
    ----------
    spread : Rate
        Spread accrued over ``[start_date, end_date]`` on top of the index
        interest, added (not compounded) to the floating flow.
    """

    spread: Rate


@dataclass
class TermRateCoupon(FloatingCoupon):
    """Floating coupon on a term-rate (IBOR-style) index.

    Parameters
    ----------
    fixing_date : date, optional
        Date the single term rate covering the whole coupon is fixed.
        Defaults to `start_date`.
    """

    fixing_date: date | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.fixing_date is None:
            self.fixing_date = self.start_date


@dataclass
class OvernightCoupon(FloatingCoupon):
    """Floating coupon on an overnight index, compounded daily (OIS-style)."""


@dataclass
class XCCYCoupon(FloatingCoupon):
    """Floating coupon of a cross-currency leg.

    Compounded daily, like :class:`OvernightCoupon`.
    """
