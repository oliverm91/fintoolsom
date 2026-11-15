"""Simplified swaps: a cheaper valuation for curve building.

They drop features a curve-building quote does not need, so the calculator can
value them on a shorter path (:mod:`fintoolsom.derivatives.calculator._simplified_swap_pv`):

* :class:`SimplifiedSwap`: every leg settles in its own currency (deliverable,
  a UF leg included) and pays on its accrual end dates (no payment lag).
* :class:`NonStartedSimplifiedSwap`: also valued as if no coupon has started or
  fixed, so its legs need no fixings and no split of paid, running and future
  coupons.

No builder returns them yet: wrap the legs of a built swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np

from ...market.currencies import Currency
from .legs import SwapLeg
from .swaps import Swap


def _without_payment_lag(leg: SwapLeg) -> SwapLeg:
    """`leg` paying (and fixing FX) on its accrual end dates.

    The same leg when it already does; otherwise a copy with new coupons.
    """
    if np.array_equal(leg._payment_ordinals, leg._end_ordinals) and np.array_equal(
        leg._fx_fixing_ordinals, leg._end_ordinals
    ):
        return leg
    coupons = [
        replace(c, payment_date=c.end_date, fx_fixing_date=c.end_date)
        for c in leg.coupons
    ]
    return replace(leg, coupons=coupons)


@dataclass(kw_only=True)
class SimplifiedSwap(Swap):
    """Two-leg swap valued on a simplified path, for curve building.

    It forgets the settlement currency and the payment lag:

    * every leg settles in its own currency, as a deliverable swap. A UF
      (``CLF``) leg is allowed and is valued in UF, discounted on the UF curve,
      not paid in CLP at its fixing.
    * every coupon pays on its accrual end date. Legs built with a payment lag
      are copied with payment and FX fixing dates set to their end dates.

    Parameters
    ----------
    receive_leg : SwapLeg
        Leg received.
    pay_leg : SwapLeg
        Leg paid.
    collateral_index : InterestIndex, optional
        Index remunerating the CSA collateral; None means uncollateralised.
    """

    payment_currency: Currency | None = field(default=None, init=False)
    is_deliverable: bool = field(default=True, init=False)

    def __post_init__(self):
        # No Swap checks: there is no payment currency and a CLF leg is accepted.
        self.receive_leg = _without_payment_lag(self.receive_leg)
        self.pay_leg = _without_payment_lag(self.pay_leg)

    def leg_settlement_currency(self, leg: SwapLeg) -> Currency:
        """The leg's own currency: every leg settles in it."""
        return leg.currency  # type: ignore[attr-defined]

    def valuate(self, market, currency: Currency) -> float:
        """Mark-to-market in `currency`; see :meth:`Calculator.valuate`."""
        from ..calculator import Calculator

        return Calculator.valuate(self, market, currency)


@dataclass(kw_only=True)
class NonStartedSimplifiedSwap(SimplifiedSwap):
    """:class:`SimplifiedSwap` valued as if no coupon has started.

    Every coupon is unpaid and every floating coupon is projected over its whole
    accrual period: realised fixings are never read, including a term-rate
    fixing already set before the start date.

    Valuation raises ``ValueError`` after :attr:`effective_date`. A curve built
    as of the quote date satisfies it: its swaps start on or after that date.

    Attributes
    ----------
    effective_date : date
        First accrual start of either leg: the latest valid valuation date. It is
        the swap's, not a curve's: a swap starting before its index curve's
        effective date projects those days on the curve's spot segment.
    """

    effective_date: date = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.effective_date = min(
            self.receive_leg.start_dates[0], self.pay_leg.start_dates[0]
        )
