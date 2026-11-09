from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum

from ...market.currencies import Currency
from ...market.index import InterestIndex
from .legs import SwapLeg


class LegSide(Enum):
    """Which leg of a swap: the one received or the one paid."""

    RECEIVE = "RECEIVE"
    PAY = "PAY"


@dataclass
class SwapBase(ABC):
    """Base for all swap instruments.

    Parameters
    ----------
    payment_currency : Currency, optional
        Settlement currency. Required for a non-deliverable swap whose legs are
        in different currencies; every flow in another currency is converted at
        its FX fixing. Must be None for a deliverable swap, where each leg
        settles in its own currency.
    collateral_index : InterestIndex, optional
        Index remunerating the CSA collateral; None means uncollateralised.
    """

    payment_currency: Currency | None = field(default=None)
    collateral_index: InterestIndex | None = field(default=None)


@dataclass(kw_only=True)
class Swap(SwapBase):
    """Swap instrument made of two legs.

    Direction is given by which leg is received (added) and which is paid
    (subtracted). Build legs with :mod:`fintoolsom.derivatives.swaps.builders`.

    Parameters
    ----------
    receive_leg : SwapLeg
        Leg received.
    pay_leg : SwapLeg
        Leg paid.
    is_deliverable : bool, default False
        True when each leg settles physically in its own currency. False when
        the swap settles in `payment_currency` (non-deliverable); irrelevant
        when both legs share a currency.

    Raises
    ------
    ValueError
        If a non-deliverable cross-currency swap has no `payment_currency`, or a
        deliverable swap has one.
    """

    receive_leg: SwapLeg
    pay_leg: SwapLeg
    is_deliverable: bool = False

    def __post_init__(self):
        if self.is_deliverable and self.payment_currency is not None:
            raise ValueError(
                "A deliverable swap settles each leg in its own currency: payment_currency must be None."
            )
        if (
            not self.is_deliverable
            and self.payment_currency is None
            and self.receive_leg.currency != self.pay_leg.currency  # type: ignore[attr-defined]
        ):
            raise ValueError(
                "A non-deliverable cross-currency swap needs a payment_currency."
            )
        if (
            self.payment_currency is not None
            and not self.payment_currency.is_deliverable
        ):
            raise ValueError(
                f"payment_currency {self.payment_currency} cannot be delivered; use "
                f"{self.payment_currency.settlement_currency}."
            )

    def __str__(self) -> str:
        if self.is_deliverable:
            mode = "deliverable"
        elif self.payment_currency is not None:
            mode = f"non-deliverable, settles {self.payment_currency}"
        else:
            mode = "single currency"
        collateral = getattr(self.collateral_index, "name", None) or "uncollateralised"
        maturity = max(self.receive_leg.end_dates[-1], self.pay_leg.end_dates[-1])

        def fmt(label: str, leg: SwapLeg) -> str:
            return f"  {label:<9}{leg}"

        return (
            f"Swap  maturity {maturity}  ({mode}, collateral: {collateral})\n"
            f"{fmt('Receive', self.receive_leg)}\n"
            f"{fmt('Pay', self.pay_leg)}"
        )

    def leg(self, side: LegSide) -> SwapLeg:
        """The swap's leg on `side`.

        Parameters
        ----------
        side : LegSide
            ``LegSide.RECEIVE`` or ``LegSide.PAY``.

        Returns
        -------
        SwapLeg

        Raises
        ------
        TypeError
            If `side` is not a :class:`LegSide`.
        """
        if side is LegSide.RECEIVE:
            return self.receive_leg
        if side is LegSide.PAY:
            return self.pay_leg
        raise TypeError(f"side must be a LegSide, got {type(side).__name__}.")

    @property
    def settlement_currency(self) -> Currency | None:
        """Currency every flow settles in, or None when each leg settles in its own."""
        return None if self.is_deliverable else self.payment_currency

    def leg_settlement_currency(self, leg: SwapLeg) -> Currency:
        """Currency the flows of `leg` are paid in.

        The swap's `settlement_currency` when non-deliverable; otherwise the leg
        currency, unless that is a unit of account that cannot be delivered (the
        UF, ``CLF``), whose flows are always paid in its settlement currency (CLP)
        at the rate of their FX fixing dates, even in a deliverable swap.

        Parameters
        ----------
        leg : SwapLeg
            One of the swap's legs.

        Returns
        -------
        Currency
        """
        settlement = self.settlement_currency
        if settlement is not None:
            return settlement
        return leg.currency.settlement_currency  # type: ignore[attr-defined]

    def valuate(self, market, currency: Currency) -> float:
        """Mark-to-market in `currency`; see :meth:`Calculator.get_swap_mtm`.

        Parameters
        ----------
        market : Market
            Market at the valuation date.
        currency : Currency
            Reporting currency.

        Returns
        -------
        float
        """
        from ..calculator import Calculator

        return Calculator.get_swap_mtm(self, market, currency)
