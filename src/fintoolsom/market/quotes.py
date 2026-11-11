from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, KW_ONLY
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING


from .conventions import FixedLegSpec, FloatingLegSpec
from .currencies import Currency, CurrencyPair, FX_Rate
from .localities import Locality
from .index import InterestIndex
from ..dates import (
    AdjustmentDateConventionBase,
    ModifiedFollowingConvention,
    get_calendar,
)
from ..dates.term import Term

if TYPE_CHECKING:
    from ..derivatives.forwards.forwards import Forward, NDF
    from ..derivatives.swaps.swaps import Swap


# ── Private: ABC and module constants ──────────────────────────────────────


class InstrumentQuote(ABC):
    """Market quote of an instrument; every concrete quote builds its instrument."""

    __hash__ = object.__hash__

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.__hash__ = object.__hash__  # type: ignore[method-assign]

    @abstractmethod
    def get_instrument(self) -> object:
        """Instrument described by the quote, with notional 100.

        Returns
        -------
        object
            Forward, NDF or Swap.
        """


_NOTIONAL = 100.0


def _check_settlement(is_deliverable: bool, payment_currency: Currency | None) -> None:
    """Validate a cross-currency quote's settlement, as :class:`Swap` does.

    Raises
    ------
    ValueError
        If a deliverable quote has a `payment_currency`, or a non-deliverable one
        has none.
    """
    if is_deliverable and payment_currency is not None:
        raise ValueError(
            "A deliverable cross-currency quote must not set payment_currency."
        )
    if not is_deliverable and payment_currency is None:
        raise ValueError(
            "A non-deliverable cross-currency quote needs a payment_currency."
        )


# ── Private: forward base classes ──────────────────────────────────────────


@dataclass
class _ForwardQuote(InstrumentQuote):
    currency_pair: CurrencyPair
    value: float
    is_buy: bool
    _: KW_ONLY
    payment_date: date | None = field(default=None)
    locality: Locality | None = field(default=None)
    fixing_date: date | None = field(
        default=None
    )  # set → NDF, None → deliverable Forward
    term: Term | None = field(default=None)
    quote_date: date | None = field(default=None)
    spot_lag: int = field(default=2)
    payment_lag: int = field(default=1)

    @abstractmethod
    def to_outright(self, spot: FX_Rate) -> float:
        """Outright forward price implied by the quote.

        Parameters
        ----------
        spot : FX_Rate
            Spot of the pair (or its inverse).

        Returns
        -------
        float
        """

    def _effective_fixing_date(self) -> date | None:
        """Explicit fixing date, else spot date + `term`; None for a deliverable forward."""
        if self.fixing_date is not None:
            return self.fixing_date
        if self.term is not None and self.quote_date is not None:
            cal = self.term.adj_convention.calendar
            spot_date = cal.add_business_days(self.quote_date, self.spot_lag)
            return self.term.advance(spot_date)
        return None

    def _effective_payment_date(self) -> date:
        """Explicit payment date, else fixing + `payment_lag` business days.

        Raises
        ------
        ValueError
            If neither a payment date nor a derivable fixing date is available.
        """
        if self.payment_date is not None:
            return self.payment_date
        fixing = self._effective_fixing_date()
        if fixing is not None:
            cal = self.term.adj_convention.calendar if self.term else get_calendar()
            return cal.add_business_days(fixing, self.payment_lag)
        raise ValueError(
            "Cannot derive payment_date: provide payment_date, or term+quote_date."
        )

    def _build(self, strike: float) -> Forward | NDF:
        """Forward instrument at `strike`.

        An NDF when a fixing date is set or derivable, or when the pair holds a
        non-deliverable currency (a UF forward on ``CLF/CLP``, fixing on the
        payment date by default); a deliverable Forward otherwise.
        """
        from ..derivatives.forwards.forwards import Forward, NDF

        fixing = self._effective_fixing_date()
        pmt = self._effective_payment_date()
        pair = self.currency_pair
        if fixing is None and not (
            pair.base_currency.is_deliverable and pair.quote_currency.is_deliverable
        ):
            fixing = pmt
        if fixing is not None:
            return NDF(
                notional=_NOTIONAL,
                strike=strike,
                payment_date=pmt,
                is_buy=self.is_buy,
                currency_pair=pair,
                fixing_date=fixing,
            )
        return Forward(
            notional=_NOTIONAL,
            strike=strike,
            payment_date=pmt,
            is_buy=self.is_buy,
            currency_pair=pair,
        )


# ── Private: swap base class ───────────────────────────────────────────────


class QuotedSide(Enum):
    RECEIVE = "RECEIVE"
    PAY = "PAY"


@dataclass
class _SwapQuote(InstrumentQuote):
    """Base of swap quotes.

    Parameters
    ----------
    quoted_side : QuotedSide
        Side of the quoted (fixed or first) leg: RECEIVE or PAY.
    term : Term
        Swap tenor.
    quote_date : date
        Quote date; the start is `spot_lag` business days later.
    adj_convention : AdjustmentDateConventionBase, optional
        Schedule adjustment; defaults to modified following, weekend calendar.
    locality : Locality, optional
        Market locality of the quote.
    spot_lag : int, default 2
        Business days from `quote_date` to the start (keyword-only).
    start_date : date, optional
        Explicit start, overriding `spot_lag` (keyword-only).
    maturity_date : date, optional
        Explicit maturity, at most 7 days from the `term` maturity (keyword-only).
    collateral_index : InterestIndex, optional
        CSA collateral index; None means uncollateralised (keyword-only).
    stub_first, long_stub : bool
        Stub placement, as in the leg builders (keyword-only).
    """

    quoted_side: QuotedSide
    term: Term
    quote_date: date
    adj_convention: AdjustmentDateConventionBase = field(
        default_factory=ModifiedFollowingConvention
    )
    locality: Locality | None = field(default=None)
    _: KW_ONLY
    spot_lag: int = field(default=2)
    start_date: date | None = field(
        default=None
    )  # explicit start; ignores spot_lag when set
    maturity_date: date | None = field(
        default=None
    )  # explicit maturity; validated against term
    collateral_index: InterestIndex | None = field(default=None)
    stub_first: bool = True
    long_stub: bool = False

    def _effective_start(self) -> date:
        """Explicit start, else `spot_lag` business days after `quote_date`."""
        if self.start_date is not None:
            return self.start_date
        cal = self.adj_convention.calendar
        return cal.add_business_days(self.quote_date, self.spot_lag)

    def _effective_maturity(self, start: date) -> date:
        """`start` advanced by `term`, or the explicit maturity if within 7 days of it.

        Raises
        ------
        ValueError
            If the explicit maturity is more than 7 days from the `term` maturity.
        """
        computed = self.term.advance(start)
        if self.maturity_date is not None:
            diff = abs((self.maturity_date - computed).days)
            if diff > 7:
                raise ValueError(
                    f"Explicit maturity_date {self.maturity_date} deviates from "
                    f"term {self.term} maturity {computed} by {diff} days (> 7)."
                )
            return self.maturity_date
        return computed


# ── Public API ─────────────────────────────────────────────────────────────


@dataclass
class ForwardPriceQuote(_ForwardQuote):
    """FX forward quoted as an outright price.

    Builds an NDF when a fixing date is set or derivable, or on a pair with a
    non-deliverable currency (a UF forward is a ``CLF/CLP`` NDF quoted as the
    outright UF level); a deliverable Forward otherwise.

    Parameters
    ----------
    currency_pair : CurrencyPair
        Pair quoted.
    value : float
        Outright forward price.
    is_buy : bool
        True to buy the base currency.
    payment_date, fixing_date : date, optional
        Explicit dates (keyword-only).
    locality : Locality, optional
        Market locality (keyword-only).
    term : Term, optional
        Tenor from the spot date (keyword-only).
    quote_date : date, optional
        Quote date (keyword-only).
    spot_lag : int, default 2
        Business days from `quote_date` to spot (keyword-only).
    payment_lag : int, default 1
        Business days from fixing to payment (keyword-only).
    """

    def to_outright(self, spot: FX_Rate) -> float:
        return self.value

    def get_instrument(self) -> Forward | NDF:
        return self._build(self.value)


@dataclass(kw_only=True)
class ForwardPointsQuote(_ForwardQuote):
    """FX forward quoted as points over spot.

    The outright strike is the market spot at `quote_date` plus the points, so it
    is consistent with the FX used to value the forward. Other parameters as in
    :class:`ForwardPriceQuote`.

    Parameters
    ----------
    points_divisor : int, default 1
        Divisor turning quoted points into price units.

    Raises
    ------
    ValueError
        If `points_divisor` is not positive.
    """

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
                f"Spot currency pair {spot.currency_pair} is incompatible with "
                f"forward currency pair {self.currency_pair}."
            )
        return effective_spot + self.value / self.points_divisor

    def get_instrument(self, market=None) -> Forward | NDF:
        # Outright strike = market spot + points, using the market FX at quote_date (the
        # same rate used to valuate the forward). Structure-only calls (no market) never
        # read the strike, so a 0 placeholder is fine there.
        strike = (
            self.to_outright(market.get_fx_rate(self.quote_date, self.currency_pair))
            if market is not None
            else 0.0
        )
        return self._build(strike)


@dataclass
class IRSQuote(_SwapQuote):
    """IRS or OIS quote: fixed vs floating leg in one currency.

    Parameters
    ----------
    quoted_side : QuotedSide
        Side of the quoted (fixed or first) leg: RECEIVE or PAY.
    term : Term
        Swap tenor.
    quote_date : date
        Quote date; the start is `spot_lag` business days later.
    adj_convention : AdjustmentDateConventionBase, optional
        Schedule adjustment; defaults to modified following, weekend calendar.
    locality : Locality, optional
        Market locality of the quote.
    spot_lag : int, default 2
        Business days from `quote_date` to the start (keyword-only).
    start_date : date, optional
        Explicit start, overriding `spot_lag` (keyword-only).
    maturity_date : date, optional
        Explicit maturity, at most 7 days from the `term` maturity (keyword-only).
    collateral_index : InterestIndex, optional
        CSA collateral index; None means uncollateralised (keyword-only).
    stub_first, long_stub : bool
        Stub placement, as in the leg builders (keyword-only).
    fixed_leg : FixedLegSpec
        Fixed leg (keyword-only); `quoted_side` refers to it.
    floating_leg : FloatingLegSpec
        Floating leg (keyword-only).

    Raises
    ------
    ValueError
        If the legs are in different currencies.
    """

    _: KW_ONLY
    fixed_leg: FixedLegSpec
    floating_leg: FloatingLegSpec

    def __post_init__(self):
        if self.fixed_leg.currency != self.floating_leg.currency:
            raise ValueError(
                f"IRS requires both legs in the same currency. "
                f"Got {self.fixed_leg.currency} and {self.floating_leg.currency}."
            )

    def get_instrument(self) -> Swap:
        from ..derivatives.swaps.swaps import Swap
        from ..derivatives.swaps.builders import fixed_leg, floating_leg

        start = self._effective_start()
        mat = self._effective_maturity(start)
        fixed = fixed_leg(
            notional=_NOTIONAL,
            start_date=start,
            term=self.term,
            frequency=self.fixed_leg.payment_frequency.value,
            adj_convention=self.adj_convention,
            rate=self.fixed_leg.rate,
            currency=self.fixed_leg.currency,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
            maturity_date=mat,
            payment_lag=self.fixed_leg.payment_lag,
        )
        floating = floating_leg(
            notional=_NOTIONAL,
            start_date=start,
            term=self.term,
            frequency=self.floating_leg.payment_frequency.value,
            adj_convention=self.adj_convention,
            index=self.floating_leg.index,
            spread=self.floating_leg.spread,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
            maturity_date=mat,
            payment_lag=self.floating_leg.payment_lag,
        )
        if self.quoted_side == QuotedSide.RECEIVE:
            return Swap(
                receive_leg=fixed,
                pay_leg=floating,
                payment_currency=self.fixed_leg.currency,
                collateral_index=self.collateral_index,
            )
        return Swap(
            receive_leg=floating,
            pay_leg=fixed,
            payment_currency=self.fixed_leg.currency,
            collateral_index=self.collateral_index,
        )


@dataclass
class IRBasisQuote(_SwapQuote):
    """Basis swap quote: float vs float in one currency.

    Parameters
    ----------
    quoted_side : QuotedSide
        Side of the quoted (fixed or first) leg: RECEIVE or PAY.
    term : Term
        Swap tenor.
    quote_date : date
        Quote date; the start is `spot_lag` business days later.
    adj_convention : AdjustmentDateConventionBase, optional
        Schedule adjustment; defaults to modified following, weekend calendar.
    locality : Locality, optional
        Market locality of the quote.
    spot_lag : int, default 2
        Business days from `quote_date` to the start (keyword-only).
    start_date : date, optional
        Explicit start, overriding `spot_lag` (keyword-only).
    maturity_date : date, optional
        Explicit maturity, at most 7 days from the `term` maturity (keyword-only).
    collateral_index : InterestIndex, optional
        CSA collateral index; None means uncollateralised (keyword-only).
    stub_first, long_stub : bool
        Stub placement, as in the leg builders (keyword-only).
    receive_leg, pay_leg : FloatingLegSpec
        Floating legs (keyword-only).

    Raises
    ------
    ValueError
        If the legs are in different currencies.
    """

    _: KW_ONLY
    receive_leg: FloatingLegSpec
    pay_leg: FloatingLegSpec

    def __post_init__(self):
        if self.receive_leg.currency != self.pay_leg.currency:
            raise ValueError(
                f"IR Basis requires both legs in the same currency. "
                f"Got {self.receive_leg.currency} and {self.pay_leg.currency}."
            )

    def get_instrument(self) -> Swap:
        from ..derivatives.swaps.swaps import Swap
        from ..derivatives.swaps.builders import floating_leg

        start = self._effective_start()
        mat = self._effective_maturity(start)

        def _build(spec: FloatingLegSpec):
            return floating_leg(
                notional=_NOTIONAL,
                start_date=start,
                term=self.term,
                frequency=spec.payment_frequency.value,
                adj_convention=self.adj_convention,
                index=spec.index,
                spread=spec.spread,
                stub_first=self.stub_first,
                long_stub=self.long_stub,
                maturity_date=mat,
                payment_lag=spec.payment_lag,
            )

        return Swap(
            receive_leg=_build(self.receive_leg),
            pay_leg=_build(self.pay_leg),
            payment_currency=self.receive_leg.currency,
            collateral_index=self.collateral_index,
        )


@dataclass
class CrossCurrencyFixedFloatQuote(_SwapQuote):
    """Deliverable cross-currency swap quote: fixed vs floating.

    Parameters
    ----------
    quoted_side : QuotedSide
        Side of the quoted (fixed or first) leg: RECEIVE or PAY.
    term : Term
        Swap tenor.
    quote_date : date
        Quote date; the start is `spot_lag` business days later.
    adj_convention : AdjustmentDateConventionBase, optional
        Schedule adjustment; defaults to modified following, weekend calendar.
    locality : Locality, optional
        Market locality of the quote.
    spot_lag : int, default 2
        Business days from `quote_date` to the start (keyword-only).
    start_date : date, optional
        Explicit start, overriding `spot_lag` (keyword-only).
    maturity_date : date, optional
        Explicit maturity, at most 7 days from the `term` maturity (keyword-only).
    collateral_index : InterestIndex, optional
        CSA collateral index; None means uncollateralised (keyword-only).
    stub_first, long_stub : bool
        Stub placement, as in the leg builders (keyword-only).
    fixed_leg : FixedLegSpec
        Fixed leg (keyword-only); `quoted_side` refers to it.
    floating_leg : FloatingLegSpec
        Floating leg (keyword-only).
    is_deliverable : bool, default True
        True when each leg settles in its own currency (keyword-only).
    payment_currency : Currency, optional
        Settlement currency of a non-deliverable swap; must be None when
        deliverable (keyword-only).
    fx_fixing_lag : int, default 0
        Business days between each FX fixing and its payment date, for a
        non-deliverable swap (keyword-only).

    Raises
    ------
    ValueError
        If the legs are in the same currency, or the settlement is inconsistent.
    """

    _: KW_ONLY
    fixed_leg: FixedLegSpec
    floating_leg: FloatingLegSpec
    is_deliverable: bool = True
    payment_currency: Currency | None = None
    fx_fixing_lag: int = 0

    def __post_init__(self):
        if self.fixed_leg.currency == self.floating_leg.currency:
            raise ValueError(
                "CrossCurrencyFixedFloat requires legs in different currencies."
            )
        _check_settlement(self.is_deliverable, self.payment_currency)

    def get_instrument(self) -> Swap:
        from ..derivatives.swaps.swaps import Swap
        from ..derivatives.swaps.builders import fixed_leg, floating_leg

        start = self._effective_start()
        mat = self._effective_maturity(start)
        fixed = fixed_leg(
            notional=_NOTIONAL,
            start_date=start,
            term=self.term,
            frequency=self.fixed_leg.payment_frequency.value,
            adj_convention=self.adj_convention,
            rate=self.fixed_leg.rate,
            currency=self.fixed_leg.currency,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
            maturity_date=mat,
            payment_lag=self.fixed_leg.payment_lag,
            fx_fixing_lag=self.fx_fixing_lag,
        )
        floating = floating_leg(
            notional=_NOTIONAL,
            start_date=start,
            term=self.term,
            frequency=self.floating_leg.payment_frequency.value,
            adj_convention=self.adj_convention,
            index=self.floating_leg.index,
            spread=self.floating_leg.spread,
            cross_currency=True,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
            maturity_date=mat,
            payment_lag=self.floating_leg.payment_lag,
            fx_fixing_lag=self.fx_fixing_lag,
        )
        receive, pay = (
            (fixed, floating)
            if self.quoted_side == QuotedSide.RECEIVE
            else (floating, fixed)
        )
        return Swap(
            receive_leg=receive,
            pay_leg=pay,
            collateral_index=self.collateral_index,
            is_deliverable=self.is_deliverable,
            payment_currency=self.payment_currency,
        )


@dataclass
class CrossCurrencyFloatFloatQuote(_SwapQuote):
    """Deliverable cross-currency swap quote: float vs float.

    Given a market, the receive notional is the pay notional converted at the FX
    on `quote_date`.

    Parameters
    ----------
    quoted_side : QuotedSide
        Side of the quoted (fixed or first) leg: RECEIVE or PAY.
    term : Term
        Swap tenor.
    quote_date : date
        Quote date; the start is `spot_lag` business days later.
    adj_convention : AdjustmentDateConventionBase, optional
        Schedule adjustment; defaults to modified following, weekend calendar.
    locality : Locality, optional
        Market locality of the quote.
    spot_lag : int, default 2
        Business days from `quote_date` to the start (keyword-only).
    start_date : date, optional
        Explicit start, overriding `spot_lag` (keyword-only).
    maturity_date : date, optional
        Explicit maturity, at most 7 days from the `term` maturity (keyword-only).
    collateral_index : InterestIndex, optional
        CSA collateral index; None means uncollateralised (keyword-only).
    stub_first, long_stub : bool
        Stub placement, as in the leg builders (keyword-only).
    receive_leg, pay_leg : FloatingLegSpec
        Floating legs (keyword-only).
    is_deliverable : bool, default True
        True when each leg settles in its own currency (keyword-only).
    payment_currency : Currency, optional
        Settlement currency of a non-deliverable swap; must be None when
        deliverable (keyword-only).
    fx_fixing_lag : int, default 0
        Business days between each FX fixing and its payment date, for a
        non-deliverable swap (keyword-only).

    Raises
    ------
    ValueError
        If the legs are in the same currency, or the settlement is inconsistent.
    """

    _: KW_ONLY
    receive_leg: FloatingLegSpec
    pay_leg: FloatingLegSpec
    is_deliverable: bool = True
    payment_currency: Currency | None = None
    fx_fixing_lag: int = 0

    def __post_init__(self):
        if self.receive_leg.currency == self.pay_leg.currency:
            raise ValueError(
                "CrossCurrencyFloatFloat requires legs in different currencies."
            )
        _check_settlement(self.is_deliverable, self.payment_currency)

    def get_instrument(self, market=None) -> Swap:
        from ..derivatives.swaps.swaps import Swap
        from ..derivatives.swaps.builders import floating_leg
        from .currencies import CurrencyPair

        start = self._effective_start()
        mat = self._effective_maturity(start)

        # Size the two legs so their notionals match at the market FX — the same rate used
        # to valuate the swap: N_receive = N_pay * FX(pay_ccy -> receive_ccy). Without a
        # market (structure-only calls that just read currencies/dates) both legs use the
        # base notional.
        recv_notional = pay_notional = _NOTIONAL
        if market is not None:
            fx = market.get_fx_rate(
                self.quote_date,
                CurrencyPair(self.pay_leg.currency, self.receive_leg.currency),
            ).value
            recv_notional = pay_notional * fx

        def _build(spec: FloatingLegSpec, notional: float):
            return floating_leg(
                notional=notional,
                start_date=start,
                term=self.term,
                frequency=spec.payment_frequency.value,
                adj_convention=self.adj_convention,
                index=spec.index,
                spread=spec.spread,
                cross_currency=True,
                stub_first=self.stub_first,
                long_stub=self.long_stub,
                maturity_date=mat,
                payment_lag=spec.payment_lag,
                fx_fixing_lag=self.fx_fixing_lag,
            )

        return Swap(
            receive_leg=_build(self.receive_leg, recv_notional),
            pay_leg=_build(self.pay_leg, pay_notional),
            collateral_index=self.collateral_index,
            is_deliverable=self.is_deliverable,
            payment_currency=self.payment_currency,
        )
