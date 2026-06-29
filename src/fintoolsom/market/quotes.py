from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, KW_ONLY
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING

from .conventions import FixedLegSpec, FloatingLegSpec
from .currencies import CurrencyPair, FX_Rate
from .localities import Locality
from .index import InterestIndex
from ..dates import AdjustmentDateConventionBase
from ..dates.term import Term

if TYPE_CHECKING:
    from ..derivatives.forwards.forwards import Forward, NDF
    from ..derivatives.swaps.swaps import IRS, IRBasis, CrossCurrencyFixFloat, CrossCurrencyBasis


# ── Private: ABC and module constants ──────────────────────────────────────

class _InstrumentQuote(ABC):
    """Enforces get_instrument() on all concrete quote subclasses."""

    @abstractmethod
    def get_instrument(self) -> object:
        """Return the instrument described by this quote (notional fixed at 100)."""


_NOTIONAL = 100.0


# ── Private: forward base classes ──────────────────────────────────────────

@dataclass
class _ForwardQuote(_InstrumentQuote):
    currency_pair: CurrencyPair
    value: float
    payment_date: date
    is_buy: bool
    locality: Locality | None = field(default=None)
    fixing_date: date | None = field(default=None)  # set → NDF, None → deliverable Forward

    @abstractmethod
    def to_outright(self, spot: FX_Rate) -> float:
        pass


@dataclass
class _ForwardUFQuote(_InstrumentQuote):
    """UF forwards are always NDFs — fixing_date is required."""
    value: float
    payment_date: date
    is_buy: bool
    fixing_date: date
    locality: Locality = field(default=Locality.CL)

    @abstractmethod
    def to_outright(self, spot_uf: float) -> float:
        ...


# ── Private: swap base class ───────────────────────────────────────────────

class QuotedSide(Enum):
    RECEIVE = "RECEIVE"
    PAY = "PAY"


@dataclass
class _SwapQuote(_InstrumentQuote):
    """collateral_index=None means uncollateralised."""
    quoted_side: QuotedSide
    term: Term
    quote_date: date
    adj_convention: AdjustmentDateConventionBase
    locality: Locality | None = field(default=None)
    _: KW_ONLY
    spot_lag: int = field(default=2)
    start_date: date | None = field(default=None)    # explicit start; ignores spot_lag when set
    maturity_date: date | None = field(default=None) # explicit maturity; validated against term
    collateral_index: InterestIndex | None = field(default=None)
    stub_first: bool = True
    long_stub: bool = False

    def _effective_start(self) -> date:
        if self.start_date is not None:
            return self.start_date
        return self.adj_convention.calendar.add_business_days(self.quote_date, self.spot_lag)

    def _effective_maturity(self, start: date) -> date:
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
    """Forward quoted as a final outright price. Returns NDF if fixing_date is set."""

    def to_outright(self, spot: FX_Rate) -> float:
        return self.value

    def get_instrument(self) -> Forward | NDF:
        from ..derivatives.forwards.forwards import Forward, NDF
        if self.fixing_date is not None:
            return NDF(
                notional=_NOTIONAL, strike=self.value,
                payment_date=self.payment_date, is_buy=self.is_buy,
                currency_pair=self.currency_pair, fixing_date=self.fixing_date,
            )
        return Forward(
            notional=_NOTIONAL, strike=self.value,
            payment_date=self.payment_date, is_buy=self.is_buy,
            currency_pair=self.currency_pair,
        )


@dataclass(kw_only=True)
class ForwardPointsQuote(_ForwardQuote):
    """Forward quoted as points over spot. Requires spot to compute the outright strike."""
    spot: FX_Rate
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

    def get_instrument(self) -> Forward | NDF:
        from ..derivatives.forwards.forwards import Forward, NDF
        strike = self.to_outright(self.spot)
        if self.fixing_date is not None:
            return NDF(
                notional=_NOTIONAL, strike=strike,
                payment_date=self.payment_date, is_buy=self.is_buy,
                currency_pair=self.currency_pair, fixing_date=self.fixing_date,
            )
        return Forward(
            notional=_NOTIONAL, strike=strike,
            payment_date=self.payment_date, is_buy=self.is_buy,
            currency_pair=self.currency_pair,
        )


@dataclass
class ForwardUFPriceQuote(_ForwardUFQuote):
    """UF forward quoted as the outright forward UF level (CLP per UF)."""

    def to_outright(self, spot_uf: float) -> float:
        return self.value

    def get_instrument(self) -> NDF:
        from ..derivatives.forwards.forwards import NDF
        return NDF(
            notional=_NOTIONAL, strike=self.value,
            payment_date=self.payment_date, is_buy=self.is_buy,
            fixing_date=self.fixing_date, is_uf_indexed=True,
        )


@dataclass(kw_only=True)
class ForwardUFPointsQuote(_ForwardUFQuote):
    """UF forward quoted as points over spot UF. Requires spot_uf to compute the strike."""
    spot_uf: float
    points_divisor: int = field(default=1)

    def __post_init__(self):
        if self.points_divisor <= 0:
            raise ValueError(
                f"points_divisor must be a positive integer. Got {self.points_divisor}."
            )

    def to_outright(self, spot_uf: float) -> float:
        return spot_uf + self.value / self.points_divisor

    def get_instrument(self) -> NDF:
        from ..derivatives.forwards.forwards import NDF
        return NDF(
            notional=_NOTIONAL, strike=self.to_outright(self.spot_uf),
            payment_date=self.payment_date, is_buy=self.is_buy,
            fixing_date=self.fixing_date, is_uf_indexed=True,
        )


@dataclass
class IRSQuote(_SwapQuote):
    """IRS or OIS: fixed vs floating leg, same currency."""
    _: KW_ONLY
    fixed_leg: FixedLegSpec
    floating_leg: FloatingLegSpec

    def __post_init__(self):
        if self.fixed_leg.currency != self.floating_leg.currency:
            raise ValueError(
                f"IRS requires both legs in the same currency. "
                f"Got {self.fixed_leg.currency} and {self.floating_leg.currency}."
            )

    def get_instrument(self) -> IRS:
        from ..derivatives.swaps.swaps import IRS
        from ..derivatives.swaps.legs import FixedLeg, TermRateLeg

        start = self._effective_start()
        mat = self._effective_maturity(start)
        fixed = FixedLeg.from_term(
            notional=_NOTIONAL,
            start_date=start,
            term=self.term,
            payment_frequency=self.fixed_leg.payment_frequency.value,
            adj_convention=self.adj_convention,
            day_count_convention=self.fixed_leg.day_count_convention,
            year_fraction_base=self.fixed_leg.year_fraction_base,
            rate=self.fixed_leg.rate.value,
            currency=self.fixed_leg.currency,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
            maturity_date=mat,
        )
        spread_bps = self.floating_leg.spread.value if self.floating_leg.spread else 0.0
        floating = TermRateLeg.from_term(
            notional=_NOTIONAL,
            start_date=start,
            term=self.term,
            payment_frequency=self.floating_leg.payment_frequency.value,
            adj_convention=self.adj_convention,
            day_count_convention=self.floating_leg.day_count_convention,
            year_fraction_base=self.floating_leg.year_fraction_base,
            index=self.floating_leg.index,
            spread_bps=spread_bps,
            maturity_date=mat,
        )
        return IRS(
            fixed_leg=fixed,
            floating_leg=floating,
            payment_currency=self.fixed_leg.currency,
            collateral_index=self.collateral_index,
        )


@dataclass
class IRBasisQuote(_SwapQuote):
    """Float vs float basis swap, same currency."""
    _: KW_ONLY
    receive_leg: FloatingLegSpec
    pay_leg: FloatingLegSpec

    def __post_init__(self):
        if self.receive_leg.currency != self.pay_leg.currency:
            raise ValueError(
                f"IR Basis requires both legs in the same currency. "
                f"Got {self.receive_leg.currency} and {self.pay_leg.currency}."
            )

    def get_instrument(self) -> IRBasis:
        from ..derivatives.swaps.swaps import IRBasis
        from ..derivatives.swaps.legs import TermRateLeg

        start = self._effective_start()
        mat = self._effective_maturity(start)

        def _build(spec: FloatingLegSpec) -> TermRateLeg:
            return TermRateLeg.from_term(
                notional=_NOTIONAL,
                start_date=start,
                term=self.term,
                payment_frequency=spec.payment_frequency.value,
                adj_convention=self.adj_convention,
                day_count_convention=spec.day_count_convention,
                year_fraction_base=spec.year_fraction_base,
                index=spec.index,
                spread_bps=spec.spread.value if spec.spread else 0.0,
                stub_first=self.stub_first,
                long_stub=self.long_stub,
                maturity_date=mat,
            )

        return IRBasis(
            leg_a=_build(self.receive_leg),
            leg_b=_build(self.pay_leg),
            payment_currency=self.receive_leg.currency,
            collateral_index=self.collateral_index,
        )


@dataclass
class CrossCurrencyFixedFloatQuote(_SwapQuote):
    """Fixed vs floating cross-currency swap."""
    _: KW_ONLY
    fixed_leg: FixedLegSpec
    floating_leg: FloatingLegSpec

    def __post_init__(self):
        if self.fixed_leg.currency == self.floating_leg.currency:
            raise ValueError(
                "CrossCurrencyFixedFloat requires legs in different currencies."
            )

    def get_instrument(self) -> CrossCurrencyFixFloat:
        from ..derivatives.swaps.swaps import CrossCurrencyFixFloat
        from ..derivatives.swaps.legs import FixedLeg, XCCYFloatingLeg

        start = self._effective_start()
        mat = self._effective_maturity(start)
        fixed = FixedLeg.from_term(
            notional=_NOTIONAL,
            start_date=start,
            term=self.term,
            payment_frequency=self.fixed_leg.payment_frequency.value,
            adj_convention=self.adj_convention,
            day_count_convention=self.fixed_leg.day_count_convention,
            year_fraction_base=self.fixed_leg.year_fraction_base,
            rate=self.fixed_leg.rate.value,
            currency=self.fixed_leg.currency,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
            maturity_date=mat,
        )
        floating = XCCYFloatingLeg.from_term(
            notional=_NOTIONAL,
            start_date=start,
            term=self.term,
            payment_frequency=self.floating_leg.payment_frequency.value,
            adj_convention=self.adj_convention,
            day_count_convention=self.floating_leg.day_count_convention,
            year_fraction_base=self.floating_leg.year_fraction_base,
            index=self.floating_leg.index,
            spread_bps=self.floating_leg.spread.value if self.floating_leg.spread else 0.0,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
            maturity_date=mat,
        )
        return CrossCurrencyFixFloat(
            fixed_leg=fixed,
            floating_leg=floating,
            payment_currency=self.fixed_leg.currency,
            collateral_index=self.collateral_index,
        )


@dataclass
class CrossCurrencyFloatFloatQuote(_SwapQuote):
    """Float vs float cross-currency swap."""
    _: KW_ONLY
    receive_leg: FloatingLegSpec
    pay_leg: FloatingLegSpec

    def __post_init__(self):
        if self.receive_leg.currency == self.pay_leg.currency:
            raise ValueError(
                "CrossCurrencyFloatFloat requires legs in different currencies."
            )

    def get_instrument(self) -> CrossCurrencyBasis:
        from ..derivatives.swaps.swaps import CrossCurrencyBasis
        from ..derivatives.swaps.legs import XCCYFloatingLeg

        start = self._effective_start()
        mat = self._effective_maturity(start)

        def _build(spec: FloatingLegSpec) -> XCCYFloatingLeg:
            return XCCYFloatingLeg.from_term(
                notional=_NOTIONAL,
                start_date=start,
                term=self.term,
                payment_frequency=spec.payment_frequency.value,
                adj_convention=self.adj_convention,
                day_count_convention=spec.day_count_convention,
                year_fraction_base=spec.year_fraction_base,
                index=spec.index,
                spread_bps=spec.spread.value if spec.spread else 0.0,
                stub_first=self.stub_first,
                long_stub=self.long_stub,
                maturity_date=mat,
            )

        return CrossCurrencyBasis(
            leg_a=_build(self.receive_leg),
            leg_b=_build(self.pay_leg),
            payment_currency=self.receive_leg.currency,
            collateral_index=self.collateral_index,
        )
