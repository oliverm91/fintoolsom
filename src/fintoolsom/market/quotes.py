from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, KW_ONLY
from datetime import date
from enum import Enum

from .conventions import (
    PaymentFrequency,
    BasisPoints,
    LegSpec,
    FixedLegSpec,
    FloatingLegSpec,
)
from .currencies import Currency, CurrencyPair, FX_Rate
from .localities import Locality
from .index import InterestIndex


# ── InstrumentQuote ABC ────────────────────────────────────────────────────

class InstrumentQuote(ABC):
    """Mixin for quotes that describe a fully structured instrument.

    Implementors must build and return the instrument (notional=100) using
    only the information stored on the quote itself."""

    @abstractmethod
    def get_instrument(self) -> object:
        """Return the instrument described by this quote (notional fixed at 100)."""


_NOTIONAL = 100.0


# ── Forward Quotes ─────────────────────────────────────────────────────────

@dataclass
class ForwardQuote(InstrumentQuote):
    """Base for FX forward quotes. Inherits InstrumentQuote to enforce get_instrument()."""
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
class ForwardPriceQuote(ForwardQuote):
    """Forward quoted as a final outright price. Returns NDF if fixing_date is set."""

    def to_outright(self, spot: FX_Rate) -> float:
        return self.value

    def get_instrument(self) -> object:
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
class ForwardPointsQuote(ForwardQuote):
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

    def get_instrument(self) -> object:
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


# ── UF Forward Quotes ──────────────────────────────────────────────────────

@dataclass
class ForwardUFQuote(InstrumentQuote):
    """Base for Chilean UF forward quotes. UF forwards are always NDFs (fixing_date required)."""
    value: float
    payment_date: date
    is_buy: bool
    fixing_date: date
    locality: Locality = field(default=Locality.CL)

    @abstractmethod
    def to_outright(self, spot_uf: float) -> float:
        ...


@dataclass
class ForwardUFPriceQuote(ForwardUFQuote):
    """UF forward quoted as the outright forward UF level (CLP per UF)."""

    def to_outright(self, spot_uf: float) -> float:
        return self.value

    def get_instrument(self) -> object:
        from ..derivatives.forwards.forwards import NDF
        return NDF(
            notional=_NOTIONAL, strike=self.value,
            payment_date=self.payment_date, is_buy=self.is_buy,
            fixing_date=self.fixing_date, is_uf_indexed=True,
        )


@dataclass(kw_only=True)
class ForwardUFPointsQuote(ForwardUFQuote):
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

    def get_instrument(self) -> object:
        from ..derivatives.forwards.forwards import NDF
        return NDF(
            notional=_NOTIONAL, strike=self.to_outright(self.spot_uf),
            payment_date=self.payment_date, is_buy=self.is_buy,
            fixing_date=self.fixing_date, is_uf_indexed=True,
        )


# ── Swap Quotes ────────────────────────────────────────────────────────────

class QuotedSide(Enum):
    RECEIVE = "RECEIVE"
    PAY = "PAY"


@dataclass
class SwapQuote(InstrumentQuote):
    """Base class for swap quotes. collateral_index=None means uncollateralised."""
    quoted_side: QuotedSide
    term: str           # maturity tenor, e.g. "5Y", "18M"
    effective_date: date
    locality: Locality | None = field(default=None)
    _: KW_ONLY
    collateral_index: InterestIndex | None = field(default=None)
    stub_first: bool = True
    long_stub: bool = False


@dataclass
class IRSQuote(SwapQuote):
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

    def get_instrument(self):
        from ..derivatives.swaps.swaps import IRS
        from ..derivatives.swaps.legs import FixedLeg, TermRateLeg

        fixed = FixedLeg.from_term(
            notional=_NOTIONAL,
            start_date=self.effective_date,
            maturity_tenor=self.term,
            payment_frequency=self.fixed_leg.payment_frequency.value,
            adj_convention=self.fixed_leg.adj_convention,
            day_count_convention=self.fixed_leg.day_count_convention,
            year_fraction_base=self.fixed_leg.year_fraction_base,
            rate=self.fixed_leg.rate.value,
            currency=self.fixed_leg.currency,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
        )
        spread_bps = self.floating_leg.spread.value if self.floating_leg.spread else 0.0
        floating = TermRateLeg.from_term(
            notional=_NOTIONAL,
            start_date=self.effective_date,
            maturity_tenor=self.term,
            payment_frequency=self.floating_leg.payment_frequency.value,
            adj_convention=self.floating_leg.adj_convention,
            day_count_convention=self.floating_leg.day_count_convention,
            year_fraction_base=self.floating_leg.year_fraction_base,
            index=self.floating_leg.index,
            spread_bps=spread_bps,
        )
        return IRS(
            fixed_leg=fixed,
            floating_leg=floating,
            payment_currency=self.fixed_leg.currency,
            collateral_index=self.collateral_index,
        )


@dataclass
class IRBasisQuote(SwapQuote):
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

    def get_instrument(self):
        from ..derivatives.swaps.swaps import IRBasis
        from ..derivatives.swaps.legs import TermRateLeg

        def _build(spec: FloatingLegSpec) -> TermRateLeg:
            return TermRateLeg.from_term(
                notional=_NOTIONAL,
                start_date=self.effective_date,
                maturity_tenor=self.term,
                payment_frequency=spec.payment_frequency.value,
                adj_convention=spec.adj_convention,
                day_count_convention=spec.day_count_convention,
                year_fraction_base=spec.year_fraction_base,
                index=spec.index,
                spread_bps=spec.spread.value if spec.spread else 0.0,
                stub_first=self.stub_first,
                long_stub=self.long_stub,
            )

        return IRBasis(
            leg_a=_build(self.receive_leg),
            leg_b=_build(self.pay_leg),
            payment_currency=self.receive_leg.currency,
            collateral_index=self.collateral_index,
        )


@dataclass
class CrossCurrencyFixedFloatQuote(SwapQuote):
    """Fixed vs floating cross-currency swap."""
    _: KW_ONLY
    fixed_leg: FixedLegSpec
    floating_leg: FloatingLegSpec

    def __post_init__(self):
        if self.fixed_leg.currency == self.floating_leg.currency:
            raise ValueError(
                "CrossCurrencyFixedFloat requires legs in different currencies."
            )

    def get_instrument(self):
        from ..derivatives.swaps.swaps import CrossCurrencyFixFloat
        from ..derivatives.swaps.legs import FixedLeg, XCCYFloatingLeg

        fixed = FixedLeg.from_term(
            notional=_NOTIONAL,
            start_date=self.effective_date,
            maturity_tenor=self.term,
            payment_frequency=self.fixed_leg.payment_frequency.value,
            adj_convention=self.fixed_leg.adj_convention,
            day_count_convention=self.fixed_leg.day_count_convention,
            year_fraction_base=self.fixed_leg.year_fraction_base,
            rate=self.fixed_leg.rate.value,
            currency=self.fixed_leg.currency,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
        )
        floating = XCCYFloatingLeg.from_term(
            notional=_NOTIONAL,
            start_date=self.effective_date,
            maturity_tenor=self.term,
            payment_frequency=self.floating_leg.payment_frequency.value,
            adj_convention=self.floating_leg.adj_convention,
            day_count_convention=self.floating_leg.day_count_convention,
            year_fraction_base=self.floating_leg.year_fraction_base,
            index=self.floating_leg.index,
            spread_bps=self.floating_leg.spread.value if self.floating_leg.spread else 0.0,
            stub_first=self.stub_first,
            long_stub=self.long_stub,
        )
        return CrossCurrencyFixFloat(
            fixed_leg=fixed,
            floating_leg=floating,
            payment_currency=self.fixed_leg.currency,
            collateral_index=self.collateral_index,
        )


@dataclass
class CrossCurrencyFloatFloatQuote(SwapQuote):
    """Float vs float cross-currency swap."""
    _: KW_ONLY
    receive_leg: FloatingLegSpec
    pay_leg: FloatingLegSpec

    def __post_init__(self):
        if self.receive_leg.currency == self.pay_leg.currency:
            raise ValueError(
                "CrossCurrencyFloatFloat requires legs in different currencies."
            )

    def get_instrument(self):
        from ..derivatives.swaps.swaps import CrossCurrencyBasis
        from ..derivatives.swaps.legs import XCCYFloatingLeg

        def _build(spec: FloatingLegSpec) -> XCCYFloatingLeg:
            return XCCYFloatingLeg.from_term(
                notional=_NOTIONAL,
                start_date=self.effective_date,
                maturity_tenor=self.term,
                payment_frequency=spec.payment_frequency.value,
                adj_convention=spec.adj_convention,
                day_count_convention=spec.day_count_convention,
                year_fraction_base=spec.year_fraction_base,
                index=spec.index,
                spread_bps=spec.spread.value if spec.spread else 0.0,
                stub_first=self.stub_first,
                long_stub=self.long_stub,
            )

        return CrossCurrencyBasis(
            leg_a=_build(self.receive_leg),
            leg_b=_build(self.pay_leg),
            payment_currency=self.receive_leg.currency,
            collateral_index=self.collateral_index,
        )
