from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import cast

import numpy as np

from .coupons import FixedCoupon, OvernightCoupon, TermRateCoupon, XCCYCoupon
from ...dates import AdjustmentDateConventionBase, DayCountConventionBase, ScheduleGenerator
from ...dates.term import Term
from ...market.currencies import Currency
from ...market.index import Index, InterestIndex


# Maps PaymentFrequency enum value to ScheduleGenerator frequency string.
_FREQ_TO_TENOR: dict[str, str] = {
    "MONTHLY": "1m",
    "QUARTERLY": "3m",
    "SEMIANNUAL": "6m",
    "ANNUAL": "1y",
}


@dataclass
class SwapLeg:
    """Base leg. Wraps a list of coupons and caches numpy arrays for performance."""
    coupons: list

    residuals: np.ndarray = field(init=False)
    amortizations: np.ndarray = field(init=False)
    time_fractions: np.ndarray = field(init=False)
    start_dates: list[date] = field(init=False)
    end_dates: list[date] = field(init=False)
    payment_dates: list[date] = field(init=False)

    def __post_init__(self):
        self.residuals = np.array([c.residual for c in self.coupons])
        self.amortizations = np.array([c.amortization for c in self.coupons])
        self.time_fractions = np.array([c.time_fraction for c in self.coupons])
        self.start_dates = [c.start_date for c in self.coupons]
        self.end_dates = [c.end_date for c in self.coupons]
        self.payment_dates = [c.payment_date for c in self.coupons]

    @property
    def n_coupons(self) -> int:
        return len(self.coupons)

    @staticmethod
    def _build_schedule(
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        stub_first: bool,
        long_stub: bool,
        maturity_date: date | None = None,
    ) -> list[date]:
        freq = _FREQ_TO_TENOR.get(payment_frequency)
        if freq is None:
            raise ValueError(
                f"Unsupported payment frequency '{payment_frequency}'. "
                f"Supported values: {list(_FREQ_TO_TENOR)}"
            )
        return ScheduleGenerator.generate_schedule(
            start_date=start_date,
            maturity_tenor=str(term),
            frequency_tenor=freq,
            adj_conv=adj_convention,
            maturity_adj_conv=term.adj_convention,
            end_date=maturity_date,
            stub_first=stub_first,
            long_stub=long_stub,
        )

    @staticmethod
    def _tf(
        day_count: type[DayCountConventionBase],
        start: date,
        end: date,
        base: int,
    ) -> float:
        return float(day_count.get_time_fraction(start, end, base))


@dataclass
class FixedLeg(SwapLeg):
    """Fixed-rate leg. Flows are precomputed at construction."""
    currency: Currency
    rates: np.ndarray = field(init=False)
    flows: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.rates = np.array([c.rate for c in self.coupons])
        self.flows = np.array([c.flow for c in self.coupons])

    @classmethod
    def from_term(
        cls,
        notional: float,
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        day_count_convention: type[DayCountConventionBase],
        year_fraction_base: int,
        rate: float,
        currency: Currency,
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
    ) -> FixedLeg:
        schedule = cls._build_schedule(
            start_date, term, payment_frequency,
            adj_convention, stub_first, long_stub, maturity_date,
        )
        n = len(schedule) - 1
        coupons = [
            FixedCoupon(
                residual=notional,
                amortization=0.0,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                time_fraction=cls._tf(day_count_convention, schedule[i], schedule[i + 1], year_fraction_base),
                rate=rate,
            )
            for i in range(n)
        ]
        return cls(coupons=coupons, currency=currency)


@dataclass
class FloatingLeg(SwapLeg):
    """Floating-rate leg. Flow amounts are projected at valuation time using the index curve."""
    index: InterestIndex
    spread_bps: float = 0.0
    spreads: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.spreads = (self.spread_bps / 10_000) * self.time_fractions * self.residuals

    @property
    def currency(self) -> Currency:
        # All concrete InterestIndex types also extend Index, which holds currency.
        return cast(Index, self.index).currency  # type: ignore[return-value]


@dataclass
class TermRateLeg(FloatingLeg):
    """Floating leg for a term-rate (IBOR-style) index. Stores per-coupon fixing dates."""
    fixing_lag: int = 0
    fixing_dates: list[date] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.fixing_dates = [c.fixing_date for c in self.coupons]  # type: ignore[union-attr]

    @classmethod
    def from_term(
        cls,
        notional: float,
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        day_count_convention: type[DayCountConventionBase],
        year_fraction_base: int,
        index: InterestIndex,
        spread_bps: float = 0.0,
        fixing_lag: int = 0,
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
    ) -> TermRateLeg:
        schedule = cls._build_schedule(
            start_date, term, payment_frequency,
            adj_convention, stub_first, long_stub, maturity_date,
        )
        calendar = adj_convention.calendar
        n = len(schedule) - 1
        coupons = [
            TermRateCoupon(
                residual=notional,
                amortization=0.0,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                time_fraction=cls._tf(day_count_convention, schedule[i], schedule[i + 1], year_fraction_base),
                spread_bps=spread_bps,
                fixing_date=calendar.add_business_days(schedule[i], -fixing_lag),
            )
            for i in range(n)
        ]
        return cls(coupons=coupons, index=index, spread_bps=spread_bps, fixing_lag=fixing_lag)


@dataclass
class OvernightLeg(FloatingLeg):
    """Floating leg for an overnight compounding (OIS-style) index."""

    @classmethod
    def from_term(
        cls,
        notional: float,
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        day_count_convention: type[DayCountConventionBase],
        year_fraction_base: int,
        index: InterestIndex,
        spread_bps: float = 0.0,
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
    ) -> OvernightLeg:
        schedule = cls._build_schedule(
            start_date, term, payment_frequency,
            adj_convention, stub_first, long_stub, maturity_date,
        )
        n = len(schedule) - 1
        coupons = [
            OvernightCoupon(
                residual=notional,
                amortization=0.0,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                time_fraction=cls._tf(day_count_convention, schedule[i], schedule[i + 1], year_fraction_base),
                spread_bps=spread_bps,
            )
            for i in range(n)
        ]
        return cls(coupons=coupons, index=index, spread_bps=spread_bps)


@dataclass
class XCCYFloatingLeg(FloatingLeg):
    """Floating leg for a cross-currency swap. Each coupon carries an FX fixing date."""
    fx_fixing_dates: list[date] | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.fx_fixing_dates is None:
            self.fx_fixing_dates = list(self.end_dates)

    @classmethod
    def from_term(
        cls,
        notional: float,
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        day_count_convention: type[DayCountConventionBase],
        year_fraction_base: int,
        index: InterestIndex,
        spread_bps: float = 0.0,
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
    ) -> XCCYFloatingLeg:
        schedule = cls._build_schedule(
            start_date, term, payment_frequency,
            adj_convention, stub_first, long_stub, maturity_date,
        )
        n = len(schedule) - 1
        coupons = [
            XCCYCoupon(
                residual=notional,
                amortization=0.0,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                time_fraction=cls._tf(day_count_convention, schedule[i], schedule[i + 1], year_fraction_base),
                spread_bps=spread_bps,
                fx_fixing_date=schedule[i + 1],
            )
            for i in range(n)
        ]
        return cls(coupons=coupons, index=index, spread_bps=spread_bps)
