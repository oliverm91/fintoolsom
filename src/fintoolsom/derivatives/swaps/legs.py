from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import cast

import numpy as np

from fintoolsom.rates.Rates import Rate, RateConvention

from .coupons import FixedCoupon, OvernightCoupon, TermRateCoupon, XCCYCoupon
from ...dates import AdjustmentDateConventionBase, ScheduleGenerator
from ...dates.term import Term
from ...dates.time_fractions import TimeFractionBase
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
    start_dates: list[date] = field(init=False)
    end_dates: list[date] = field(init=False)
    payment_dates: list[date] = field(init=False)

    def __post_init__(self):
        self.residuals = np.array([c.residual for c in self.coupons])
        self.amortizations = np.array([c.amortization for c in self.coupons])
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
        rate: Rate,
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
        maturity = max(schedule)
        coupons = [
            FixedCoupon(
                residual=notional,
                amortization=0.0 if schedule[i+1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                rate=rate,
            )
            for i in range(n)
        ]
        return cls(coupons=coupons, currency=currency)


@dataclass
class FloatingLeg(SwapLeg):
    """Floating-rate leg. Flow amounts are projected at valuation time using the index curve."""
    index: InterestIndex
    spread: Rate
    spreads_values: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.spreads_values = self.residuals * (self.spread.get_wealth_factor(self.start_dates, self.end_dates) - 1)
    @property
    def currency(self) -> Currency:
        # All concrete InterestIndex types also extend Index, which holds currency.
        return cast(Index, self.index).currency  # type: ignore[return-value]


@dataclass
class TermRateLeg(FloatingLeg):
    """Floating leg for a term-rate or price index.
    fixing_dates are explicit — they typically don't coincide with start_date."""
    fixing_dates: list[date] = field(default_factory=list)
    @classmethod
    def from_term(
        cls,
        notional: float,
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        index: InterestIndex,
        spread: Rate = Rate(RateConvention(), 0.0),
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
        fixing_dates = [calendar.add_business_days(schedule[i], -fixing_lag) for i in range(n)]
        maturity = max(schedule)
        coupons = [
            TermRateCoupon(
                residual=notional,
                amortization=0.0 if schedule[i+1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                spread=spread,
                fixing_date=fixing_dates[i],
            )
            for i in range(n)
        ]
        return cls(coupons=coupons, index=index, spread=spread, fixing_dates=fixing_dates)


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
        index: InterestIndex,
        spread: Rate = Rate(RateConvention(), 0.0),
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
    ) -> OvernightLeg:
        schedule = cls._build_schedule(
            start_date, term, payment_frequency,
            adj_convention, stub_first, long_stub, maturity_date,
        )
        n = len(schedule) - 1
        maturity = max(schedule)
        coupons = [
            OvernightCoupon(
                residual=notional,
                amortization=0.0 if schedule[i+1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                spread=spread,
            )
            for i in range(n)
        ]
        return cls(coupons=coupons, index=index, spread=spread)


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
        index: InterestIndex,
        spread: Rate = Rate(RateConvention(), 0.0),
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
    ) -> XCCYFloatingLeg:
        schedule = cls._build_schedule(
            start_date, term, payment_frequency,
            adj_convention, stub_first, long_stub, maturity_date,
        )
        n = len(schedule) - 1
        maturity = max(schedule)
        coupons = [
            XCCYCoupon(
                residual=notional,
                amortization=0.0 if schedule[i+1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                spread=spread,
                fx_fixing_date=schedule[i + 1],
            )
            for i in range(n)
        ]
        return cls(coupons=coupons, index=index, spread=spread)
