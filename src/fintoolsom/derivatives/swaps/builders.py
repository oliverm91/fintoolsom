from __future__ import annotations

from datetime import date

from .legs import FixedLeg, OvernightLeg, TermRateLeg, XCCYFloatingLeg
from ...dates import AdjustmentDateConventionBase
from ...dates.term import Term
from ...rates import Rate
from ...market.currencies import Currency
from ...market.index import InterestIndex


def fixed_leg(
    notional: float,
    start_date: date,
    term: Term,
    frequency: str,
    adj_convention: AdjustmentDateConventionBase,
    rate: Rate,
    currency: Currency,
    *,
    stub_first: bool = True,
    long_stub: bool = False,
    maturity_date: date | None = None,
) -> FixedLeg:
    return FixedLeg.from_term(
        notional=notional,
        start_date=start_date,
        term=term,
        payment_frequency=frequency,
        adj_convention=adj_convention,
        rate=rate,
        currency=currency,
        stub_first=stub_first,
        long_stub=long_stub,
        maturity_date=maturity_date,
    )


def term_rate_leg(
    notional: float,
    start_date: date,
    term: Term,
    frequency: str,
    adj_convention: AdjustmentDateConventionBase,
    index: InterestIndex,
    *,
    spread: Rate = 0.0,
    fixing_lag: int = 0,
    stub_first: bool = True,
    long_stub: bool = False,
    maturity_date: date | None = None,
) -> TermRateLeg:
    return TermRateLeg.from_term(
        notional=notional,
        start_date=start_date,
        term=term,
        payment_frequency=frequency,
        adj_convention=adj_convention,
        index=index,
        spread=spread,
        fixing_lag=fixing_lag,
        stub_first=stub_first,
        long_stub=long_stub,
        maturity_date=maturity_date,
    )


def overnight_leg(
    notional: float,
    start_date: date,
    term: Term,
    frequency: str,
    adj_convention: AdjustmentDateConventionBase,
    index: InterestIndex,
    *,
    spread: Rate = 0.0,
    stub_first: bool = True,
    long_stub: bool = False,
    maturity_date: date | None = None,
) -> OvernightLeg:
    return OvernightLeg.from_term(
        notional=notional,
        start_date=start_date,
        term=term,
        payment_frequency=frequency,
        adj_convention=adj_convention,
        index=index,
        spread=spread,
        stub_first=stub_first,
        long_stub=long_stub,
        maturity_date=maturity_date,
    )


def xccy_floating_leg(
    notional: float,
    start_date: date,
    term: Term,
    frequency: str,
    adj_convention: AdjustmentDateConventionBase,
    index: InterestIndex,
    *,
    spread: Rate = 0.0,
    stub_first: bool = True,
    long_stub: bool = False,
    maturity_date: date | None = None,
) -> XCCYFloatingLeg:
    return XCCYFloatingLeg.from_term(
        notional=notional,
        start_date=start_date,
        term=term,
        payment_frequency=frequency,
        adj_convention=adj_convention,
        index=index,
        spread=spread,
        stub_first=stub_first,
        long_stub=long_stub,
        maturity_date=maturity_date,
    )
