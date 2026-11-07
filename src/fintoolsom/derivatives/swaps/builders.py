"""Convenience constructors for swap legs.

Every builder makes a bullet leg: constant notional, repaid with the final coupon
(the final notional exchange; no initial exchange is modelled). Common parameters:

notional : float
    Constant notional.
start_date : date
    First accrual start.
term : Term
    Leg tenor; its adjustment convention adjusts the maturity.
frequency : str
    MONTHLY, QUARTERLY, SEMIANNUAL or ANNUAL.
adj_convention : AdjustmentDateConventionBase
    Business-day adjustment of the schedule dates.
stub_first : bool, default True
    Place an irregular period at the start (True) or at the end (keyword-only).
long_stub : bool, default False
    Merge the irregular period into its neighbour (keyword-only).
maturity_date : date, optional
    Explicit maturity overriding `term` (keyword-only).
fx_fixing_lag : int, default 0
    Business days between each FX fixing and its payment date; used only when the
    leg settles non-deliverably in another currency (keyword-only).
"""
from __future__ import annotations

from datetime import date

from .legs import FixedLeg, OvernightLeg, TermRateLeg, XCCYFloatingLeg
from ...dates import AdjustmentDateConventionBase
from ...dates.term import Term
from ...rates import Rate
from ...market.currencies import Currency
from ...market.index import InterestIndex, OvernightIndex


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
    fx_fixing_lag: int = 0,
) -> FixedLeg:
    """Fixed-rate leg.

    Parameters
    ----------
    notional, start_date, term, frequency, adj_convention
        See the module docstring.
    rate : Rate
        Fixed rate of every coupon.
    currency : Currency
        Currency of the flows.
    stub_first, long_stub, maturity_date, fx_fixing_lag
        See the module docstring.

    Returns
    -------
    FixedLeg
    """
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
        fx_fixing_lag=fx_fixing_lag,
    )


def term_rate_leg(
    notional: float,
    start_date: date,
    term: Term,
    frequency: str,
    adj_convention: AdjustmentDateConventionBase,
    index: InterestIndex,
    *,
    spread: Rate | None = None,
    fixing_lag: int | None = None,
    stub_first: bool = True,
    long_stub: bool = False,
    maturity_date: date | None = None,
    fx_fixing_lag: int = 0,
) -> TermRateLeg:
    """Floating leg on a term-rate index: one fixing per coupon.

    Parameters
    ----------
    notional, start_date, term, frequency, adj_convention
        See the module docstring.
    index : InterestIndex
        Term-rate index; its currency is the leg currency.
    spread : Rate, optional
        Spread over the index; zero when omitted.
    fixing_lag : int, optional
        Business days between each rate fixing and its accrual start; defaults to
        the index ``spot_lag``.
    stub_first, long_stub, maturity_date, fx_fixing_lag
        See the module docstring.

    Returns
    -------
    TermRateLeg
    """
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
        fx_fixing_lag=fx_fixing_lag,
    )


def overnight_leg(
    notional: float,
    start_date: date,
    term: Term,
    frequency: str,
    adj_convention: AdjustmentDateConventionBase,
    index: InterestIndex,
    *,
    spread: Rate | None = None,
    stub_first: bool = True,
    long_stub: bool = False,
    maturity_date: date | None = None,
    fx_fixing_lag: int = 0,
) -> OvernightLeg:
    """Floating leg on an overnight index, compounded daily.

    Parameters
    ----------
    notional, start_date, term, frequency, adj_convention
        See the module docstring.
    index : InterestIndex
        Overnight rate or interest price index; its currency is the leg currency.
    spread : Rate, optional
        Spread over the index; zero when omitted.
    stub_first, long_stub, maturity_date, fx_fixing_lag
        See the module docstring.

    Returns
    -------
    OvernightLeg
    """
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
        fx_fixing_lag=fx_fixing_lag,
    )


def xccy_floating_leg(
    notional: float,
    start_date: date,
    term: Term,
    frequency: str,
    adj_convention: AdjustmentDateConventionBase,
    index: InterestIndex,
    *,
    spread: Rate | None = None,
    stub_first: bool = True,
    long_stub: bool = False,
    maturity_date: date | None = None,
    fx_fixing_lag: int = 0,
) -> XCCYFloatingLeg:
    """Cross-currency floating leg on an overnight index, compounded daily.

    Parameters
    ----------
    notional, start_date, term, frequency, adj_convention
        See the module docstring.
    index : InterestIndex
        Overnight index; its currency is the leg currency.
    spread : Rate, optional
        Spread over the index; zero when omitted.
    stub_first, long_stub, maturity_date, fx_fixing_lag
        See the module docstring.

    Returns
    -------
    XCCYFloatingLeg
    """
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
        fx_fixing_lag=fx_fixing_lag,
    )


def floating_leg(
    notional: float,
    start_date: date,
    term: Term,
    frequency: str,
    adj_convention: AdjustmentDateConventionBase,
    index: InterestIndex,
    *,
    spread: Rate | None = None,
    cross_currency: bool = False,
    stub_first: bool = True,
    long_stub: bool = False,
    maturity_date: date | None = None,
    fx_fixing_lag: int = 0,
) -> TermRateLeg | OvernightLeg:
    """Floating leg of the kind the index calls for.

    An overnight index (daily compounding) gives an :class:`OvernightLeg`, or an
    :class:`XCCYFloatingLeg` when `cross_currency`; any other index gives a
    :class:`TermRateLeg` fixing ``index.spot_lag`` business days before each start.

    Parameters
    ----------
    notional, start_date, term, frequency, adj_convention
        See the module docstring.
    index : InterestIndex
        Floating index; its currency is the leg currency.
    spread : Rate, optional
        Spread over the index; zero when omitted.
    cross_currency : bool, default False
        Build an :class:`XCCYFloatingLeg` for an overnight index.
    stub_first, long_stub, maturity_date, fx_fixing_lag
        See the module docstring.

    Returns
    -------
    TermRateLeg or OvernightLeg
    """
    common = dict(
        notional=notional,
        start_date=start_date,
        term=term,
        frequency=frequency,
        adj_convention=adj_convention,
        index=index,
        spread=spread,
        stub_first=stub_first,
        long_stub=long_stub,
        maturity_date=maturity_date,
        fx_fixing_lag=fx_fixing_lag,
    )
    if not isinstance(index, OvernightIndex):
        return term_rate_leg(**common)
    return xccy_floating_leg(**common) if cross_currency else overnight_leg(**common)
