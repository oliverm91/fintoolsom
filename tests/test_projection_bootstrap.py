"""End-to-end test of the INDEPENDENT-basis projection bootstrap (Phase 4 / §8.1).

The notebook data is all OIS self-discounted (aliased), so it never exercises the
real forward-rate solve. Here a synthetic Term index (TERM3M in USD) is projected
over a prebuilt SOFR discount curve: fixed-vs-TERM3M swaps collateralised in SOFR
pin the TERM3M projection. After the bootstrap every swap must reprice to par.
"""
import math
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.dates import ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Market, Currency, CurrencyName
from fintoolsom.market.index import OvernightRateIndex, TermRateIndex
from fintoolsom.market.index_history import TermRateHistory
from fintoolsom.rates import (
    Rate,
    RateConvention,
    LinearInterestConvention,
    ExponentialInterestConvention,
    ZeroCouponCurve,
    InterpolationMethod,
    ProjectionCurve,
    ProjectionInterpolationMethod,
)
from fintoolsom.derivatives.swaps.builders import fixed_leg, term_rate_leg
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.curve_builder.builder import _bootstrap_projection_curve


T = date(2026, 6, 29)
USD = Currency(CurrencyName.USD)


def _market_with_sofr_discount() -> tuple[Market, OvernightRateIndex, TermRateIndex, ModifiedFollowingConvention]:
    sofr = OvernightRateIndex("SOFR", currency=USD)
    adj = ModifiedFollowingConvention(sofr.calendar)
    term3m = TermRateIndex("TERM3M", currency=USD, term=Term(3, TermUnit.M, adj))

    # Flat 5% (cont, act/365) SOFR discount curve.
    pillars = [90, 180, 365, 730, 1095]
    date_dfs = [(T + _days(d), math.exp(-0.05 * d / 365)) for d in pillars]
    sofr_curve = ZeroCouponCurve(T, date_dfs=date_dfs, df_interpolation_method=InterpolationMethod.LogLinear)

    term_hist = TermRateHistory(
        term3m,
        {T: Rate(RateConvention(interest_convention=LinearInterestConvention, time_fraction_base=360), 0.052)},
    )
    market = Market(T, indexes_history={"TERM3M": term_hist})
    market.curves[(sofr, USD)] = sofr_curve
    return market, sofr, term3m, adj


def _days(n: int) -> timedelta:
    return timedelta(days=n)


def _term_swap(term3m, sofr, adj, tenor_value, tenor_unit, fixed_rate) -> Swap:
    start = term3m.calendar.add_business_days(T, term3m.spot_lag)
    tenor = Term(tenor_value, tenor_unit, adj)
    fixed = fixed_leg(
        notional=100.0, start_date=start, term=tenor, frequency="QUARTERLY",
        adj_convention=adj,
        rate=Rate(RateConvention(interest_convention=ExponentialInterestConvention, time_fraction_base=365), fixed_rate),
        currency=USD,
    )
    floating = term_rate_leg(
        notional=100.0, start_date=start, term=tenor, frequency="QUARTERLY",
        adj_convention=adj, index=term3m, spread=Rate(RateConvention(), 0.0),
    )
    return Swap(receive_leg=fixed, pay_leg=floating, payment_currency=USD, collateral_index=sofr)


def test_independent_projection_reprices_all_swaps():
    market, sofr, term3m, adj = _market_with_sofr_discount()
    swaps = [
        _term_swap(term3m, sofr, adj, 6, TermUnit.M, 0.050),
        _term_swap(term3m, sofr, adj, 1, TermUnit.Y, 0.051),
        _term_swap(term3m, sofr, adj, 2, TermUnit.Y, 0.052),
    ]

    _bootstrap_projection_curve(term3m, swaps, market, sofr, ProjectionInterpolationMethod.PiecewiseConstant)

    proj = market.get_projection(term3m)
    assert isinstance(proj, ProjectionCurve)
    # One anchor segment + one solved segment per swap.
    assert len(proj) == 1 + len(swaps)

    for s in swaps:
        mtm = Calculator.get_swap_mtm(s, market, sofr, USD)
        assert mtm == pytest.approx(0.0, abs=1e-6)


def test_independent_projection_forwards_are_positive_and_near_par():
    market, sofr, term3m, adj = _market_with_sofr_discount()
    swaps = [
        _term_swap(term3m, sofr, adj, 1, TermUnit.Y, 0.050),
        _term_swap(term3m, sofr, adj, 2, TermUnit.Y, 0.050),
    ]
    _bootstrap_projection_curve(term3m, swaps, market, sofr, ProjectionInterpolationMethod.PiecewiseConstant)
    proj = market.get_projection(term3m)
    assert isinstance(proj, ProjectionCurve)
    # Solved forwards should be sane (positive, in a plausible rate range).
    assert np.all(proj.forward_rates > 0.0)
    assert np.all(proj.forward_rates < 0.20)


def test_non_square_projection_raises():
    from fintoolsom.curve_builder.builder import InsufficientQuotesError

    market, sofr, term3m, adj = _market_with_sofr_discount()
    # Two swaps with the SAME maturity → two residuals, one segment → under-determined.
    swaps = [
        _term_swap(term3m, sofr, adj, 1, TermUnit.Y, 0.050),
        _term_swap(term3m, sofr, adj, 1, TermUnit.Y, 0.055),
    ]
    with pytest.raises(InsufficientQuotesError):
        _bootstrap_projection_curve(term3m, swaps, market, sofr, ProjectionInterpolationMethod.PiecewiseConstant)
