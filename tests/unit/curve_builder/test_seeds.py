"""Seeds and short-end anchors (_seeds._Seeder)."""

import math
from collections import defaultdict
from datetime import date
from types import SimpleNamespace

import pytest

from fintoolsom.curve_builder import InsufficientQuotesError
from fintoolsom.curve_builder._seeds import _Seeder
from fintoolsom.dates import ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import (
    Currency,
    Market,
    OvernightInterestPriceIndex,
    OvernightRateIndex,
)
from fintoolsom.market.index import TermRateIndex
from fintoolsom.market.index_history import OvernightInterestPriceHistory
from fintoolsom.rates import IndexCurve, Rate, RateConvention

RISKLESS = OvernightRateIndex("OIS", currency=Currency.CLP)


def _seeder(market: Market) -> _Seeder:
    return _Seeder(market, market.risk_free_index)


def test_fixed_leg_seed_compounds_the_coupons_to_the_leg_end():
    t = date(2024, 1, 2)
    starts = [date(2024, 1, 4), date(2024, 7, 4)]
    ends = [date(2024, 7, 4), date(2025, 1, 6)]
    rate = Rate(RateConvention(), 0.05)  # annual compounding, act/365
    leg = SimpleNamespace(
        coupons=[
            SimpleNamespace(start_date=s, end_date=e, rate=rate)
            for s, e in zip(starts, ends)
        ]
    )
    z = _seeder(Market(t=t, risk_free_index=RISKLESS)).fixed_leg_zero(leg)
    # Spot gap and both coupons at 5% annual over [t, last end]: continuous rate log(1.05).
    assert z == pytest.approx(math.log(1.05), rel=1e-12)


def test_basis_seed_shifts_the_index_curve_by_the_spread():
    # CLP leg ICP + s on the CLP funding curve (cancels) vs USD leg SOFR flat on the
    # unknown USD funding curve: z_USD = z_SOFR - s.
    t, maturity = date(2024, 1, 2), date(2026, 1, 2)
    sofr = OvernightRateIndex("SOFR", currency=Currency.USD)
    z_sofr, spread = 0.045, 0.003
    market = Market(
        t=t,
        risk_free_index=RISKLESS,
        curves=[
            IndexCurve(t, RISKLESS, [t, maturity], [0.06]),
            IndexCurve(t, sofr, [t, maturity], [z_sofr]),
        ],
    )
    clp_leg = SimpleNamespace(
        index=RISKLESS, currency=Currency.CLP, spread=Rate(RateConvention(), spread)
    )
    usd_leg = SimpleNamespace(index=sofr, currency=Currency.USD, spread=None)
    swap = SimpleNamespace(
        receive_leg=clp_leg,
        pay_leg=usd_leg,
        leg_settlement_currency=lambda leg: leg.currency,
    )
    implied = defaultdict(dict)

    _seeder(market).basis_zero(
        swap,
        implied,
        {Currency.USD: []},
        {Currency.USD: [maturity]},
        lambda curve, _: maturity if curve == Currency.USD else None,
    )
    assert implied[Currency.USD][maturity] == pytest.approx(z_sofr - spread, rel=1e-9)


def test_index_anchor_supports_price_based_overnight_index():
    # A price-based overnight index (ICP levels) must still anchor the short end, via
    # OvernightInterestPriceHistory.spot_overnight_rate — not fall back to the curve date.
    t = date(2024, 1, 4)
    icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
    history = OvernightInterestPriceHistory(
        index=icp,
        values={
            date(2024, 1, 2): 100.00,
            date(2024, 1, 3): 100.01,
            date(2024, 1, 4): 100.02,
        },
    )
    market = Market(t=t, indexes_history={"ICP": history}, risk_free_index=icp)

    fixed, guess = _seeder(market).index_anchor(icp)
    assert (
        fixed.knots == [icp.get_maturity(t)] and fixed.forwards == []
    )  # effective: next business day
    assert fixed.spot_rate is not None and fixed.spot_rate.rate > 0.0
    ((anchor_date, df),) = fixed.knot_dfs(t, icp)
    assert anchor_date == icp.get_maturity(t) and 0.0 < df < 1.0
    assert guess > 0.0


def test_overnight_index_without_history_starts_at_the_curve_date_with_the_flat_guess():
    t = date(2024, 1, 2)
    fixed, guess = _seeder(Market(t=t, risk_free_index=RISKLESS)).index_anchor(RISKLESS)
    assert (fixed.knots, fixed.forwards, fixed.spot_rate, guess) == (
        [t],
        [],
        None,
        0.04,
    )


def test_a_term_index_anchor_needs_a_rate_history():
    term = TermRateIndex(
        "TERM3M",
        currency=Currency.USD,
        term=Term(3, TermUnit.M, ModifiedFollowingConvention("US")),
    )
    with pytest.raises(InsufficientQuotesError, match="no rate history"):
        _seeder(Market(t=date(2024, 1, 2), risk_free_index=RISKLESS)).index_anchor(term)
