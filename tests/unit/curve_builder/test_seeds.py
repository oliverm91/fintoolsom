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
from fintoolsom.market import Currency, Market, OvernightInterestPriceIndex, OvernightRateIndex
from fintoolsom.market.index import TermRateIndex
from fintoolsom.market.index_history import OvernightInterestPriceHistory
from fintoolsom.rates import Rate, RateConvention, ZeroCouponCurve

RISKLESS = OvernightRateIndex("OIS", currency=Currency.CLP)


def _seeder(market: Market) -> _Seeder:
    return _Seeder(market, market.riskless_index)


def test_fixed_leg_seed_compounds_the_coupons_to_the_leg_end():
    t = date(2024, 1, 2)
    starts = [date(2024, 1, 4), date(2024, 7, 4)]
    ends = [date(2024, 7, 4), date(2025, 1, 6)]
    rate = Rate(RateConvention(), 0.05)  # annual compounding, act/365
    leg = SimpleNamespace(coupons=[SimpleNamespace(start_date=s, end_date=e, rate=rate) for s, e in zip(starts, ends)])
    z = _seeder(Market(t=t, riskless_index=RISKLESS)).fixed_leg_zero(leg)
    # Spot gap and both coupons at 5% annual over [t, last end]: continuous rate log(1.05).
    assert z == pytest.approx(math.log(1.05), rel=1e-12)


def test_basis_seed_shifts_the_index_curve_by_the_spread():
    # CLP leg ICP + s on the CLP funding curve (cancels) vs USD leg SOFR flat on the
    # unknown USD funding curve: z_USD = z_SOFR - s.
    t, maturity = date(2024, 1, 2), date(2026, 1, 2)
    sofr = OvernightRateIndex("SOFR", currency=Currency.USD)
    z_sofr, spread = 0.045, 0.003
    tau = (maturity - t).days / 365
    market = Market(
        t=t, riskless_index=RISKLESS,
        curves={
            RISKLESS: ZeroCouponCurve(t, date_dfs=[(maturity, math.exp(-0.06 * tau))]),
            sofr: ZeroCouponCurve(t, date_dfs=[(maturity, math.exp(-z_sofr * tau))]),
        },
    )
    clp_leg = SimpleNamespace(index=RISKLESS, currency=Currency.CLP, spread=Rate(RateConvention(), spread))
    usd_leg = SimpleNamespace(index=sofr, currency=Currency.USD, spread=None)
    swap = SimpleNamespace(receive_leg=clp_leg, pay_leg=usd_leg, leg_settlement_currency=lambda leg: leg.currency)
    implied = defaultdict(dict)

    _seeder(market).basis_zero(
        swap, implied, {Currency.USD: []}, {Currency.USD: [maturity]},
        lambda curve, _: maturity if curve == Currency.USD else None,
    )
    assert implied[Currency.USD][maturity] == pytest.approx(z_sofr - spread, rel=1e-9)


def test_short_end_anchor_supports_price_based_overnight_index():
    # A price-based overnight index (ICP levels) must still anchor the short end, via
    # OvernightInterestPriceHistory.spot_overnight_rate — not fall back to the flat guess.
    t = date(2024, 1, 4)
    icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
    history = OvernightInterestPriceHistory(
        index=icp, values={date(2024, 1, 2): 100.00, date(2024, 1, 3): 100.01, date(2024, 1, 4): 100.02},
    )
    market = Market(t=t, indexes_history={"ICP": history}, riskless_index=icp)

    anchor, guess = _seeder(market).short_end_anchor(icp)
    assert anchor is not None  # not the flat-guess fallback
    anchor_date, df = anchor
    assert anchor_date == icp.get_maturity(t)  # next business day
    assert 0.0 < df < 1.0
    assert guess > 0.0


def test_curve_without_overnight_history_gets_the_flat_guess():
    market = Market(t=date(2024, 1, 2), riskless_index=RISKLESS)
    assert _seeder(market).short_end_anchor(Currency.USD) == (None, 0.04)


def test_projection_anchor_needs_a_rate_history():
    term = TermRateIndex("TERM3M", currency=Currency.USD, term=Term(3, TermUnit.M, ModifiedFollowingConvention("US")))
    with pytest.raises(InsufficientQuotesError, match="no rate history"):
        _seeder(Market(t=date(2024, 1, 2), riskless_index=RISKLESS)).projection_anchor(term)
