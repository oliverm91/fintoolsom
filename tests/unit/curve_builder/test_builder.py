"""End-to-end behaviour of build_curves and the _CurveBuilder orchestrator."""

import inspect
from datetime import date
import pytest
import warnings

import math

from fintoolsom.curve_builder import (
    InsufficientQuotesError,
    RedundantQuoteWarning,
    RepricingError,
    build_curves,
)
from fintoolsom.curve_builder._keys import _curves_needed
from fintoolsom.curve_builder.builder import _CurveBuilder
from fintoolsom.rates import InterpolationMethod
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from fintoolsom.market.quotes import ForwardPriceQuote
from fintoolsom.rates import IndexCurve
from tests.factories.curve_factory import discount_curve_from_dfs


USDCLP = CurrencyPair(Currency.USD, Currency.CLP)


RISKLESS = OvernightRateIndex(
    "OIS", currency=Currency.CLP
)  # funds CLP; USD has its own curve


def _index_curve(t: date, d: date, df: float) -> IndexCurve:
    """RISKLESS curve with one segment reaching `df` at `d`."""
    forward = -math.log(df) / ((d - t).days / 365)
    return IndexCurve(t, RISKLESS, [t, t + (d - t) / 2, d], [forward, forward])


def _market(t: date, spot: float, usd_df: float, payment_date: date) -> Market:
    market = Market(
        t=t,
        risk_free_index=RISKLESS,
        curves=[
            discount_curve_from_dfs(t, Currency.USD, RISKLESS, [payment_date], [usd_df])
        ],
    )
    market.add_fx_rate(t, FX_Rate(USDCLP, spot))
    return market


_FORWARD_DATES = [date(2024, 7, 2), date(2025, 1, 2)]
_USD_DFS = [0.98, 0.96]
_CLP_DFS = [0.95, 0.90]


def _two_forward_case():
    t, spot = date(2024, 1, 2), 900.0
    market = Market(
        t=t,
        risk_free_index=RISKLESS,
        curves=[
            discount_curve_from_dfs(t, Currency.USD, RISKLESS, _FORWARD_DATES, _USD_DFS)
        ],
    )
    market.add_fx_rate(t, FX_Rate(USDCLP, spot))
    quotes = [
        ForwardPriceQuote(
            currency_pair=USDCLP,
            value=spot * usd_df / clp_df,
            is_buy=True,
            payment_date=d,
            quote_date=t,
        )
        for d, usd_df, clp_df in zip(_FORWARD_DATES, _USD_DFS, _CLP_DFS)
    ]
    return market, quotes


def test_build_curves_solves_missing_currency_curve_from_two_forward_quotes():
    market, quotes = _two_forward_case()
    build_curves(quotes, market)
    curve = market.get_index_curve(RISKLESS)
    assert curve.get_dfs(_FORWARD_DATES) == pytest.approx(_CLP_DFS, abs=1e-6)
    # USD curve, already fully seeded at these maturities, must be left untouched.
    assert market.get_discount_curve(Currency.USD).get_dfs(
        _FORWARD_DATES
    ) == pytest.approx(_USD_DFS)


def test_build_curves_raises_insufficient_quotes_when_under_determined():
    t = date(2024, 1, 2)
    payment_date = date(2024, 7, 2)
    market = Market(t=t, risk_free_index=RISKLESS)
    market.add_fx_rate(t, FX_Rate(USDCLP, 900.0))
    quote = ForwardPriceQuote(
        currency_pair=USDCLP,
        value=928.0,
        is_buy=True,
        payment_date=payment_date,
        quote_date=t,
    )
    # Neither USD nor CLP curve exists yet: 1 quote, 2 unknown pillars.
    with pytest.raises(InsufficientQuotesError):
        build_curves([quote], market)


def test_build_curves_raises_when_quote_date_mismatches_market_t():
    t = date(2024, 1, 2)
    market = Market(t=date(2024, 1, 3), risk_free_index=RISKLESS)
    quote = ForwardPriceQuote(
        currency_pair=USDCLP,
        value=900.0,
        is_buy=True,
        payment_date=date(2024, 7, 2),
        quote_date=t,
    )
    with pytest.raises(ValueError):
        build_curves([quote], market)


def test_curves_needed_forward_quote_does_not_crash_on_missing_collateral_index():
    # ForwardPriceQuote has no collateral_index attribute at all; _curves_needed
    # must not assume every quote type carries one.
    quote = ForwardPriceQuote(
        currency_pair=USDCLP,
        value=900.0,
        is_buy=True,
        payment_date=date(2024, 7, 2),
        quote_date=date(2024, 1, 2),
    )
    keys = _curves_needed(quote, RISKLESS)
    assert keys == frozenset({Currency.USD, RISKLESS})


def _builder(market: Market, quotes=()) -> _CurveBuilder:
    return _CurveBuilder(list(quotes), market)


def _forward_case():
    t, payment_date, spot, usd_df, target_clp_df = (
        date(2024, 1, 2),
        date(2024, 7, 2),
        900.0,
        0.98,
        0.95,
    )
    market = _market(t, spot, usd_df, payment_date)
    quote = ForwardPriceQuote(
        currency_pair=USDCLP,
        value=spot * usd_df / target_clp_df,
        is_buy=True,
        payment_date=payment_date,
        quote_date=t,
    )
    return market, quote


def test_quote_falls_back_to_the_curve_it_can_still_pin():
    # USD funding (the forward's preferred curve) already covers the maturity: the
    # forward's pillar goes to the unbuilt CLP curve instead.
    market, quote = _forward_case()
    builder = _builder(market, [quote])
    assert builder.plan.candidates[quote] == (Currency.USD, RISKLESS)
    assert builder.plan.assign_pillars(
        [quote.get_instrument()], [builder.plan.candidates[quote]]
    ) == [RISKLESS]


def test_a_curve_pinned_by_one_quote_raises():
    market, quote = _forward_case()
    with pytest.raises(
        InsufficientQuotesError, match="OIS would have a single segment"
    ):
        build_curves([quote], market)
    assert "OIS" not in market.curves


def test_every_pinning_quote_is_checked_after_the_build():
    market, quotes = _two_forward_case()
    builder = _builder(market, quotes)
    builder.build()
    assert [i for i, _ in builder._pinning] == [q.get_instrument() for q in quotes]


def test_a_quote_moved_off_par_after_its_solve_raises():
    market, quotes = _two_forward_case()
    builder = _builder(market, quotes)
    builder.build()
    # A later solve moving the curve the forward was fitted on: simulate it.
    market.set_curve(_index_curve(market.t, date(2024, 7, 2), 0.90))
    with pytest.raises(RepricingError, match="no longer reprice"):
        builder._check_repricing()


def test_a_build_without_ditched_quotes_does_not_warn():
    market, quotes = _two_forward_case()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RedundantQuoteWarning)
        build_curves(quotes, market)


def test_a_forward_inside_built_curves_is_ditched_with_its_mtm_over_the_base_notional():
    market, quote = _forward_case()
    market.set_curve(_index_curve(market.t, date(2024, 12, 31), 0.93))
    builder = _builder(market, [quote])
    with pytest.warns(
        RedundantQuoteWarning, match="ForwardPriceQuote 2024-07-02 USD-CLP: MTM"
    ) as record:
        builder.build()
    ((ditched, mtm, pct, bp),) = record[0].message.ditched
    assert ditched is quote and mtm != 0
    assert bp is None  # not a swap: no quoted rate to express the error in
    assert "bp" not in str(record[0].message) and f"= {pct:.4%} of notional" in str(
        record[0].message
    )
    instrument = quote.get_instrument()
    assert pct == pytest.approx(
        mtm / (instrument.notional * 900.0), rel=1e-12
    )  # base USD notional, in CLP


# ── Interpolation method ────────────────────────────────────────────────────


def test_curves_default_to_the_builders_interpolation():
    t = date(2024, 1, 2)
    params = inspect.signature(build_curves).parameters
    discount = discount_curve_from_dfs(
        t, Currency.USD, RISKLESS, [date(2025, 1, 2)], [0.95]
    )
    assert (
        discount.interpolation_method
        is params["discount_interp_method"].default
        is InterpolationMethod.PiecewiseConstant
    )
    index = IndexCurve(
        t, RISKLESS, [t, date(2024, 7, 2), date(2025, 1, 2)], [0.05, 0.05]
    )
    assert (
        index.interpolation_method
        is params["index_interp_method"].default
        is InterpolationMethod.PiecewiseConstant
    )


def test_an_extended_curve_keeps_its_own_interpolation():
    from fintoolsom.market import ForwardPointsQuote, IRSQuote
    from tests.factories.quote_factory import load_quotes, quote_market, quote_mtm

    real = load_quotes()
    icp_irs = [
        q
        for q in real
        if isinstance(q, IRSQuote)
        and q.fixed_leg.currency == Currency.CLP
        and q.collateral_index.name == "ICP"
    ]
    forwards = [q for q in real if isinstance(q, ForwardPointsQuote)]
    market = quote_market()
    # USD funding from the FX forwards, monotone cubic.
    build_curves(
        icp_irs + forwards,
        market,
        discount_interp_method=InterpolationMethod.HermiteCubicSpline,
    )
    usd = market.get_discount_curve(Currency.USD)
    between = [
        date(2024, 2, 15),
        date(2024, 5, 20),
        date(2024, 11, 5),
    ]  # inside the forwards' range
    before = usd.get_dfs(between)

    # The rest extends USD funding past 1Y, PiecewiseConstant by default: the short end must not move.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RedundantQuoteWarning)
        build_curves([q for q in real if q not in icp_irs + forwards], market)
    extended = market.get_discount_curve(Currency.USD)
    assert extended.last_date > usd.last_date
    assert extended.interpolation_method is InterpolationMethod.HermiteCubicSpline
    assert extended.get_dfs(between) == pytest.approx(before, rel=1e-12)
    assert max(abs(quote_mtm(q, market)) for q in icp_irs + forwards) < 1e-6
