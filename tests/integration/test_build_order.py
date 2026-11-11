"""Build order: who owns which pillar decides when each group solves.

Synthetic par quotes priced off one known truth market (see
:func:`tests.factories.quote_factory.par_quotes`), so a correct build reprices every
pinning quote:

* B: CLP IRS, ICP+spread/SOFR XCCY and SOFR OIS on the same 30 terms, all SOFR CSA;
* C: misaligned tenor grids, plus a genuine cycle: non-CSA USD OIS (discounted on USD
  funding, built by the XCCY) against SOFR-CSA XCCY (reading SOFR, built by the OIS);
* C + FX forwards + short ICP-CSA CLP IRS: USD funding then exists early, which used
  to let the OIS solve SOFR off an extrapolated USD curve (no error, wrong curves).
"""

import random

import pytest

from fintoolsom.curve_builder import build_curves
from fintoolsom.curve_builder._plan import _Planner
from fintoolsom.market import (
    CrossCurrencyFloatFloatQuote,
    Currency,
    ForwardPointsQuote,
    IRSQuote,
)
from tests.factories.quote_factory import (
    SOFR,
    load_quotes,
    par_quotes,
    quote_mtm,
    term_market,
)

TOL = 1e-6  # CLP, on quote notionals of 100 USD (90,000 CLP)
ALIGNED = [
    "1M",
    "2M",
    "3M",
    "4M",
    "5M",
    "6M",
    "7M",
    "8M",
    "9M",
    "10M",
    "11M",
    "12M",
    "15M",
    "18M",
    "21M",
    "2Y",
    "3Y",
    "4Y",
    "5Y",
    "6Y",
    "7Y",
    "8Y",
    "9Y",
    "10Y",
    "11Y",
    "12Y",
    "15Y",
    "20Y",
    "25Y",
    "30Y",
]
OIS_TERMS = [
    "1M",
    "2M",
    "3M",
    "6M",
    "9M",
    "12M",
    "18M",
    "2Y",
    "3Y",
    "4Y",
    "5Y",
    "7Y",
    "10Y",
    "15Y",
    "20Y",
    "30Y",
]
IRS_TERMS = [
    "3M",
    "6M",
    "9M",
    "12M",
    "15M",
    "18M",
    "21M",
    "2Y",
    "30M",
    "3Y",
    "4Y",
    "5Y",
    "6Y",
    "7Y",
    "8Y",
    "9Y",
    "10Y",
    "12Y",
    "15Y",
    "20Y",
]
REDUNDANT_XCCY = {"1M", "2M", "4M", "6M", "12M"}  # inside the FX forwards' range
XCCY_TERMS = [
    "1M",
    "2M",
    "4M",
    "6M",
    "12M",
    "2Y",
    "3Y",
    "5Y",
    "7Y",
    "10Y",
    "15Y",
    "20Y",
    "25Y",
]


def _stress(ois_collateral):
    quotes = (
        par_quotes("usd_ois", OIS_TERMS, collateral=ois_collateral)
        + par_quotes("clp_irs", IRS_TERMS)
        + par_quotes("xccy", XCCY_TERMS)
        + par_quotes(
            "term_irs", ["12M", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
        )
        + par_quotes("term_basis", ["18M", "4Y", "6Y", "8Y", "12Y", "25Y"])
    )
    random.Random(7).shuffle(quotes)
    return quotes


@pytest.fixture(scope="module")
def three_swap_types():
    return (
        par_quotes("usd_ois", ALIGNED)
        + par_quotes("clp_irs", ALIGNED)
        + par_quotes("xccy", ALIGNED)
    )


@pytest.fixture(scope="module")
def cycle():
    return _stress(ois_collateral=None)


@pytest.fixture
def merges(monkeypatch) -> list[int]:
    """Number of groups merged by every merge_cycles call that merged something."""
    calls: list[int] = []
    original = _Planner.merge_cycles

    def spy(self, pending, waits):
        before = len(pending)
        merged = original(self, pending, waits)
        if merged:
            calls.append(before - len(pending) + 1)
        return merged

    monkeypatch.setattr(_Planner, "merge_cycles", spy)
    return calls


def _build(quotes):
    market = term_market()
    build_curves(quotes, market)
    return market


def _max_error(quotes, market) -> float:
    return max(abs(quote_mtm(q, market)) for q in quotes)


def test_three_sofr_csa_swap_types_on_one_grid_build(three_swap_types, merges):
    market = _build(three_swap_types)
    assert _max_error(three_swap_types, market) < TOL
    assert (
        merges == []
    )  # SOFR first (the OIS read only SOFR), then ICP and USD together


def test_misaligned_grids_without_a_cycle_build_without_merging(merges):
    quotes = _stress(ois_collateral=SOFR)
    market = _build(quotes)
    assert _max_error(quotes, market) < TOL
    assert merges == []


def test_a_genuine_cycle_is_merged_and_solved_jointly(cycle, merges):
    market = _build(cycle)
    assert _max_error(cycle, market) < TOL
    # Only the OIS and the XCCY/CLP IRS groups form the cycle; the term-index groups
    # read SOFR but own none of it, so they solve afterwards on their own.
    assert merges == [2]


def test_curves_partly_built_early_are_not_read_before_complete(cycle, merges):
    real = load_quotes()
    early = [
        q
        for q in real
        if isinstance(q, ForwardPointsQuote)
        or (
            isinstance(q, IRSQuote)
            and q.fixed_leg.currency == Currency.CLP
            and q.collateral_index.name == "ICP"
        )
    ]
    quotes = early + cycle
    market = _build(quotes)
    # The forwards own USD funding up to 1Y: the XCCY up to 12M are redundant there (and
    # read a SOFR short end that differs from the truth's between pillars). Every other
    # quote pins a pillar and reprices; the non-CSA OIS used to miss by ~950 CLP.
    redundant = [
        q
        for q in cycle
        if isinstance(q, CrossCurrencyFloatFloatQuote) and str(q.term) in REDUNDANT_XCCY
    ]
    assert _max_error([q for q in quotes if q not in redundant], market) < TOL
    assert merges == [2]


def test_quote_order_does_not_change_the_cycle_build(cycle):
    shuffled = list(cycle)
    random.Random(11).shuffle(shuffled)
    a, b = _build(cycle), _build(shuffled)
    assert set(a.curves) == set(b.curves)
    for key, curve in a.curves.items():
        dates = list(curve.knot_dates)
        assert curve.get_dfs(dates) == pytest.approx(
            b.curves[key].get_dfs(dates), rel=1e-9
        )


def test_ditched_quotes_are_warned_with_their_mtm():
    from fintoolsom.curve_builder import RedundantQuoteWarning
    from tests.factories.quote_factory import quote_market

    real = load_quotes()
    market = quote_market()
    with pytest.warns(RedundantQuoteWarning, match="5 quote") as record:
        build_curves(real, market)
    (warning,) = [
        w.message for w in record if isinstance(w.message, RedundantQuoteWarning)
    ]
    # The XCCY up to 1Y sit inside the FX forwards' range of USD funding.
    assert {str(q.term) for q, *_ in warning.ditched} == {"1M", "3M", "6M", "9M", "1Y"}
    for quote, mtm, pct, bp in warning.ditched:
        assert isinstance(quote, CrossCurrencyFloatFloatQuote)
        assert mtm == pytest.approx(quote_mtm(quote, market), rel=1e-12)
        assert pct is None and bp is not None and abs(bp) < 20  # a swap: bp only
        assert (
            f"CrossCurrencyFloatFloatQuote {quote.term} USD-CLP: MTM {mtm:,.4f} CLP = {bp:+.4f} bp"
            in str(warning)
        )
    assert "of notional" not in str(warning)


def test_a_ditched_swap_reports_how_many_bp_it_is_off_par():
    import dataclasses

    from fintoolsom.curve_builder import RedundantQuoteWarning
    from fintoolsom.rates import Rate
    from tests.factories.quote_factory import built_market

    market, real = built_market()
    irs = next(
        q
        for q in real
        if isinstance(q, IRSQuote)
        and q.fixed_leg.currency == Currency.CLP
        and str(q.term) == "5Y"
    )
    xccy = next(
        q
        for q in real
        if isinstance(q, CrossCurrencyFloatFloatQuote) and str(q.term) == "5Y"
    )
    off_irs = dataclasses.replace(
        irs,
        fixed_leg=dataclasses.replace(
            irs.fixed_leg,
            rate=Rate(
                irs.fixed_leg.rate.rate_convention, irs.fixed_leg.rate.rate + 2e-4
            ),
        ),
    )
    off_xccy = dataclasses.replace(
        xccy,
        receive_leg=dataclasses.replace(
            xccy.receive_leg,
            spread=Rate(
                xccy.receive_leg.spread.rate_convention,
                xccy.receive_leg.spread.rate - 3e-4,
            ),
        ),
    )
    # Every curve is already built past 5Y: both quotes are ditched. MTM is linear in the
    # quoted rate, so MTM / DV01 is exactly the bump.
    with pytest.warns(RedundantQuoteWarning) as record:
        build_curves([off_irs, off_xccy], market)
    assert "IRSQuote 5Y CLP: MTM" in str(record[0].message)
    assert "CrossCurrencyFloatFloatQuote 5Y USD-CLP: MTM" in str(record[0].message)
    bps = {type(q): bp for q, _, _, bp in record[0].message.ditched}
    assert abs(bps[IRSQuote]) == pytest.approx(2.0, rel=1e-6)
    assert abs(bps[CrossCurrencyFloatFloatQuote]) == pytest.approx(3.0, rel=1e-6)
