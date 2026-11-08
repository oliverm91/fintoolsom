"""Solver planning (_plan._Planner): candidates, pillar ownership, readiness, cycles."""
import dataclasses
import math
from datetime import date

import pytest

from fintoolsom.curve_builder._plan import _Planner
from fintoolsom.curve_builder._seeds import _Seeder
from fintoolsom.market import CrossCurrencyFloatFloatQuote, Currency, ForwardPointsQuote, IRSQuote, Market, OvernightRateIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from fintoolsom.market.quotes import ForwardPriceQuote
from fintoolsom.rates import DiscountCurve
from tests.factories.quote_factory import (
    ICP, SOFR, TERM3M, TERM6M, built_market, load_quotes, quote_market, term_basis_quotes, term_irs_quotes,
)

USD = Currency.USD


def _planner(quotes, market=None) -> _Planner:
    market = quote_market() if market is None else market
    return _Planner(quotes, market, _Seeder(market, market.risk_free_index))


def _is(kind, ccy=None, collateral="any"):
    def match(q):
        if not isinstance(q, kind):
            return False
        if ccy is not None and q.fixed_leg.currency != ccy:
            return False
        return collateral == "any" or getattr(q.collateral_index, "name", None) == collateral
    return match


@pytest.fixture(scope="module")
def real():
    return load_quotes()


def _group_of(plan, match):
    return next(g for g, qs in plan.groups.items() if any(match(q) for q in qs))


# ── Candidates ──────────────────────────────────────────────────────────────

def test_each_quote_type_prefers_the_curve_it_is_most_sensitive_to(real):
    plan = _planner(real)
    first = {q: plan.candidates[q][0] for q in real}
    assert {first[q] for q in real if _is(IRSQuote, USD)(q)} == {SOFR}           # OIS: its index
    assert {first[q] for q in real if _is(IRSQuote, Currency.CLP)(q)} == {ICP}   # CLP IRS: ICP
    assert {first[q] for q in real if isinstance(q, ForwardPointsQuote)} == {USD}  # forwards: USD funding
    assert {first[q] for q in real if isinstance(q, CrossCurrencyFloatFloatQuote)} == {USD}  # XCCY: the basis


def test_a_quote_never_falls_back_to_a_curve_another_type_prefers(real):
    plan = _planner(real)
    # ICP (CLP IRS) and SOFR (OIS) are preferred by other quotes: an XCCY can only pin USD.
    assert {plan.candidates[q] for q in real if isinstance(q, CrossCurrencyFloatFloatQuote)} == {(USD,)}


def test_a_basis_swap_prefers_the_index_no_irs_pins(real):
    irs = term_irs_quotes(real, tenors=["2Y"])
    basis = term_basis_quotes(irs)
    plan = _planner(irs + basis)
    assert plan.candidates[irs[0]][0] == TERM3M
    assert plan.candidates[basis[0]][0] == TERM6M


# ── Ownership ───────────────────────────────────────────────────────────────

def test_one_pillar_per_quote_on_its_preferred_curve(real):
    plan = _planner(real)
    ois = _group_of(plan, _is(IRSQuote, USD))
    owned = plan.owned_pillars(ois)
    assert len(owned) == len(plan.groups[ois]) and {c for c, _ in owned} == {SOFR}


def test_quotes_inside_built_curves_own_nothing(real):
    market, quotes = built_market(real)
    plan = _planner(quotes, market)
    assert all(not plan.owned_pillars(g) for g in plan.groups)


def test_a_quote_falls_back_when_its_preferred_curve_is_built():
    t, pay = date(2024, 1, 2), date(2024, 7, 2)
    riskless = OvernightRateIndex("OIS", currency=Currency.CLP)
    market = Market(t=t, risk_free_index=riskless, curves=[DiscountCurve(t, USD, riskless, [pay], [0.98])])
    usdclp = CurrencyPair(USD, Currency.CLP)
    market.add_fx_rate(t, FX_Rate(usdclp, 900.0))
    quote = ForwardPriceQuote(currency_pair=usdclp, value=928.0, is_buy=True, payment_date=pay, quote_date=t)
    plan = _planner([quote], market)
    (group,) = plan.groups
    assert plan.owned_pillars(group) == {(riskless, pay)}
    fixed, free, seeds = plan.partition_pillars(group, plan.instruments_of(group), [riskless])
    assert free == {riskless: [pay]}
    # An index curve is seeded in its first forward: covered interest parity on the df.
    assert math.exp(-seeds[riskless][0] * (pay - t).days / 365) == pytest.approx(900.0 * 0.98 / 928.0, rel=1e-12)


# ── Readiness ───────────────────────────────────────────────────────────────

def test_real_set_is_ordered_by_who_owns_what(real):
    plan = _planner(real)
    pending = list(plan.groups)
    owned = {g: plan.owned_pillars(g) for g in pending}
    waits = {g: plan.waits(g, pending, owned) for g in pending}
    icp = _group_of(plan, _is(IRSQuote, Currency.CLP, "ICP"))
    ois = _group_of(plan, _is(IRSQuote, USD))
    fwd = _group_of(plan, lambda q: isinstance(q, ForwardPointsQuote))
    xccy = _group_of(plan, lambda q: isinstance(q, CrossCurrencyFloatFloatQuote))
    assert waits[icp] == set() and waits[ois] == set()
    assert waits[fwd] == {icp}                # reads ICP the short CLP IRS still own
    assert waits[xccy] == {icp, ois, fwd}     # reads SOFR; USD's short end belongs to the forwards


def _cycle_quotes(real):
    """Non-CSA USD OIS (discount on USD funding) vs SOFR-CSA XCCY and CLP IRS."""
    ois = [dataclasses.replace(q, collateral_index=None) for q in real if _is(IRSQuote, USD)(q)]
    rest = [q for q in real if _is(IRSQuote, Currency.CLP, "SOFR")(q) or _is(CrossCurrencyFloatFloatQuote, collateral="SOFR")(q)]
    return ois, rest


def test_groups_reading_each_others_pillars_wait_on_each_other(real):
    ois, rest = _cycle_quotes(real)
    plan = _planner(ois + rest)
    a, b = _group_of(plan, lambda q: q in ois), _group_of(plan, lambda q: q in rest)
    pending = [a, b]
    owned = {g: plan.owned_pillars(g) for g in pending}
    assert plan.waits(a, pending, owned) == {b}  # OIS discount on USD funding, built by the XCCY
    assert plan.waits(b, pending, owned) == {a}  # XCCY read SOFR, built by the OIS


def test_a_cycle_is_merged_into_one_square_group(real):
    ois, rest = _cycle_quotes(real)
    plan = _planner(ois + rest)
    pending = list(plan.groups)
    owned = {g: plan.owned_pillars(g) for g in pending}
    waits = {g: plan.waits(g, pending, owned) for g in pending}
    assert plan.merge_cycles(pending, waits)
    (merged,) = pending
    assert set(plan.groups[merged]) == set(ois + rest)
    assert len(plan.owned_pillars(merged)) == len(ois + rest)  # one pillar per quote
    assert not plan.merge_cycles(pending, {merged: set()})  # nothing left to merge
