"""Valuation values checked against independent computations.

Two kinds of checks, neither reusing the calculator's code paths:

* an independent re-pricer, written from first principles with only curve
  discount factors, the loaded fixings and spot, run over every factory swap case;
* closed-form identities that hold whatever the implementation: a floating leg
  projected and discounted on one curve is worth ``N * DF(start)`` (or its
  realised growth once started), a par swap is worth 0, a non-deliverable swap
  fixing on its payment dates equals its deliverable twin, a forward at its fair
  strike is worth 0, ...
"""

from __future__ import annotations
from datetime import date, timedelta
import pytest
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.forwards.forwards import NDF, Forward
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.legs import FixedLeg, TermRateLeg
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair
from tests.factories.swap_factory import (
    CAMARA,
    FLAT_RATES,
    FX_TO_CLP,
    ICP,
    PAST_FX_FACTOR,
    SOFR,
    SPREAD,
    START_DATES,
    SWAP_TERM_YEARS,
    TAB_3M,
    TERM_SOFR_3M,
    UF_KNOWN_UNTIL,
    VAL_DATE,
    Ccy,
    SwapCase,
    SwapStatus,
    _rate,
    adj_convention,
    build_market,
    build_swap,
    swap_case_params,
    swap_cases,
    uf_level,
)


CLP, USD, CLF = Currency.CLP, Currency.USD, Currency.CLF


USDCLP = CurrencyPair(USD, CLP)


MARKET = build_market()


T = VAL_DATE


def _df(key, d: date) -> float:
    curve = (
        MARKET.get_discount_curve(key)
        if isinstance(key, Currency)
        else MARKET.get_index_curve(key)
    )
    return curve.get_df(d)


def _funding_df(ccy: Currency, d: date) -> float:
    return _df(ICP if ccy == CLP else ccy, d)  # ICP funds CLP; others own curve


def _discount(ccy: Currency, d: date, collateral) -> float:
    df = _funding_df(ccy, d)
    if collateral is not None:
        df *= _df(collateral, d) / _funding_df(collateral.currency, d)
    return df


def _yf360(s: date, e: date) -> float:
    return (e - s).days / 360


def _compounded(index, s: date, e: date) -> float:
    """Growth factor of a flat daily fixing compounded on the index calendar."""
    r, cal, g, d = FLAT_RATES[index], index.calendar, 1.0, s
    while d < e:
        nd = min(cal.add_business_days(d, 1), e)
        g *= 1 + r * _yf360(d, nd)
        d = nd
    return g


def _realised_growth(index, s: date, e: date) -> float:
    if index == ICP:
        levels = MARKET.get_index(ICP.name).values
        return levels[e] / levels[s]
    return _compounded(index, s, e)


def _projected_growth(index, s: date, e: date) -> float:
    return _df(index, s) / _df(index, e)


def _flow(leg, i: int) -> float:
    c = leg.coupons[i]
    n, s, e = c.residual, c.start_date, c.end_date
    if isinstance(leg, FixedLeg):
        interest = n * c.rate.rate * _yf360(s, e)
    elif isinstance(leg, TermRateLeg):
        if leg.fixing_dates[i] <= T:
            interest = n * FLAT_RATES[leg.index] * _yf360(s, e)
        else:
            interest = n * (_projected_growth(leg.index, s, e) - 1)
    elif s <= T:  # running overnight coupon: realised to t, projected after
        growth = _realised_growth(leg.index, s, min(T, e))
        if T < e:
            growth *= _projected_growth(leg.index, T, e)
        interest = n * (growth - 1)
    else:
        interest = n * (_projected_growth(leg.index, s, e) - 1)
    spread = 0.0 if isinstance(leg, FixedLeg) else n * leg.spread.rate * _yf360(s, e)
    return interest + spread + c.amortization


def _fx(base: Currency, quote: Currency, f: date) -> float:
    if base == quote:
        return 1.0
    if base == CLF:
        known, level = UF_KNOWN_UNTIL, uf_level
    else:  # USD: loaded up to today, realised fixings off spot by PAST_FX_FACTOR
        known = T
        level = lambda d: FX_TO_CLP[Ccy.USD] * (1 if d == T else PAST_FX_FACTOR)  # noqa: E731
    if f <= known:
        return level(f)
    return (
        level(known)
        * (_funding_df(base, f) / _funding_df(base, known))
        / (_funding_df(quote, f) / _funding_df(quote, known))
    )


def _settlement(swap: Swap, leg) -> Currency:
    if not swap.is_deliverable and swap.payment_currency is not None:
        return swap.payment_currency
    return CLP if leg.currency == CLF else leg.currency


def _hand_leg_pv_clp(swap: Swap, leg) -> float:
    settle = _settlement(swap, leg)
    pv = 0.0
    for i, c in enumerate(leg.coupons):
        if c.payment_date <= T:
            continue
        fx = _fx(leg.currency, settle, leg.fx_fixing_dates[i])
        pv += (
            _flow(leg, i)
            * fx
            * _discount(settle, c.payment_date, swap.collateral_index)
        )
    return pv * (1.0 if settle == CLP else FX_TO_CLP[Ccy.USD])


def _hand_swap_mtm_clp(swap: Swap) -> float:
    return _hand_leg_pv_clp(swap, swap.receive_leg) - _hand_leg_pv_clp(
        swap, swap.pay_leg
    )


@pytest.mark.parametrize("case", swap_case_params())
def test_every_swap_matches_the_independent_repricer(case: SwapCase):
    swap = build_swap(case)
    expected = _hand_swap_mtm_clp(swap)
    scale = 1_000_000_000.0  # ~ the CLP notionals: 1e-9 relative
    assert swap.valuate(MARKET, CLP) == pytest.approx(
        expected, rel=1e-9, abs=1e-9 * scale
    )


N = 1_000_000.0


def _float_leg(index, status=SwapStatus.NOT_STARTED, spread=None):
    ccy = Ccy.USD if index.currency == USD else Ccy.CLP
    kw = dict(
        notional=N,
        start_date=START_DATES[status],
        term=Term(SWAP_TERM_YEARS, TermUnit.Y, adj_convention(ccy)),
        frequency="QUARTERLY",
        adj_convention=adj_convention(ccy),
        index=index,
    )
    if spread is not None:
        kw["spread"] = spread
    if isinstance(index, type(TAB_3M)):
        return builders.term_rate_leg(**kw)
    return builders.overnight_leg(**kw)


@pytest.mark.parametrize(
    "index", [ICP, CAMARA, SOFR, TAB_3M, TERM_SOFR_3M], ids=lambda i: i.name
)
def test_floating_leg_on_its_own_curve_is_worth_notional_at_start(index):
    # sum N (P(s)/P(e) - 1) P(e) + N P(e_n) telescopes to N P(s_0).
    leg = _float_leg(index)
    assert leg.payment_dates == leg.end_dates
    pv = Calculator._leg_pv(leg, index, MARKET)  # discounted on the index curve
    assert pv == pytest.approx(N * _df(index, leg.start_dates[0]), rel=1e-12)


def test_spread_adds_its_annuity():
    plain, spread = _float_leg(CAMARA), _float_leg(CAMARA, spread=SPREAD)
    annuity = sum(
        N * SPREAD.rate * _yf360(c.start_date, c.end_date) * _df(CAMARA, c.payment_date)
        for c in spread.coupons
    )
    diff = Calculator._leg_pv(spread, CAMARA, MARKET) - Calculator._leg_pv(
        plain, CAMARA, MARKET
    )
    assert diff == pytest.approx(annuity, rel=1e-12)


@pytest.mark.parametrize("index", [ICP, CAMARA], ids=lambda i: i.name)
def test_started_overnight_leg_is_worth_notional_grown_by_realised_fixings(index):
    # Projection and discounting on one curve: everything after t telescopes to
    # (realised growth of the running coupon) * N * P(t), with P(t) = 1.
    leg = _float_leg(index, SwapStatus.STARTED)
    running = next(c for c in leg.coupons if c.start_date <= T < c.payment_date)
    if index == ICP:
        levels = MARKET.get_index(ICP.name).values
        growth = levels[T] / levels[running.start_date]
    else:
        growth = _compounded(CAMARA, running.start_date, T)
    assert growth > 1
    assert Calculator._leg_pv(leg, index, MARKET) == pytest.approx(
        N * growth, rel=1e-12
    )


def test_started_term_leg_is_worth_its_fixed_coupons_plus_notional_at_the_last_fixed_end():
    # Unpaid coupons already fixed (running one, and the next one fixing today)
    # pay N R yf; the projected ones telescope to N P(end of the last fixed coupon).
    leg = _float_leg(TAB_3M, SwapStatus.STARTED)
    fixed = [
        c for c, f in zip(leg.coupons, leg.fixing_dates) if f <= T < c.payment_date
    ]
    assert len(fixed) == 2 and leg.fixing_dates[leg.coupons.index(fixed[1])] == T
    r = FLAT_RATES[TAB_3M]
    expected = sum(
        N * r * _yf360(c.start_date, c.end_date) * _df(TAB_3M, c.payment_date)
        for c in fixed
    )
    expected += N * _df(TAB_3M, fixed[-1].end_date)
    assert Calculator._leg_pv(leg, TAB_3M, MARKET) == pytest.approx(expected, rel=1e-12)


def test_par_swap_is_worth_zero():
    float_leg = _float_leg(ICP)
    annuity = sum(
        _yf360(c.start_date, c.end_date) * _df(ICP, c.payment_date)
        for c in float_leg.coupons
    )
    par = (
        _df(ICP, float_leg.start_dates[0]) - _df(ICP, float_leg.end_dates[-1])
    ) / annuity
    fixed = builders.fixed_leg(
        notional=N,
        start_date=START_DATES[SwapStatus.NOT_STARTED],
        term=Term(SWAP_TERM_YEARS, TermUnit.Y, adj_convention(Ccy.CLP)),
        frequency="QUARTERLY",
        adj_convention=adj_convention(Ccy.CLP),
        rate=_rate(par),
        currency=CLP,
    )
    swap = Swap(receive_leg=fixed, pay_leg=float_leg, payment_currency=CLP)
    assert swap.valuate(MARKET, CLP) == pytest.approx(0.0, abs=1e-6)


def _ccs_twins():
    by_key = {}
    for case in swap_cases():
        if case.swap_type == "ccs":
            key = (
                case.receive,
                case.pay,
                case.status,
                case.with_spread,
                case.collateral,
            )
            by_key.setdefault(key, {})[case.deliverable] = case
    # A CCS with a UF leg is non-deliverable only: it has no deliverable twin.
    return [
        pytest.param(v[True], v[False], id=v[True].id)
        for v in by_key.values()
        if True in v
    ]


@pytest.mark.parametrize("deliverable, non_deliverable", _ccs_twins())
def test_non_deliverable_fixing_on_payment_equals_deliverable(
    deliverable, non_deliverable
):
    # spot * F_usd(p) / F_clp(p) converted, then discounted on F_clp(p) * C(p) / F_c(p),
    # is spot * F_usd(p) * C(p) / F_c(p): the deliverable USD discounting at spot.
    # Holds for any collateral while every unpaid flow fixes in the future.
    a, b = build_swap(deliverable), build_swap(non_deliverable)
    assert a.valuate(MARKET, CLP) == pytest.approx(
        b.valuate(MARKET, CLP), rel=1e-12, abs=1e-6
    )


@pytest.mark.parametrize(
    "case", [c for c in swap_case_params() if "usd" in c.id.lower()][:40]
)
def test_reporting_currency_is_a_spot_conversion(case):
    swap = build_swap(case)
    clp, usd = swap.valuate(MARKET, CLP), swap.valuate(MARKET, USD)
    assert usd * FX_TO_CLP[Ccy.USD] == pytest.approx(clp, rel=1e-12, abs=1e-6)


SPOT = FX_TO_CLP[Ccy.USD]


PAY = date(2025, 7, 1)


def test_forward_matches_its_hand_formula_and_reports_in_any_currency():
    fwd = Forward(N, 940.0, PAY, is_buy=True, currency_pair=USDCLP)
    expected = N * (SPOT * _funding_df(USD, PAY) - 940.0 * _funding_df(CLP, PAY))
    assert Calculator.valuate(fwd, MARKET, CLP) == pytest.approx(expected, rel=1e-12)
    assert Calculator.valuate(fwd, MARKET, USD) == pytest.approx(
        expected / SPOT, rel=1e-12
    )
    sell = Forward(N, 940.0, PAY, is_buy=False, currency_pair=USDCLP)
    assert Calculator.valuate(sell, MARKET, CLP) == pytest.approx(-expected, rel=1e-12)


def test_forward_at_the_covered_parity_strike_is_worth_zero():
    fair = SPOT * _funding_df(USD, PAY) / _funding_df(CLP, PAY)
    fwd = Forward(N, fair, PAY, is_buy=True, currency_pair=USDCLP)
    assert Calculator.valuate(fwd, MARKET, CLP) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("pay", [T - timedelta(days=7), T], ids=["past", "today"])
def test_settled_forwards_are_worth_zero(pay):
    fwd = Forward(N, 940.0, pay, is_buy=True, currency_pair=USDCLP)
    ndf = NDF(
        N,
        940.0,
        pay,
        is_buy=True,
        currency_pair=USDCLP,
        fixing_date=pay - timedelta(days=2),
    )
    assert Calculator.valuate(fwd, MARKET, CLP) == 0.0
    assert Calculator.valuate(ndf, MARKET, CLP) == 0.0


@pytest.mark.parametrize(
    "fixing, fx",
    [
        (
            date(2025, 6, 27),
            SPOT
            * _funding_df(USD, date(2025, 6, 27))
            / _funding_df(CLP, date(2025, 6, 27)),
        ),
        (T, SPOT),  # fixes today: spot
        (T - timedelta(days=3), SPOT * PAST_FX_FACTOR),  # fixed, not yet paid: realised
    ],
    ids=["projected", "fixes-today", "fixed-unpaid"],
)
def test_ndf_matches_its_hand_formula(fixing, fx):
    ndf = NDF(N, 940.0, PAY, is_buy=True, currency_pair=USDCLP, fixing_date=fixing)
    expected = N * (fx - 940.0) * _funding_df(CLP, PAY)
    assert Calculator.valuate(ndf, MARKET, CLP) == pytest.approx(expected, rel=1e-12)
    assert Calculator.valuate(ndf, MARKET, USD) == pytest.approx(
        expected / SPOT, rel=1e-12
    )


@pytest.mark.parametrize(
    "fixing", [UF_KNOWN_UNTIL - timedelta(days=5), UF_KNOWN_UNTIL + timedelta(days=40)]
)
def test_uf_ndf_matches_its_hand_formula(fixing):
    pair = CurrencyPair(CLF, CLP)
    pay = fixing + timedelta(days=1)
    ndf = NDF(
        1_000.0, 37_000.0, pay, is_buy=False, currency_pair=pair, fixing_date=fixing
    )
    expected = -1_000.0 * (_fx(CLF, CLP, fixing) - 37_000.0) * _funding_df(CLP, pay)
    assert Calculator.valuate(ndf, MARKET, CLP) == pytest.approx(expected, rel=1e-12)
