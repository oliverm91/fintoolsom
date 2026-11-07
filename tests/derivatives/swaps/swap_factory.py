"""Test-data factory for swap valuation tests.

Builds a single synthetic market (flat curves, daily fixings, ICP levels, FX) and
enumerates swap cases over:

* swap type    — IRS (both legs same currency) / CCS (legs in different currencies)
* leg kind     — fixed, term rate, overnight compounding rate, interest price index (ICP)
* currency     — CLP, USD, UF
* status       — not started, started (mid-coupon), finished (matured)
* spread       — floating legs with the builder's default (no) spread or with a spread
* collateral   — uncollateralised, or CSA-discounted on SOFR (USD), CAMARA (CLP),
                 ICP (CLP, price index) or ESTR (EUR: a third, non-leg currency)

Values are synthetic: the factory only guarantees every input the calculator may ask
for exists, so tests can check that valuation runs end to end.

UF note: the library has no UF (CLF) ``Currency`` yet, so a UF-denominated leg cannot
be built. UF cases are generated anyway and marked xfail until ``Currency.CLF`` exists;
once it does, the factory picks it up automatically (curves + CLF/CLP FX included).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from itertools import combinations, product

import pytest

from fintoolsom.dates import ActualDayCountConvention, Calendar
from fintoolsom.dates.adjustments import ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.legs import SwapLeg
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.market import (
    Currency,
    Market,
    OvernightInterestPriceHistory,
    OvernightInterestPriceIndex,
    OvernightRateHistory,
    OvernightRateIndex,
    TermRateHistory,
    TermRateIndex,
)
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from fintoolsom.rates import LinearInterestConvention, Rate, RateConvention, ZeroCouponCurve

# ── Dates ──────────────────────────────────────────────────────────────────

VAL_DATE = date(2024, 7, 8)  # Monday
HISTORY_START = date(2021, 12, 1)  # before the earliest swap start


class SwapStatus(Enum):
    NOT_STARTED = "not_started"
    STARTED = "started"  # valuation date sits inside the 2nd quarterly coupon
    FINISHED = "finished"  # matured before the valuation date


SWAP_TERM_YEARS = 2
PAYMENT_FREQUENCY = "QUARTERLY"
START_DATES: dict[SwapStatus, date] = {
    SwapStatus.NOT_STARTED: date(2024, 7, 15),
    SwapStatus.STARTED: date(2024, 1, 10),
    SwapStatus.FINISHED: date(2022, 1, 10),
}

# ── Conventions ────────────────────────────────────────────────────────────

LIN_ACT360 = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)


def _rate(value: float) -> Rate:
    return Rate(LIN_ACT360, value)


SPREAD = _rate(0.005)

# ── Currencies ─────────────────────────────────────────────────────────────

REPORTING_CURRENCY = Currency.CLP
UF_CURRENCY: Currency | None = getattr(Currency, "CLF", None)


class Ccy(Enum):
    CLP = "CLP"
    USD = "USD"
    UF = "UF"


def currency_of(ccy: Ccy) -> Currency:
    if ccy is Ccy.CLP:
        return Currency.CLP
    if ccy is Ccy.USD:
        return Currency.USD
    if UF_CURRENCY is None:
        raise NotImplementedError("No UF (CLF) Currency in fintoolsom.market.currencies.")
    return UF_CURRENCY


def adj_convention(ccy: Ccy) -> ModifiedFollowingConvention:
    return ModifiedFollowingConvention(Calendar(country=currency_of(ccy).locality.value))


NOTIONALS: dict[Ccy, float] = {Ccy.CLP: 950_000_000.0, Ccy.USD: 1_000_000.0, Ccy.UF: 25_000.0}
FIXED_RATES: dict[Ccy, float] = {Ccy.CLP: 0.055, Ccy.USD: 0.045, Ccy.UF: 0.025}
FX_TO_CLP: dict[Ccy, float] = {Ccy.CLP: 1.0, Ccy.USD: 950.0, Ccy.UF: 37_500.0}

# ── Indexes ────────────────────────────────────────────────────────────────

SOFR =OvernightRateIndex("SOFR", currency=Currency.USD)
TERM_SOFR_3M = TermRateIndex(
    "TSOFR3M", currency=Currency.USD, term=Term(3, TermUnit.M, adj_convention(Ccy.USD))
)
CAMARA = OvernightRateIndex("CAMARA", currency=Currency.CLP)
TAB_3M = TermRateIndex(
    "TAB90", currency=Currency.CLP, term=Term(3, TermUnit.M, adj_convention(Ccy.CLP))
)
ICP = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
# Chilean market: ICP funds CLP; every other currency has its own cross-currency curve.
RISKLESS = ICP
# Collateral-only index (no EUR legs are generated): a CSA rate in a third currency.
ESTR = OvernightRateIndex("ESTR", currency=Currency.EUR)

# CSA collateral options. None = uncollateralised (discount on the funding curve only).
COLLATERAL_INDEXES = {
    "uncoll": None,
    "coll_sofr": SOFR,
    "coll_camara": CAMARA,
    "coll_icp": ICP,
    "coll_estr": ESTR,
}


class LegKind(Enum):
    FIXED = "fixed"
    TERM = "term"  # TermRateLeg on a term-rate index
    OVERNIGHT = "overnight"  # OvernightLeg on an overnight rate index
    PRICE = "price"  # OvernightLeg on an interest-bearing price index (ICP)


# Floating index per (currency, kind). Missing pairs are not generated.
FLOAT_INDEXES = {
    (Ccy.USD, LegKind.TERM): TERM_SOFR_3M,
    (Ccy.USD, LegKind.OVERNIGHT): SOFR,
    (Ccy.CLP, LegKind.TERM): TAB_3M,
    (Ccy.CLP, LegKind.OVERNIGHT): CAMARA,
    (Ccy.CLP, LegKind.PRICE): ICP,
}

# Flat continuously-compounded zero rates for curves; fixings use the same level.
# Uncollateralised curves of the non-riskless currencies (cross-currency implied).
CURRENCY_CURVE_RATES = {Currency.USD: 0.049, Currency.EUR: 0.034}
if UF_CURRENCY is not None:
    CURRENCY_CURVE_RATES[UF_CURRENCY] = 0.022
FLAT_RATES = {
    ESTR: 0.035,
    SOFR: 0.050,
    TERM_SOFR_3M: 0.051,
    CAMARA: 0.060,
    TAB_3M: 0.062,
    ICP: 0.060,
}


def leg_kinds(ccy: Ccy) -> list[LegKind]:
    return [LegKind.FIXED] + [k for (c, k) in FLOAT_INDEXES if c is ccy]


# ── Cases ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LegCase:
    ccy: Ccy
    kind: LegKind

    @property
    def is_floating(self) -> bool:
        return self.kind is not LegKind.FIXED

    def __str__(self) -> str:
        return f"{self.ccy.value}-{self.kind.value}"


@dataclass(frozen=True)
class SwapCase:
    swap_type: str  # "irs" | "ccs"
    receive: LegCase
    pay: LegCase
    status: SwapStatus
    with_spread: bool
    collateral: str = "uncoll"  # key into COLLATERAL_INDEXES

    @property
    def uses_uf(self) -> bool:
        return Ccy.UF in (self.receive.ccy, self.pay.ccy)

    @property
    def collateral_index(self):
        return COLLATERAL_INDEXES[self.collateral]

    @property
    def id(self) -> str:
        spread = "spread" if self.with_spread else "nospread"
        if not (self.receive.is_floating or self.pay.is_floating):
            spread = "nofloat"
        return (
            f"{self.swap_type}-{self.receive}-vs-{self.pay}"
            f"-{self.status.value}-{spread}-{self.collateral}"
        )


def _leg_pairs() -> list[tuple[str, LegCase, LegCase]]:
    pairs: list[tuple[str, LegCase, LegCase]] = []
    # IRS: same currency, distinct leg kinds, at least one floating leg.
    for ccy in Ccy:
        for a, b in combinations(leg_kinds(ccy), 2):
            pairs.append(("irs", LegCase(ccy, a), LegCase(ccy, b)))
    # CCS: different currencies, any leg kind on each side (fixed/fixed included).
    for ca, cb in combinations(list(Ccy), 2):
        for a, b in product(leg_kinds(ca), leg_kinds(cb)):
            pairs.append(("ccs", LegCase(ca, a), LegCase(cb, b)))
    return pairs


def swap_cases() -> list[SwapCase]:
    cases: list[SwapCase] = []
    for swap_type, receive, pay in _leg_pairs():
        has_float = receive.is_floating or pay.is_floating
        spreads = (False, True) if has_float else (False,)
        for status, with_spread, collateral in product(SwapStatus, spreads, COLLATERAL_INDEXES):
            cases.append(SwapCase(swap_type, receive, pay, status, with_spread, collateral))
    return cases


def swap_case_params() -> list:
    """``swap_cases()`` as pytest params (ids set, UF cases xfail while unsupported)."""
    params = []
    for case in swap_cases():
        marks = []
        if case.uses_uf and UF_CURRENCY is None:
            marks.append(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    strict=True,
                    reason="UF is not a Currency yet: UF-denominated legs cannot be built.",
                )
            )
        params.append(pytest.param(case, id=case.id, marks=marks))
    return params


# ── Builders ───────────────────────────────────────────────────────────────


def build_leg(leg: LegCase, status: SwapStatus, with_spread: bool) -> SwapLeg:
    currency = currency_of(leg.ccy)
    common = dict(
        notional=NOTIONALS[leg.ccy],
        start_date=START_DATES[status],
        term=Term(SWAP_TERM_YEARS, TermUnit.Y, adj_convention(leg.ccy)),
        frequency=PAYMENT_FREQUENCY,
        adj_convention=adj_convention(leg.ccy),
    )
    if leg.kind is LegKind.FIXED:
        return builders.fixed_leg(rate=_rate(FIXED_RATES[leg.ccy]), currency=currency, **common)

    index = FLOAT_INDEXES[(leg.ccy, leg.kind)]
    # "No spread" deliberately uses the builder default, as a caller would.
    spread_kw = {"spread": SPREAD} if with_spread else {}
    if leg.kind is LegKind.TERM:
        return builders.term_rate_leg(index=index, fixing_lag=index.spot_lag, **common, **spread_kw)
    return builders.overnight_leg(index=index, **common, **spread_kw)


def build_swap(case: SwapCase) -> Swap:
    return Swap(
        receive_leg=build_leg(case.receive, case.status, case.with_spread),
        pay_leg=build_leg(case.pay, case.status, case.with_spread),
        payment_currency=currency_of(case.receive.ccy) if case.swap_type == "irs" else None,
        collateral_index=case.collateral_index,
    )


# ── Market ─────────────────────────────────────────────────────────────────


def _flat_curve(rate: float) -> ZeroCouponCurve:
    pillars = [VAL_DATE + timedelta(days=d) for d in (1, 7, 30, 91, 182, 365, 730, 1095, 1825, 3650)]
    dfs = [(d, math.exp(-rate * (d - VAL_DATE).days / 365)) for d in pillars]
    return ZeroCouponCurve(VAL_DATE, date_dfs=dfs)


def _history_days() -> list[date]:
    n = (VAL_DATE - HISTORY_START).days
    return [HISTORY_START + timedelta(days=i) for i in range(n + 1)]


def build_market() -> Market:
    """Market at VAL_DATE with everything the swap calculator can request."""
    days = _history_days()

    # One curve per index (floating and collateral indexes; ICP doubles as CLP funding),
    # plus one cross-currency curve per non-riskless currency. Nothing is stored twice.
    indexes = {*FLOAT_INDEXES.values(), *(c for c in COLLATERAL_INDEXES.values() if c is not None)}
    curves: dict = {index: _flat_curve(FLAT_RATES[index]) for index in indexes}
    curves.update({ccy: _flat_curve(r) for ccy, r in CURRENCY_CURVE_RATES.items()})

    # Daily fixings (every calendar day, so any calendar's business days are covered),
    # exposed both as Market.interest_rates (read by the calculator today) and as
    # IndexHistory objects.
    interest_rates: dict[str, dict[date, Rate]] = {}
    for index in FLOAT_INDEXES.values():
        interest_rates[index.name.upper()] = {d: _rate(FLAT_RATES[index]) for d in days}
    index_to_rate = {name: name for name in interest_rates}

    icp_levels: dict[date, float] = {}
    level = 10_000.0
    for d in days:
        icp_levels[d] = level
        level *= 1 + FLAT_RATES[ICP] / 360

    histories = {
        SOFR.name: OvernightRateHistory(SOFR, dict(interest_rates[SOFR.name])),
        CAMARA.name: OvernightRateHistory(CAMARA, dict(interest_rates[CAMARA.name])),
        TERM_SOFR_3M.name: TermRateHistory(TERM_SOFR_3M, dict(interest_rates[TERM_SOFR_3M.name])),
        TAB_3M.name: TermRateHistory(TAB_3M, dict(interest_rates[TAB_3M.name])),
        ICP.name: OvernightInterestPriceHistory(ICP, icp_levels),
    }

    market = Market(
        t=VAL_DATE,
        riskless_index=RISKLESS,
        curves=curves,
        interest_rates=interest_rates,
        index_to_interest_rate_map=index_to_rate,
        indexes_history=histories,
    )
    for ccy, fx in FX_TO_CLP.items():
        if ccy is Ccy.CLP or (ccy is Ccy.UF and UF_CURRENCY is None):
            continue
        market.add_fx_rate(VAL_DATE, FX_Rate(CurrencyPair(currency_of(ccy), REPORTING_CURRENCY), fx))
    return market
