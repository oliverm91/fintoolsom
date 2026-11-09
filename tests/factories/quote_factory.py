"""Real quote set (CLP/USD, 2026-06-29) and the market built from it.

Port of ``notebooks/build_quotes.py`` reading ``tests/data/quotes_20260629.json``:
CLP fixed vs ICP and USD fixed vs SOFR OIS, USD/CLP forward points and ICP/SOFR
cross-currency basis swaps. Unlike the notebook loader, it reads the quoted
``spread_bps`` and uses one ICP index (the market's risk-free index) everywhere.
"""

from __future__ import annotations

import dataclasses
import json
from copy import deepcopy
from datetime import date
from pathlib import Path

from fintoolsom.curve_builder import build_curves
from fintoolsom.dates import (
    ActualDayCountConvention,
    ModifiedFollowingConvention,
    TF_Actual360,
)
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import (
    CrossCurrencyFloatFloatQuote,
    Currency,
    CurrencyPair,
    FixedLegSpec,
    FloatingLegSpec,
    ForwardPointsQuote,
    InstrumentQuote,
    IRBasisQuote,
    IRSQuote,
    Market,
    OvernightInterestPriceIndex,
    OvernightRateIndex,
    PaymentFrequency,
    QuotedSide,
)
from fintoolsom.market.currencies import FX_Rate, FX_RateData
from fintoolsom.market.index import TermRateIndex
from fintoolsom.market.index_history import (
    OvernightInterestPriceHistory,
    OvernightRateHistory,
    TermRateHistory,
)
from fintoolsom.market.localities import Locality
from fintoolsom.rates import (
    CompoundedInterestConvention,
    LinearInterestConvention,
    Rate,
    RateConvention,
)

QUOTES_FILE = Path(__file__).resolve().parent.parent / "data" / "quotes_20260629.json"
QUOTE_DATE = date(2026, 6, 29)
USDCLP_SPOT = 900.0
SOFR_FIXING = 0.0525

SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)
ICP = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
USDCLP = CurrencyPair(Currency.USD, Currency.CLP)

_CURRENCY = {"USD": Currency.USD, "CLP": Currency.CLP}
_INDEX = {"SOFR": SOFR, "ICP": ICP}
_INTEREST = {
    "Linear": LinearInterestConvention,
    "Compounded": CompoundedInterestConvention,
}
_TIME_FRACTION = {"TF_Actual360": (TF_Actual360(), ActualDayCountConvention, 360)}


def _merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        out[k] = (
            _merge(out[k], v)
            if isinstance(out.get(k), dict) and isinstance(v, dict)
            else v
        )
    return out


def _fixed_spec(leg: dict, adj, tf_key: str) -> FixedLegSpec:
    tf, dcc, base = _TIME_FRACTION[tf_key]
    convention = RateConvention(
        _INTEREST[leg.get("interest_convention", "Linear")], dcc, base
    )
    return FixedLegSpec(
        currency=_CURRENCY[leg["currency"]],
        payment_frequency=PaymentFrequency[leg["payment_frequency"]],
        adj_convention=adj,
        time_fraction=tf,
        rate=Rate(convention, leg["rate"]),
    )


def _float_spec(leg: dict, adj, tf_key: str) -> FloatingLegSpec:
    index = _INDEX[leg["index"]]
    spread = Rate(
        RateConvention(LinearInterestConvention, ActualDayCountConvention, 360),
        leg.get("spread_bps", 0.0) / 10_000,
    )
    return FloatingLegSpec(
        currency=_CURRENCY[leg["currency"]] if "currency" in leg else index.currency,
        payment_frequency=PaymentFrequency[leg["payment_frequency"]],
        adj_convention=adj,
        time_fraction=_TIME_FRACTION[tf_key][0],
        index=index,
        spread=spread,
    )


def load_quotes(
    quote_date: date = QUOTE_DATE, path: Path = QUOTES_FILE
) -> list[InstrumentQuote]:
    """Every quote of the JSON file, as InstrumentQuote objects."""
    groups = json.loads(path.read_text(encoding="utf-8"))
    quotes: list[InstrumentQuote] = []
    for group in groups:
        if group["term_adj_convention"] != "ModifiedFollowingConvention":
            raise ValueError(
                f"Unsupported adjustment {group['term_adj_convention']!r}."
            )
        adj = ModifiedFollowingConvention(group["term_calendar"])
        tf_key = group.get("day_count_convention")
        common = dict(
            quote_date=quote_date,
            adj_convention=adj,
            spot_lag=group.get("spot_lag", 2),
            stub_first=group.get("stub_first", True),
            long_stub=group.get("long_stub", False),
        )
        for q in group["quotes"]:
            term = Term(q["term"]["value"], TermUnit[q["term"]["unit"]], adj)
            collateral_name = q.get("collateral_index", group.get("collateral_index"))
            collateral = _INDEX[collateral_name] if collateral_name else None
            kind = group["quoteType"]
            if kind == "IRSQuote":
                quotes.append(
                    IRSQuote(
                        quoted_side=QuotedSide[group["quoted_side"]],
                        term=term,
                        collateral_index=collateral,
                        fixed_leg=_fixed_spec(
                            _merge(group["fixed_leg"], q.get("fixed_leg", {})),
                            adj,
                            tf_key,
                        ),
                        floating_leg=_float_spec(
                            _merge(group["floating_leg"], q.get("floating_leg", {})),
                            adj,
                            tf_key,
                        ),
                        **common,
                    )
                )
            elif kind == "CrossCurrencyFloatFloatQuote":
                quotes.append(
                    CrossCurrencyFloatFloatQuote(
                        quoted_side=QuotedSide[group["quoted_side"]],
                        term=term,
                        collateral_index=collateral,
                        receive_leg=_float_spec(
                            _merge(group["receive_leg"], q.get("receive_leg", {})),
                            adj,
                            tf_key,
                        ),
                        pay_leg=_float_spec(
                            _merge(group["pay_leg"], q.get("pay_leg", {})), adj, tf_key
                        ),
                        **common,
                    )
                )
            elif kind == "ForwardPointsQuote":
                base, quote_ccy = group["currency_pair"]
                quotes.append(
                    ForwardPointsQuote(
                        currency_pair=CurrencyPair(
                            _CURRENCY[base], _CURRENCY[quote_ccy]
                        ),
                        value=q["value"],
                        is_buy=group["is_buy"],
                        points_divisor=group.get("points_divisor", 1),
                        term=term,
                        quote_date=quote_date,
                        spot_lag=group.get("spot_lag", 2),
                        payment_lag=group.get("payment_lag", 1),
                        locality=Locality[group["locality"]]
                        if group.get("locality")
                        else None,
                    )
                )
            else:
                raise ValueError(f"Unsupported quote type {kind!r}.")
    return quotes


def quote_market(t: date = QUOTE_DATE) -> Market:
    """Market data the quotes need: SOFR fixing, ICP level, USD/CLP spot; no curves."""
    sofr_history = OvernightRateHistory(
        SOFR,
        {
            t: Rate(
                RateConvention(LinearInterestConvention, ActualDayCountConvention, 360),
                SOFR_FIXING,
            )
        },
    )
    icp_history = OvernightInterestPriceHistory(ICP, {t: 100.0})
    return Market(
        t,
        fx_history={USDCLP: FX_RateData(USDCLP, {t: FX_Rate(USDCLP, USDCLP_SPOT)})},
        indexes_history={"SOFR": sofr_history, "ICP": icp_history},
        risk_free_index=ICP,
    )


def built_market(
    quotes: list[InstrumentQuote] | None = None,
) -> tuple[Market, list[InstrumentQuote]]:
    """Market with every curve bootstrapped from `quotes` (the full JSON set by default)."""
    quotes = load_quotes() if quotes is None else quotes
    market = quote_market(quotes[0].quote_date)
    build_curves(quotes, market)
    return market, quotes


# ── Synthetic USD term-index quotes on top of the real set ──────────────────
# Projection-only indexes (no quote is collateralised in them): fixed vs TERM3M IRS
# and TERM6M vs TERM3M basis swaps, all SOFR-collateralised. The basis swaps need
# the TERM3M curve first, then pin TERM6M.

US_ADJ = ModifiedFollowingConvention("US")
TERM3M = TermRateIndex(
    "TERM3M", currency=Currency.USD, term=Term(3, TermUnit.M, US_ADJ)
)
TERM6M = TermRateIndex(
    "TERM6M", currency=Currency.USD, term=Term(6, TermUnit.M, US_ADJ)
)
TERM_FIXINGS = {TERM3M: 0.0540, TERM6M: 0.0545}
TERM_TENORS = [
    "12M",
    "18M",
    "2Y",
    "3Y",
    "4Y",
    "5Y",
    "7Y",
    "10Y",
    "12Y",
    "15Y",
    "20Y",
    "25Y",
    "30Y",
]
_LIN360 = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)


def _term_float(
    index: TermRateIndex, frequency: PaymentFrequency, spread: float = 0.0
) -> FloatingLegSpec:
    return FloatingLegSpec(
        currency=Currency.USD,
        payment_frequency=frequency,
        adj_convention=US_ADJ,
        time_fraction=TF_Actual360(),
        index=index,
        spread=Rate(_LIN360, spread),
    )


def term_irs_quotes(
    base: list[InstrumentQuote], spread_bps: float = 15.0, tenors=TERM_TENORS
) -> list[IRSQuote]:
    """Receive-fixed vs TERM3M IRS; fixed = the SOFR OIS rate of the tenor + `spread_bps`."""
    ois = {
        str(q.term): q
        for q in base
        if isinstance(q, IRSQuote) and q.fixed_leg.currency == Currency.USD
    }
    return [
        IRSQuote(
            quoted_side=QuotedSide.RECEIVE,
            term=ois[tenor].term,
            quote_date=QUOTE_DATE,
            adj_convention=US_ADJ,
            collateral_index=SOFR,
            fixed_leg=FixedLegSpec(
                currency=Currency.USD,
                payment_frequency=PaymentFrequency.QUARTERLY,
                adj_convention=US_ADJ,
                time_fraction=TF_Actual360(),
                rate=Rate(_LIN360, ois[tenor].fixed_leg.rate.rate + spread_bps / 1e4),
            ),
            floating_leg=_term_float(TERM3M, PaymentFrequency.QUARTERLY),
        )
        for tenor in tenors
    ]


def term_basis_quotes(
    irs: list[IRSQuote], spread_bps: float = 8.0
) -> list[IRBasisQuote]:
    """Receive TERM6M / pay TERM3M + `spread_bps`, one per tenor of `irs`."""
    return [
        IRBasisQuote(
            quoted_side=QuotedSide.RECEIVE,
            term=q.term,
            quote_date=QUOTE_DATE,
            adj_convention=US_ADJ,
            collateral_index=SOFR,
            receive_leg=_term_float(TERM6M, PaymentFrequency.SEMIANNUAL),
            pay_leg=_term_float(TERM3M, PaymentFrequency.QUARTERLY, spread_bps / 1e4),
        )
        for q in irs
    ]


def term_market(t: date = QUOTE_DATE) -> Market:
    """:func:`quote_market` plus today's TERM3M and TERM6M fixings."""
    market = quote_market(t)
    for index, fixing in TERM_FIXINGS.items():
        market.add_index(TermRateHistory(index, {t: Rate(_LIN360, fixing)}))
    return market


# ── Par quotes off a known truth market ─────────────────────────────────────
# Synthetic quotes on any tenor grid, collateral and product mix, each at par on one
# fixed "truth" market (the real quotes plus the term-index quotes): every set built
# from them is consistent, so a correct build reprices all of them.

_TRUTH: list[Market] = []


def truth_market() -> Market:
    """Market built from the real quotes plus the TERM3M IRS and TERM6M basis quotes (cached)."""
    if not _TRUTH:
        real = load_quotes()
        irs = term_irs_quotes(real)
        market = term_market()
        build_curves(real + irs + term_basis_quotes(irs), market)
        _TRUTH.append(market)
    return _TRUTH[0]


def quote_mtm(quote: InstrumentQuote, market: Market) -> float:
    """MTM of `quote`'s instrument in CLP."""
    from fintoolsom.derivatives.calculator import Calculator

    fx = isinstance(quote, (CrossCurrencyFloatFloatQuote, ForwardPointsQuote))
    return Calculator.valuate(
        quote.get_instrument(market) if fx else quote.get_instrument(),
        market,
        Currency.CLP,
    )


def _term(label: str, adj) -> Term:
    return Term(int(label[:-1]), TermUnit[label[-1]], adj)


def _at_par(make, x0: float):
    """``make(x)`` at the x where its MTM on the truth market is 0 (MTM is linear in x)."""
    truth, x = truth_market(), x0
    for _ in range(2):
        v0, v1 = quote_mtm(make(x), truth), quote_mtm(make(x + 1e-4), truth)
        x -= v0 * 1e-4 / (v1 - v0)
    return make(x)


def _with_rate(spec, x):
    return dataclasses.replace(spec, rate=Rate(spec.rate.rate_convention, x))


def _with_spread(spec, x):
    return dataclasses.replace(spec, spread=Rate(spec.spread.rate_convention, x))


def _template(kind: str):
    real = load_quotes()
    if kind in ("clp_irs", "usd_ois"):
        ccy = Currency.CLP if kind == "clp_irs" else Currency.USD
        return next(
            q
            for q in real
            if isinstance(q, IRSQuote)
            and q.fixed_leg.currency == ccy
            and str(q.term) == "2Y"
        )
    if kind == "xccy":
        q = next(
            q
            for q in real
            if isinstance(q, CrossCurrencyFloatFloatQuote) and str(q.term) == "2Y"
        )
        return dataclasses.replace(
            q, pay_leg=_with_spread(q.pay_leg, 0.0)
        )  # spread on the ICP leg instead
    if kind == "term_irs":
        return term_irs_quotes(real, tenors=["2Y"])[0]
    return term_basis_quotes(term_irs_quotes(real, tenors=["2Y"]))[0]


def par_quotes(kind: str, terms: list[str], collateral=SOFR) -> list[InstrumentQuote]:
    """Par quotes of one product on `terms`, priced off :func:`truth_market`.

    Parameters
    ----------
    kind : str
        'clp_irs' (CLP fixed vs ICP), 'usd_ois' (USD fixed vs SOFR), 'xccy' (receive
        ICP + spread / pay SOFR), 'term_irs' (fixed vs TERM3M) or 'term_basis'
        (TERM6M vs TERM3M + spread).
    terms : list of str
        Tenors such as '18M' or '5Y'.
    collateral : Index or None, optional
        CSA collateral of the IRS, OIS and XCCY quotes (SOFR by default; None: no CSA).
    """
    tmpl = _template(kind)
    out = []
    for label in terms:
        if kind in ("term_irs", "term_basis"):
            base = dataclasses.replace(tmpl, term=_term(label, US_ADJ))
        else:
            base = dataclasses.replace(
                tmpl,
                term=_term(label, tmpl.adj_convention),
                collateral_index=collateral,
            )
        if kind == "xccy":
            out.append(
                _at_par(
                    lambda x, b=base: dataclasses.replace(
                        b, receive_leg=_with_spread(b.receive_leg, x)
                    ),
                    0.0,
                )
            )
        elif kind == "term_basis":
            out.append(
                _at_par(
                    lambda x, b=base: dataclasses.replace(
                        b, pay_leg=_with_spread(b.pay_leg, x)
                    ),
                    8e-4,
                )
            )
        else:
            out.append(
                _at_par(
                    lambda x, b=base: dataclasses.replace(
                        b, fixed_leg=_with_rate(b.fixed_leg, x)
                    ),
                    0.05,
                )
            )
    return out
