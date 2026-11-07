"""Real quote set (CLP/USD, 2026-06-29) and the market built from it.

Port of ``notebooks/build_quotes.py`` reading ``tests/data/quotes_20260629.json``:
CLP fixed vs ICP and USD fixed vs SOFR OIS, USD/CLP forward points and ICP/SOFR
cross-currency basis swaps. Unlike the notebook loader, it reads the quoted
``spread_bps`` and uses one ICP index (the market's riskless index) everywhere.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

from fintoolsom.curve_builder import build_curves
from fintoolsom.dates import ActualDayCountConvention, Calendar, ModifiedFollowingConvention, TF_Actual360
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import (
    CrossCurrencyFloatFloatQuote,
    Currency,
    CurrencyPair,
    FixedLegSpec,
    FloatingLegSpec,
    ForwardPointsQuote,
    InstrumentQuote,
    IRSQuote,
    Market,
    OvernightInterestPriceIndex,
    OvernightRateIndex,
    PaymentFrequency,
    QuotedSide,
)
from fintoolsom.market.currencies import FX_Rate, FX_RateData
from fintoolsom.market.index_history import OvernightInterestPriceHistory, OvernightRateHistory
from fintoolsom.market.localities import Locality
from fintoolsom.rates import CompoundedInterestConvention, LinearInterestConvention, Rate, RateConvention

QUOTES_FILE = Path(__file__).resolve().parent.parent / "data" / "quotes_20260629.json"
QUOTE_DATE = date(2026, 6, 29)
USDCLP_SPOT = 900.0
SOFR_FIXING = 0.0525

SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)
ICP = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
USDCLP = CurrencyPair(Currency.USD, Currency.CLP)

_CURRENCY = {"USD": Currency.USD, "CLP": Currency.CLP}
_INDEX = {"SOFR": SOFR, "ICP": ICP}
_INTEREST = {"Linear": LinearInterestConvention, "Compounded": CompoundedInterestConvention}
_TIME_FRACTION = {"TF_Actual360": (TF_Actual360(), ActualDayCountConvention, 360)}


def _merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def _fixed_spec(leg: dict, adj, tf_key: str) -> FixedLegSpec:
    tf, dcc, base = _TIME_FRACTION[tf_key]
    convention = RateConvention(_INTEREST[leg.get("interest_convention", "Linear")], dcc, base)
    return FixedLegSpec(
        currency=_CURRENCY[leg["currency"]],
        payment_frequency=PaymentFrequency[leg["payment_frequency"]],
        adj_convention=adj,
        time_fraction=tf,
        rate=Rate(convention, leg["rate"]),
    )


def _float_spec(leg: dict, adj, tf_key: str) -> FloatingLegSpec:
    index = _INDEX[leg["index"]]
    spread = Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 360), leg.get("spread_bps", 0.0) / 10_000)
    return FloatingLegSpec(
        currency=_CURRENCY[leg["currency"]] if "currency" in leg else index.currency,
        payment_frequency=PaymentFrequency[leg["payment_frequency"]],
        adj_convention=adj,
        time_fraction=_TIME_FRACTION[tf_key][0],
        index=index,
        spread=spread,
    )


def load_quotes(quote_date: date = QUOTE_DATE, path: Path = QUOTES_FILE) -> list[InstrumentQuote]:
    """Every quote of the JSON file, as InstrumentQuote objects."""
    groups = json.loads(path.read_text(encoding="utf-8"))
    quotes: list[InstrumentQuote] = []
    for group in groups:
        if group["term_adj_convention"] != "ModifiedFollowingConvention":
            raise ValueError(f"Unsupported adjustment {group['term_adj_convention']!r}.")
        adj = ModifiedFollowingConvention(Calendar(group["term_calendar"]))
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
                quotes.append(IRSQuote(
                    quoted_side=QuotedSide[group["quoted_side"]], term=term, collateral_index=collateral,
                    fixed_leg=_fixed_spec(_merge(group["fixed_leg"], q.get("fixed_leg", {})), adj, tf_key),
                    floating_leg=_float_spec(_merge(group["floating_leg"], q.get("floating_leg", {})), adj, tf_key),
                    **common,
                ))
            elif kind == "CrossCurrencyFloatFloatQuote":
                quotes.append(CrossCurrencyFloatFloatQuote(
                    quoted_side=QuotedSide[group["quoted_side"]], term=term, collateral_index=collateral,
                    receive_leg=_float_spec(_merge(group["receive_leg"], q.get("receive_leg", {})), adj, tf_key),
                    pay_leg=_float_spec(_merge(group["pay_leg"], q.get("pay_leg", {})), adj, tf_key),
                    **common,
                ))
            elif kind == "ForwardPointsQuote":
                base, quote_ccy = group["currency_pair"]
                quotes.append(ForwardPointsQuote(
                    currency_pair=CurrencyPair(_CURRENCY[base], _CURRENCY[quote_ccy]),
                    value=q["value"], is_buy=group["is_buy"],
                    points_divisor=group.get("points_divisor", 1),
                    term=term, quote_date=quote_date,
                    spot_lag=group.get("spot_lag", 2), payment_lag=group.get("payment_lag", 1),
                    locality=Locality[group["locality"]] if group.get("locality") else None,
                ))
            else:
                raise ValueError(f"Unsupported quote type {kind!r}.")
    return quotes


def quote_market(t: date = QUOTE_DATE) -> Market:
    """Market data the quotes need: SOFR fixing, ICP level, USD/CLP spot; no curves."""
    sofr_history = OvernightRateHistory(
        SOFR, {t: Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 360), SOFR_FIXING)}
    )
    icp_history = OvernightInterestPriceHistory(ICP, {t: 100.0})
    return Market(
        t,
        fx_history={USDCLP: FX_RateData(USDCLP, {t: FX_Rate(USDCLP, USDCLP_SPOT)})},
        indexes_history={"SOFR": sofr_history, "ICP": icp_history},
        riskless_index=ICP,
    )


def built_market(quotes: list[InstrumentQuote] | None = None) -> tuple[Market, list[InstrumentQuote]]:
    """Market with every curve bootstrapped from `quotes` (the full JSON set by default)."""
    quotes = load_quotes() if quotes is None else quotes
    market = quote_market(quotes[0].quote_date)
    build_curves(quotes, market)
    return market, quotes
