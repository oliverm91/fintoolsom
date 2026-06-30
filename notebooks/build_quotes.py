from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

from fintoolsom.dates import Calendar, ModifiedFollowingConvention, TF_Actual360
from fintoolsom.dates.date_counts import ActualDayCountConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.dates.time_fractions import TimeFractionBase
from fintoolsom.market import (
    BasisPoints,
    Currency,
    CurrencyPair,
    FX_Rate,
    FixedLegSpec,
    FloatingLegSpec,
    ForwardPointsQuote,
    InstrumentQuote,
    IRSQuote,
    CrossCurrencyFloatFloatQuote,
    PaymentFrequency,
    QuotedSide,
)
from fintoolsom.market.index import InterestPriceIndex, RateIndex
from fintoolsom.market.localities import Locality
from fintoolsom.rates import Rate, RateConvention, LinearInterestConvention, CompoundedInterestConvention


# ── Lookup tables ────────────────────────────────────────────────────────────

_CURRENCY: dict[str, Currency] = {
    "USD": Currency.USD,
    "CLP": Currency.CLP,
    "EUR": Currency.EUR,
    "GBP": Currency.GBP,
}

_INDEX: dict[str, RateIndex | InterestPriceIndex] = {
    "SOFR":   RateIndex("SOFR",   currency=Currency.USD),
    "CAMARA": RateIndex("CAMARA", currency=Currency.CLP),
    "ICP":    InterestPriceIndex("ICP", currency=Currency.CLP),
}

_DAY_COUNT: dict[str, TimeFractionBase] = {
    "TF_Actual360": TF_Actual360(),
}

_RATE_DC_BASE: dict[str, tuple] = {
    "TF_Actual360": (ActualDayCountConvention, 360),
}

_INTEREST_CONVENTION = {
    "Linear":     LinearInterestConvention,
    "Compounded": CompoundedInterestConvention,
}


# ── Private helpers ──────────────────────────────────────────────────────────

def _build_adj_conv(conv_name: str, cal_name: str):
    cal = Calendar(cal_name)
    if conv_name == "ModifiedFollowingConvention":
        return ModifiedFollowingConvention(cal)
    raise ValueError(f"Unknown adj_convention: {conv_name!r}")


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _fixed_spec(leg: dict, adj_conv, tf: TimeFractionBase, dc_key: str) -> FixedLegSpec:
    dcc, base = _RATE_DC_BASE[dc_key]
    interest_conv = _INTEREST_CONVENTION[leg.get("interest_convention", "Linear")]
    rate_conv = RateConvention(interest_conv, dcc, base)
    return FixedLegSpec(
        currency=_CURRENCY[leg["currency"]],
        payment_frequency=PaymentFrequency[leg["payment_frequency"]],
        adj_convention=adj_conv,
        time_fraction=tf,
        rate=Rate(rate_conv, leg["rate"]),
    )


def _float_spec(leg: dict, adj_conv, tf: TimeFractionBase) -> FloatingLegSpec:
    index = _INDEX[leg["index"]]
    currency_str = leg.get("currency")
    currency = _CURRENCY[currency_str] if currency_str else index.currency
    spread_bps = leg.get("spread_bps")
    return FloatingLegSpec(
        currency=currency,
        payment_frequency=PaymentFrequency[leg["payment_frequency"]],
        adj_convention=adj_conv,
        time_fraction=tf,
        index=index,
        spread=BasisPoints(spread_bps) if spread_bps is not None else None,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def build_quotes(
    quote_date: date,
    quotes_file_path: str = "quotes.json",
) -> list[InstrumentQuote]:
    path = Path(quotes_file_path)
    with open(path) as f:
        groups: list[dict] = json.load(f)

    result: list[InstrumentQuote] = []

    for group in groups:
        quote_type: str = group["quoteType"]
        adj_conv = _build_adj_conv(group["term_adj_convention"], group["term_calendar"])

        dc_key: str | None = group.get("day_count_convention")
        tf: TimeFractionBase | None = _DAY_COUNT[dc_key] if dc_key else None

        spot_lag: int = group.get("spot_lag", 2)
        stub_first: bool = group.get("stub_first", True)
        long_stub: bool = group.get("long_stub", False)
        group_collateral_str: str | None = group.get("collateral_index")

        for q in group["quotes"]:
            term = Term(
                value=q["term"]["value"],
                unit=TermUnit[q["term"]["unit"]],
                adj_convention=adj_conv,
            )
            collateral_str = q.get("collateral_index", group_collateral_str)
            collateral = _INDEX[collateral_str] if collateral_str else None

            if quote_type == "IRSQuote":
                fixed_data = _deep_merge(group.get("fixed_leg", {}), q.get("fixed_leg", {}))
                float_data = _deep_merge(group.get("floating_leg", {}), q.get("floating_leg", {}))
                result.append(IRSQuote(
                    quoted_side=QuotedSide[group["quoted_side"]],
                    term=term,
                    quote_date=quote_date,
                    adj_convention=adj_conv,
                    spot_lag=spot_lag,
                    stub_first=stub_first,
                    long_stub=long_stub,
                    collateral_index=collateral,
                    fixed_leg=_fixed_spec(fixed_data, adj_conv, tf, dc_key),
                    floating_leg=_float_spec(float_data, adj_conv, tf),
                ))

            elif quote_type == "CrossCurrencyFloatFloatQuote":
                recv_data = _deep_merge(group.get("receive_leg", {}), q.get("receive_leg", {}))
                pay_data = _deep_merge(group.get("pay_leg", {}), q.get("pay_leg", {}))
                result.append(CrossCurrencyFloatFloatQuote(
                    quoted_side=QuotedSide[group["quoted_side"]],
                    term=term,
                    quote_date=quote_date,
                    adj_convention=adj_conv,
                    spot_lag=spot_lag,
                    stub_first=stub_first,
                    long_stub=long_stub,
                    collateral_index=collateral,
                    receive_leg=_float_spec(recv_data, adj_conv, tf),
                    pay_leg=_float_spec(pay_data, adj_conv, tf),
                ))

            elif quote_type == "ForwardPointsQuote":
                cp_arr: list[str] = group["currency_pair"]
                cp = CurrencyPair(_CURRENCY[cp_arr[0]], _CURRENCY[cp_arr[1]])
                spot = FX_Rate(cp, group["spot"])
                locality_str: str | None = group.get("locality")
                result.append(ForwardPointsQuote(
                    currency_pair=cp,
                    value=q["value"],
                    is_buy=group["is_buy"],
                    spot=spot,
                    points_divisor=group.get("points_divisor", 1),
                    term=term,
                    quote_date=quote_date,
                    spot_lag=spot_lag,
                    payment_lag=group.get("payment_lag", 1),
                    locality=Locality[locality_str] if locality_str else None,
                ))

    return result
