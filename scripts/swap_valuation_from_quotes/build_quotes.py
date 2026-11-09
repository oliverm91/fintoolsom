"""Builds the fintoolsom Market inputs and the curve quotes from the Bloomberg quotes in ``fin_data.financial_data``.

Quote families (all rates lin/act/360):

=========  ==========================================  ====================================================
Ticker     Instrument                                  Conventions
=========  ==========================================  ====================================================
CHSWP*     CLP fixed vs ICP (camara) OIS               semiannual from 1Y, zero-coupon below; SOFR collateral
                                                       from ``collateral_cutoff_months`` on, none before
USOSFR*    USD fixed vs SOFR OIS                       annual from 18M, zero-coupon below; SOFR collateral
CPXOSS*    USD/CLP xccy basis, ICP + spread vs SOFR    semiannual, SOFR collateral, spread in bp on the ICP leg
CHN*       USDCLP NDF forward points                   uncollateralised, settles on ``settle_dt``
USDCLP     USDCLP spot                                 mid
CLICP      ICP level series (PX_LAST)                  ICP history: realised part of the running coupons and
                                                       the short end of the ICP curve
SOFRRATE   SOFR fixings
=========  ==========================================  ====================================================

Run ``run_valuation.py`` to use it.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd

from fintoolsom.dates import ActualDayCountConvention, ModifiedFollowingConvention
from fintoolsom.dates import schedules as _schedules
from fintoolsom.dates.time_fractions import TF_Actual360
from fintoolsom.derivatives.swaps import legs as _swap_legs
from fintoolsom.market import (
    Currency,
    CurrencyPair,
    CrossCurrencyFloatFloatQuote,
    FloatingLegSpec,
    FixedLegSpec,
    ForwardPointsQuote,
    FX_Rate,
    IRSQuote,
    Market,
    OvernightInterestPriceHistory,
    OvernightInterestPriceIndex,
    OvernightRateHistory,
    OvernightRateIndex,
    PaymentFrequency,
    QuotedSide,
    Term,
    TermUnit,
)
from fintoolsom.rates import LinearInterestConvention, Rate, RateConvention

from local_db import read_sql

# --------------------------------------------------------------------------- shared conventions
LIN_ACT_360 = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)
TIME_FRACTION = TF_Actual360()
CL_ADJ = ModifiedFollowingConvention("CL")
US_ADJ = ModifiedFollowingConvention("US")
ICP = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)
USDCLP = CurrencyPair(Currency.USD, Currency.CLP)

SEMIANNUAL, ANNUAL = PaymentFrequency.SEMIANNUAL, PaymentFrequency.ANNUAL
ZERO_COUPON = SimpleNamespace(
    value="ZERO"
)  # stands in for a PaymentFrequency, see _enable_zero_coupon_frequency

ICP_FIRST_COUPON_MONTHS = 12  # shorter CHSWP quotes are zero coupon
SOFR_FIRST_COUPON_MONTHS = 18  # shorter USOSFR quotes are zero coupon
STUB_FIRST = True  # the first coupon is the short one

FIXINGS_MARGIN_DAYS = 7  # calendar days of fixings kept before the swap start (or the valuation date, if earlier)


def fixings_start(swap_start: date, t: date) -> date:
    """First date of the ICP and SOFR histories. The running coupons read fixings from the swap start, the short end of the
    curves reads the latest ones, so it is the earlier of the two minus a margin."""
    return min(swap_start, t) - timedelta(days=FIXINGS_MARGIN_DAYS)


def rate(value: float) -> Rate:
    return Rate(LIN_ACT_360, value)


# --------------------------------------------------------------------------- zero-coupon workaround
def _enable_zero_coupon_frequency() -> None:
    """Lets a leg use the frequency ``ZERO_COUPON`` (one period from start to maturity).

    fintoolsom 1.0.0b2 has no zero-coupon payment frequency: ``PaymentFrequency`` and the leg map do not list it and
    ``ScheduleGenerator`` loops until its iteration limit when given ``"0m"``. Remove this once fintoolsom supports it.
    """
    _swap_legs._FREQ_TO_TENOR["ZERO"] = "0m"
    original = _schedules.ScheduleGenerator.generate_schedule

    def generate_schedule(
        start_date,
        maturity_tenor,
        frequency_tenor,
        adj_conv,
        maturity_adj_conv=None,
        end_date=None,
        stub_first=True,
        long_stub=True,
    ):
        if str(frequency_tenor).lower() in ("0", "0m", "0y"):
            if end_date is not None:
                return [start_date, end_date]
            adj = maturity_adj_conv or adj_conv
            return [
                start_date,
                adj.adjust(
                    _schedules.Tenor(maturity_tenor, adj).get_unadjusted_maturity(
                        start_date
                    )
                ),
            ]
        return original(
            start_date,
            maturity_tenor,
            frequency_tenor,
            adj_conv,
            maturity_adj_conv,
            end_date,
            stub_first,
            long_stub,
        )

    _schedules.ScheduleGenerator.generate_schedule = staticmethod(generate_schedule)


_enable_zero_coupon_frequency()


# --------------------------------------------------------------------------- data access (local snapshot of fin_data)
def _get_quotes_table(t: date) -> pd.DataFrame:
    """One row per ticker with a column per field (px_bid, px_ask, maturity, sw_eff_dt, settle_dt...) on ``t``."""
    raw = read_sql(
        """SELECT REPLACE(ticker,' Curncy','') AS ticker, field, value FROM financial_data
           WHERE date=:d AND (ticker LIKE 'CHSWP%' OR ticker LIKE 'USOSFR%' OR ticker LIKE 'CPXOSS%'
                              OR ticker LIKE 'CHN%' OR ticker LIKE 'USDCLP%')""",
        d=t.isoformat(),
    )
    if raw.empty:
        raise ValueError(
            f"No quotes in the local snapshot for {t}. Re-run export_snapshot.py for that date."
        )
    return raw.pivot_table(
        index="ticker", columns="field", values="value", aggfunc="first"
    )


def _get_fixings(start: date, t: date) -> pd.DataFrame:
    fixings = read_sql(
        """SELECT date, ticker, value FROM financial_data
           WHERE field='px_last' AND date BETWEEN :d0 AND :d1 AND (ticker LIKE 'CLICP%' OR ticker LIKE 'SOFRRATE%')""",
        d0=start.isoformat(),
        d1=t.isoformat(),
    )
    fixings["date"] = pd.to_datetime(fixings["date"]).dt.date
    fixings["value"] = fixings["value"].astype(float)
    return fixings.sort_values("date")


# --------------------------------------------------------------------------- tickers
def _tenor(code: str) -> tuple[int, str]:
    """Bloomberg tenor suffix to (value, unit): 1Z/2Z weeks, A-F/I months (A=1M..F=6M, I=9M), 1F 18M, digits years."""
    letters = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "I": 9}
    if code in ("1Z", "2Z"):
        return int(code[0]), "W"
    if code == "1F":
        return 18, "M"
    if code in letters:
        return letters[code], "M"
    return int(code), "Y"


def _months(value: int, unit: str) -> float:
    return {"W": 0.25, "M": 1, "Y": 12}[unit] * value


def _term(value: int, unit: str, adj) -> Term:
    return Term(value, TermUnit(unit), adj)


# --------------------------------------------------------------------------- public API
@dataclass
class MarketAndQuotes:
    market: Market
    quotes: list
    description: list[tuple[str, str]]  # (ticker, what it was built as)


def get_market_and_quotes(
    t: date,
    fixings_from: date,
    collateral_cutoff_months: int = 24,
    drop_tickers: tuple[str, ...] = ("CHN2Y",),
    live: bool = False,
) -> MarketAndQuotes:
    """Market (ICP and SOFR fixings, USDCLP spot, risk-free index ICP) and the quotes to build its curves on ``t``.

    Parameters
    ----------
    fixings_from : first date of the ICP and SOFR histories, see ``fixings_start``.
    live : read the quotes (bdp) and the ICP and SOFR histories (bdh) from Bloomberg, see ``live_data.py``, instead of the
        local snapshot. ``t`` must then be today.
    collateral_cutoff_months : CHSWP quotes of this tenor or longer are SOFR collateralised, shorter ones are not.
    drop_tickers : quotes left out. CHN2Y shares its maturity with CPXOSS2, which owns that pillar.
    """
    if live:
        from live_data import (
            get_live_fixings,
            get_live_quotes_table,
        )  # imported only when asked for: needs PropTools and Bloomberg

        table, fixings = get_live_quotes_table(), get_live_fixings(fixings_from, t)
    else:
        table, fixings = _get_quotes_table(t), _get_fixings(fixings_from, t)

    def mid(ticker: str) -> float:
        return (
            float(table.loc[ticker, "px_bid"]) + float(table.loc[ticker, "px_ask"])
        ) / 2

    def has_prices(ticker: str) -> bool:
        return not (
            pd.isna(table.loc[ticker].get("px_bid"))
            or pd.isna(table.loc[ticker].get("px_ask"))
        )

    def as_date(ticker: str, field: str) -> date:
        return pd.to_datetime(table.loc[ticker, field]).date()

    # Market: fixings and spot
    sofr_fixings = fixings[fixings.ticker.str.startswith("SOFRRATE")]
    icp_levels = fixings[fixings.ticker.str.startswith("CLICP")]
    if icp_levels.empty or icp_levels.date.max() < t:
        source = (
            "Bloomberg"
            if live
            else "the local snapshot (re-run export_snapshot.py for that date)"
        )
        raise ValueError(f"No CLICP level for {t} in {source}.")
    sofr_history = OvernightRateHistory(
        SOFR, {r.date: rate(r.value / 100) for r in sofr_fixings.itertuples()}
    )
    icp_history = OvernightInterestPriceHistory(
        ICP, {r.date: r.value for r in icp_levels.itertuples()}
    )

    market = Market(t, risk_free_index=ICP)
    market.add_index(icp_history)
    market.add_index(sofr_history)
    market.add_fx_rate(t, FX_Rate(USDCLP, mid("USDCLP")))

    # Quotes
    quotes, description = [], []
    for ticker in sorted(table.index):
        swap = re.match(r"(CHSWP|USOSFR|CPXOSS)(\w+)$", ticker)
        forward = re.match(r"CHN(\d+)([WMY])$", ticker)
        if ticker in drop_tickers or not (swap or forward):
            continue
        if swap:
            family, code = swap.groups()
            if not has_prices(ticker):
                continue
            value, unit = (int(code), "Y") if family == "CPXOSS" else _tenor(code)
            maturity = as_date(ticker, "maturity")

        if swap and family == "CHSWP":
            frequency = (
                SEMIANNUAL
                if _months(value, unit) >= ICP_FIRST_COUPON_MONTHS
                else ZERO_COUPON
            )
            collateral = (
                SOFR if _months(value, unit) >= collateral_cutoff_months else None
            )
            quotes.append(
                IRSQuote(
                    quoted_side=QuotedSide.RECEIVE,
                    term=_term(value, unit, CL_ADJ),
                    quote_date=t,
                    adj_convention=CL_ADJ,
                    maturity_date=maturity,
                    collateral_index=collateral,
                    stub_first=STUB_FIRST,
                    fixed_leg=FixedLegSpec(
                        Currency.CLP,
                        frequency,
                        CL_ADJ,
                        TIME_FRACTION,
                        rate(mid(ticker) / 100),
                    ),
                    floating_leg=FloatingLegSpec(
                        Currency.CLP, frequency, CL_ADJ, TIME_FRACTION, ICP, rate(0.0)
                    ),
                )
            )
            description.append(
                (
                    ticker,
                    f"CLP fixed vs ICP {value}{unit}, collateral {'SOFR' if collateral else 'none'}",
                )
            )
        elif swap and family == "USOSFR":
            frequency = (
                ANNUAL
                if _months(value, unit) >= SOFR_FIRST_COUPON_MONTHS
                else ZERO_COUPON
            )
            quotes.append(
                IRSQuote(
                    quoted_side=QuotedSide.RECEIVE,
                    term=_term(value, unit, US_ADJ),
                    quote_date=t,
                    adj_convention=US_ADJ,
                    maturity_date=maturity,
                    collateral_index=SOFR,
                    stub_first=STUB_FIRST,
                    fixed_leg=FixedLegSpec(
                        Currency.USD,
                        frequency,
                        US_ADJ,
                        TIME_FRACTION,
                        rate(mid(ticker) / 100),
                    ),
                    floating_leg=FloatingLegSpec(
                        Currency.USD, frequency, US_ADJ, TIME_FRACTION, SOFR, rate(0.0)
                    ),
                )
            )
            description.append(
                (ticker, f"USD fixed vs SOFR {value}{unit}, collateral SOFR")
            )
        elif swap and family == "CPXOSS":
            quotes.append(
                CrossCurrencyFloatFloatQuote(
                    quoted_side=QuotedSide.RECEIVE,
                    term=_term(value, unit, CL_ADJ),
                    quote_date=t,
                    adj_convention=CL_ADJ,
                    maturity_date=maturity,
                    collateral_index=SOFR,
                    stub_first=STUB_FIRST,
                    receive_leg=FloatingLegSpec(
                        Currency.CLP,
                        SEMIANNUAL,
                        CL_ADJ,
                        TIME_FRACTION,
                        ICP,
                        rate(mid(ticker) / 1e4),
                    ),
                    pay_leg=FloatingLegSpec(
                        Currency.USD, SEMIANNUAL, CL_ADJ, TIME_FRACTION, SOFR, rate(0.0)
                    ),
                )
            )
            description.append(
                (
                    ticker,
                    f"xccy ICP{mid(ticker):+.1f}bp vs SOFR {value}{unit}, collateral SOFR",
                )
            )
        elif forward:
            settle = as_date(ticker, "settle_dt")
            quotes.append(
                ForwardPointsQuote(
                    currency_pair=USDCLP,
                    value=mid(ticker),
                    is_buy=True,
                    payment_date=settle,
                    fixing_date=CL_ADJ.calendar.add_business_days(settle, -1),
                    quote_date=t,
                )
            )
            description.append(
                (ticker, f"USDCLP NDF points {mid(ticker):+.3f}, uncollateralised")
            )

    return MarketAndQuotes(market, quotes, description)
