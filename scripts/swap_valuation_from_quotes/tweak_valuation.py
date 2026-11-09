"""Scratch space for tweaking the swap valuation and seeing how the legs' MTMs move against the stored ones."""

import argparse
import csv
import re
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from datetime import date
from heapq import nsmallest
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Literal, get_args

import pandas as pd

from fintoolsom.curve_builder import (
    InsufficientQuotesError,
    RepricingError,
    build_curves,
)
from fintoolsom.dates import Term, TermUnit
from fintoolsom.derivatives import Calculator
from fintoolsom.derivatives.swaps import LegSide
from fintoolsom.market import (
    CrossCurrencyFloatFloatQuote,
    Currency,
    FixedLegSpec,
    FloatingLegSpec,
    ForwardPointsQuote,
    IRSQuote,
    QuotedSide,
)
from fintoolsom.rates import InterpolationMethod

from build_quotes import (
    ANNUAL,
    CL_ADJ,
    ICP,
    SEMIANNUAL,
    SOFR,
    TIME_FRACTION,
    US_ADJ,
    USDCLP,
    ZERO_COUPON,
    _get_quotes_table,
    _months,
    _tenor,
    _term,
    fixings_start,
    get_market_and_quotes,
    rate,
)
from build_swap import get_swap
from stored_values import STORED_MTM, stored_mtm

# Which coupon is short when the tenor is not a whole number of periods (18M yearly: 6M + 12M or 12M + 6M).
StubPosition = Literal["first", "last"]


@dataclass
class ValuationConfig:
    """Valuation settings to tweak."""

    index_curve_interpolation_method: InterpolationMethod
    discount_curve_interpolation_method: InterpolationMethod
    sofr_stub: StubPosition
    icp_stub: StubPosition
    basis_stub: StubPosition
    sofr_first_non_zero_coupon_term: Term
    icp_first_non_zero_coupon_term: Term
    basis_first_non_zero_coupon_term: Term
    basis_first_quote: Term  # FX forwards are not taken into account from this quote on
    icp_first_sofr_collateralized_term: Term


def _terms(tenors: str, adj) -> tuple[Term, ...]:
    """``"1M 18M 2Y"`` as Terms."""
    return tuple(Term(int(s[:-1]), TermUnit(s[-1]), adj) for s in tenors.split())


_INDEX_METHODS = (
    #InterpolationMethod.CubicSpline,
    #InterpolationMethod.Linear,
    InterpolationMethod.PiecewiseConstant,
)
_DISCOUNT_METHODS = (
    #InterpolationMethod.CubicSpline,
    #InterpolationMethod.HermiteCubicSpline,
    InterpolationMethod.LogLinear,
)

# Tenors with bid and ask in the snapshot.
_SOFR_TERMS = _terms(
    "18M",
    US_ADJ,
)
_ICP_TERMS = _terms(
    "1Y 18M 2Y 3Y", CL_ADJ
)
_ICP_FIRST_COL = _terms(
    "1Y 18M 2Y", CL_ADJ
)
_BASIS_FIRST_QUOTE_TERMS = _terms("2Y", CL_ADJ)  # where the FX forwards stop
_BASIS_NON_ZERO_TERMS = _terms("2Y", CL_ADJ)


class ValuationConfigOptions:
    """Every value each field of :class:`ValuationConfig` can take, under the same names."""

    index_curve_interpolation_method = _INDEX_METHODS
    discount_curve_interpolation_method = _DISCOUNT_METHODS
    sofr_stub = get_args(StubPosition)
    icp_stub = get_args(StubPosition)
    basis_stub = get_args(StubPosition)
    sofr_first_non_zero_coupon_term = _SOFR_TERMS
    icp_first_non_zero_coupon_term = _ICP_TERMS
    basis_first_non_zero_coupon_term = _BASIS_NON_ZERO_TERMS
    basis_first_quote = _BASIS_FIRST_QUOTE_TERMS
    icp_first_sofr_collateralized_term = _ICP_FIRST_COL


def _term_months(term: Term) -> float:
    return _months(term.value, term.unit.value)


_SHORTEST_BASIS_QUOTE_MONTHS = 24
_LAST_IRREGULAR_SOFR_MONTHS = 18  # 18M yearly is the only SOFR swap with a short coupon


def _is_logical(config: ValuationConfig) -> bool:
    """False for a combination that can only repeat the valuation of another one.

    - SOFR stub: it only matters if the 18M swap has coupons.
    - ICP stub: every semiannual ICP swap is a whole number of periods.
    - Basis first non-zero term: all the terms up to the first basis swap used make every basis swap periodic.
    - ICP collateral below the first non-zero term: a single-payment swap is worth the same under any collateral.

    The basis stub is not filtered: it moves the MTM even though the basis tenors are whole numbers of periods. Neither is
    a ``basis_first_quote`` below the shortest basis quote: it uses the same basis swaps but stops the FX forwards earlier.
    """
    sofr_has_stub = (
        _term_months(config.sofr_first_non_zero_coupon_term)
        <= _LAST_IRREGULAR_SOFR_MONTHS
    )
    first_basis_swap = max(
        _term_months(config.basis_first_quote), _SHORTEST_BASIS_QUOTE_MONTHS
    )
    return (
        (sofr_has_stub or config.sofr_stub == "first")
        and config.icp_stub == "first"
        and _term_months(config.basis_first_non_zero_coupon_term) >= first_basis_swap
        and _term_months(config.icp_first_sofr_collateralized_term)
        >= _term_months(config.icp_first_non_zero_coupon_term)
    )


def all_valuation_configs() -> Iterator[ValuationConfig]:
    """Yields the combinations of the values in :class:`ValuationConfigOptions` that pass :func:`_is_logical`."""
    names = [f.name for f in fields(ValuationConfig)]
    options = [getattr(ValuationConfigOptions, name) for name in names]
    for values in product(*options):
        config = ValuationConfig(**dict(zip(names, values)))
        if _is_logical(config):
            yield config


def get_quotes(
    t: date, config: ValuationConfig
) -> tuple[list, tuple[InterpolationMethod, InterpolationMethod]]:
    """Quotes of ``t`` from ``local_data.sqlite`` built under ``config``, and its ``(index, discount)`` interpolation methods.

    Below its ``*_first_non_zero_coupon_term`` a swap is zero coupon. Basis swaps shorter than ``basis_first_quote`` and FX
    forwards from that tenor on are left out.
    """
    table = _get_quotes_table(t)

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

    def frequency(months: float, first_periodic: Term, periodic):
        return periodic if months >= _term_months(first_periodic) else ZERO_COUPON

    quotes: list = []
    for ticker in sorted(table.index):
        swap = re.match(r"(CHSWP|USOSFR|CPXOSS)(\w+)$", ticker)
        forward = re.match(r"CHN(\d+)([WMY])$", ticker)
        if forward:
            months = _months(int(forward[1]), forward[2])
            if months < _term_months(config.basis_first_quote):
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
            continue
        if not swap or not has_prices(ticker):
            continue

        family, code = swap.groups()
        value, unit = (int(code), "Y") if family == "CPXOSS" else _tenor(code)
        months = _months(value, unit)
        swap_fields = dict(
            quoted_side=QuotedSide.RECEIVE,
            quote_date=t,
            maturity_date=as_date(ticker, "maturity"),
        )
        if family == "CHSWP":
            freq = frequency(months, config.icp_first_non_zero_coupon_term, SEMIANNUAL)
            first_collateralized = _term_months(
                config.icp_first_sofr_collateralized_term
            )
            quotes.append(
                IRSQuote(
                    **swap_fields,
                    term=_term(value, unit, CL_ADJ),
                    adj_convention=CL_ADJ,
                    collateral_index=SOFR if months >= first_collateralized else None,
                    stub_first=config.icp_stub == "first",
                    fixed_leg=FixedLegSpec(
                        Currency.CLP,
                        freq,
                        CL_ADJ,
                        TIME_FRACTION,
                        rate(mid(ticker) / 100),
                    ),
                    floating_leg=FloatingLegSpec(
                        Currency.CLP, freq, CL_ADJ, TIME_FRACTION, ICP, rate(0.0)
                    ),
                )
            )
        elif family == "USOSFR":
            freq = frequency(months, config.sofr_first_non_zero_coupon_term, ANNUAL)
            quotes.append(
                IRSQuote(
                    **swap_fields,
                    term=_term(value, unit, US_ADJ),
                    adj_convention=US_ADJ,
                    collateral_index=SOFR,
                    stub_first=config.sofr_stub == "first",
                    fixed_leg=FixedLegSpec(
                        Currency.USD,
                        freq,
                        US_ADJ,
                        TIME_FRACTION,
                        rate(mid(ticker) / 100),
                    ),
                    floating_leg=FloatingLegSpec(
                        Currency.USD, freq, US_ADJ, TIME_FRACTION, SOFR, rate(0.0)
                    ),
                )
            )
        elif months >= _term_months(config.basis_first_quote):
            freq = frequency(
                months, config.basis_first_non_zero_coupon_term, SEMIANNUAL
            )
            quotes.append(
                CrossCurrencyFloatFloatQuote(
                    **swap_fields,
                    term=_term(value, unit, CL_ADJ),
                    adj_convention=CL_ADJ,
                    collateral_index=SOFR,
                    stub_first=config.basis_stub == "first",
                    receive_leg=FloatingLegSpec(
                        Currency.CLP,
                        freq,
                        CL_ADJ,
                        TIME_FRACTION,
                        ICP,
                        rate(mid(ticker) / 1e4),
                    ),
                    pay_leg=FloatingLegSpec(
                        Currency.USD, freq, CL_ADJ, TIME_FRACTION, SOFR, rate(0.0)
                    ),
                )
            )

    methods = (
        config.index_curve_interpolation_method,
        config.discount_curve_interpolation_method,
    )
    return quotes, methods


def compare_leg_mtms(t: date, config: ValuationConfig) -> None:
    """Prints the MTM of each leg on ``t`` under ``config`` next to its value in ``stored_values.STORED_MTM``."""
    raise NotImplementedError


SWAP_ID = 10
GRID_CSV = Path(__file__).with_name("valuation_grid.csv")
_CONFIG_COLUMNS = [f.name for f in fields(ValuationConfig)]
_GRID_COLUMNS = [
    "config",
    "date",
    *_CONFIG_COLUMNS,
    "status",
    "receive_pv",
    "pay_pv",
    "mtm",
    "stored_receive_pv",
    "stored_pay_pv",
    "stored_mtm",
    "diff_receive_pv",
    "diff_pay_pv",
    "diff_mtm",
    "seconds",
    "error",
]
_BUILD_ERRORS = (InsufficientQuotesError, RepricingError, ValueError)
GRID_TOP_SIZE = 15

# One per process, so each worker loads the swap and the markets only once.
_state: dict = {}


def _load_grid_state() -> None:
    booked = get_swap(SWAP_ID, collateral=1)
    _state["booked"] = booked
    _state["markets"] = {
        t: get_market_and_quotes(t, fixings_start(booked.start_date, t)).market
        for t in sorted({t for t, _ in STORED_MTM})
    }


def _config_cell(value) -> str:
    if isinstance(value, Term):
        return f"{value.value}{value.unit.value}"
    return getattr(value, "name", value)


def _value_task(task: tuple[int, date, ValuationConfig]) -> dict:
    """Builds the curves of one (config, date) and values the swap. A build that fails is a result, not an error."""
    number, t, config = task
    started = perf_counter()
    booked, market = _state["booked"], _state["markets"][t]
    market.curves.clear()  # curves left in the market would be extended, not rebuilt
    row = dict.fromkeys(_GRID_COLUMNS, "")
    row.update(config=number, date=t.isoformat(), status="ok")
    row.update({c: _config_cell(getattr(config, c)) for c in _CONFIG_COLUMNS})
    try:
        quotes, (index_method, discount_method) = get_quotes(t, config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build_curves(
                quotes,
                market,
                discount_interp_method=discount_method,
                index_interp_method=index_method,
            )
    except _BUILD_ERRORS as exc:
        row.update(status=type(exc).__name__, error=str(exc)[:300])
        row.update(seconds=round(perf_counter() - started, 3))
        return row

    swap = booked.swap
    receive = Calculator.get_leg_pv(swap, LegSide.RECEIVE, market, Currency.CLP)
    pay = Calculator.get_leg_pv(swap, LegSide.PAY, market, Currency.CLP)
    mtm = Calculator.get_swap_mtm(swap, market, Currency.CLP)
    stored = stored_mtm(t, [booked.receive_leg_id, booked.pay_leg_id])
    stored_receive = stored[booked.receive_leg_id]
    stored_pay = -stored[booked.pay_leg_id]  # the DB signs the pay leg negative
    stored_swap = stored_receive - stored_pay
    row.update(
        receive_pv=receive,
        pay_pv=pay,
        mtm=mtm,
        stored_receive_pv=stored_receive,
        stored_pay_pv=stored_pay,
        stored_mtm=stored_swap,
        diff_receive_pv=receive - stored_receive,
        diff_pay_pv=pay - stored_pay,
        diff_mtm=mtm - stored_swap,
        seconds=round(perf_counter() - started, 3),
    )
    return row


def _config_error(rows: list[dict]) -> float | None:
    """Mean absolute MTM difference over the stored dates; None if a date failed to build."""
    if any(row["status"] != "ok" for row in rows):
        return None
    return sum(abs(row["diff_mtm"]) for row in rows) / len(rows)


def _print_top(top: list[tuple[float, list[dict]]], dates: list[date]) -> None:
    """Table of the best configs so far: stubs and non-zero terms read SOFR/ICP/basis."""
    header = (
        f"  {'cfg':>6}{'mean |diff|':>14}"
        + "".join(f"{f'{t:%d-%b}':>14}" for t in dates)
        + f"  {'index':<19}{'discount':<20}{'stubs':<7}{'non-zero':<14}{'basis':>6}{'coll':>6}"
    )
    print(header)
    for error, rows in top:
        first = rows[0]
        stubs = "/".join(first[f"{c}_stub"][0] for c in ("sofr", "icp", "basis"))
        non_zero = "/".join(
            first[f"{c}_first_non_zero_coupon_term"] for c in ("sofr", "icp", "basis")
        )
        print(
            f"  {first['config']:>6}{error:>14,.0f}"
            + "".join(f"{row['diff_mtm']:>14,.0f}" for row in rows)
            + f"  {first['index_curve_interpolation_method']:<19}{first['discount_curve_interpolation_method']:<20}"
            f"{stubs:<7}{non_zero:<14}{first['basis_first_quote']:>6}{first['icp_first_sofr_collateralized_term']:>6}"
        )


def _sort_csv_by_error(csv_path: Path) -> None:
    """Rewrites ``csv_path`` by ascending mean absolute MTM difference per config, with a config's dates together.

    A config that fails on any date goes last.
    """
    grid = pd.read_csv(csv_path, float_precision="round_trip")
    grid["_error"] = grid.groupby(_CONFIG_COLUMNS)["diff_mtm"].transform(
        lambda diffs: diffs.abs().mean(skipna=False)
    )
    grid = grid.sort_values(["_error", *_CONFIG_COLUMNS, "date"], na_position="last")
    grid.drop(columns="_error").to_csv(csv_path, index=False)


def run_config_grid(
    batch_size: int = 30, csv_path: Path = GRID_CSV, workers: int = 4
) -> list[tuple[float, list[dict]]]:
    """Values swap ``SWAP_ID`` on every stored date under every config, in ``workers`` processes.

    Every ``batch_size`` results it appends them to ``csv_path`` and prints the progress and the best configs so far,
    ranked by mean absolute MTM difference to the stored values. A config that fails on any date is not ranked. When
    all are done it sorts ``csv_path`` by that same error. Returns those best configs as ``(error, rows)``.
    """
    started = perf_counter()
    dates = sorted({t for t, _ in STORED_MTM})
    configs = list(all_valuation_configs())
    total = len(configs) * len(dates)
    tasks = (
        (number, t, config)
        for number, config in enumerate(configs, start=1)
        for t in dates
    )
    batch: list[dict] = []
    partial: dict[int, list[dict]] = {}  # rows of the configs still missing a date
    top: list[tuple[float, list[dict]]] = []
    done = failed = 0

    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, _GRID_COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()

        def flush() -> None:
            nonlocal done
            writer.writerows(batch)
            handle.flush()
            done += len(batch)
            batch.clear()
            elapsed = perf_counter() - started
            remaining = elapsed / done * (total - done)
            print(
                f"\n[{done:,}/{total:,} {done / total:6.1%}] {failed} failed, "
                f"{elapsed:,.0f}s elapsed, about {remaining:,.0f}s left"
            )
            _print_top(top, dates)

        with ProcessPoolExecutor(workers, initializer=_load_grid_state) as pool:
            for row in pool.map(_value_task, tasks):
                batch.append(row)
                failed += row["status"] != "ok"
                rows = partial.setdefault(row["config"], [])
                rows.append(row)
                if len(rows) == len(dates):
                    del partial[row["config"]]
                    error = _config_error(rows)
                    if error is not None:
                        top = nsmallest(
                            GRID_TOP_SIZE, [*top, (error, rows)], key=lambda c: c[0]
                        )
                if len(batch) == batch_size:
                    flush()
        if batch:
            flush()
    _sort_csv_by_error(csv_path)
    return top


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Values the swap under many valuation configs."
    )
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--csv", type=Path, default=GRID_CSV)
    args = parser.parse_args()
    run_config_grid(args.batch_size, args.csv, args.workers)
