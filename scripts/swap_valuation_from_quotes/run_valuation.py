"""Builds the curves from the ``fin_data`` quotes with fintoolsom.curve_builder and values a booked swap.

    python scripts/swap_valuation_from_quotes/run_valuation.py                      # swap 10 on 2026-11-06
    python scripts/swap_valuation_from_quotes/run_valuation.py --swap-id 10 --date 2026-11-06 --collateral-cutoff 24
    python scripts/swap_valuation_from_quotes/run_valuation.py --collateral 0        # value the swap uncollateralised (1 = SOFR)
    python scripts/swap_valuation_from_quotes/run_valuation.py --live                # current valuation from Bloomberg

Prints the leg PVs, the swap MTM and DV01 (CLP), next to the MTMs stored for swap 10 on 2026-11-06 (constants in ``stored_values.py``). For any
other swap or date there is no stored MTM: the table then just leaves those columns out, without a message or an error.
Runs fully offline on the SQLite snapshot ``local_data.sqlite`` (see ``local_db.py``). The snapshot only covers the dates
and swap exported by ``export_snapshot.py`` (the only script that needs MySQL).

``--live`` ignores ``--date``: it values on today with the current Bloomberg quotes (bdp) and the ICP and SOFR histories (bdh)
instead of the snapshot (see ``live_data.py``), compares to nothing and only states the current MTM of the swap and of each leg.
The swap itself is still read from the snapshot, so it must contain ``--swap-id``.
"""

import argparse
import warnings
from datetime import date, datetime

from fintoolsom.curve_builder import build_curves
from fintoolsom.derivatives import Calculator
from fintoolsom.derivatives.swaps import FloatingLeg, LegSide
from fintoolsom.market import Currency
from fintoolsom.rates import InterpolationMethod

from build_quotes import fixings_start, get_market_and_quotes
from build_swap import COLLATERAL_CHOICES, get_swap
from stored_values import stored_mtm


def _leg_dv01(swap, side: LegSide, market) -> float | None:
    """DV01 (CLP) of a fixed leg; None for a floating leg, which is not calculated."""
    if isinstance(swap.leg(side), FloatingLeg):
        return None
    return Calculator.get_leg_dv01(swap, side, market, Currency.CLP)


def _format_report(
    result: dict,
    t: date,
    swap_id: int,
    n_quotes: int,
    curves: list[str],
    as_of: datetime | None = None,
) -> str:
    """Table of fintoolsom vs stored values. ``diff (bp)`` is ``diff / DV01``: the parallel rate move (bp) that explains the
    difference, positive when fintoolsom's rates would have to be higher than the stored ones. Fixed legs use their own DV01, a
    floating leg shows n/a and the swap uses the net DV01 of its fixed legs. With ``as_of`` (live valuation) it only states the
    current PVs and MTM."""
    rows = (
        ("Receive leg PV", "receive_pv"),
        ("Pay leg PV", "pay_pv"),
        ("Swap MTM", "mtm"),
    )
    description = f"collateral: {result['collateral']} | {n_quotes} quotes -> curves {', '.join(curves)}"

    if as_of is not None:
        header = f"{'':<16}{'current':>18}"
        lines = [
            f"Swap {swap_id} | LIVE (Bloomberg) {as_of:%Y-%m-%d %H:%M} | {description}",
            "CLP",
            header,
            "-" * len(header),
        ]
        lines += [f"{label:<16}{result[key]:>18,.0f}" for label, key in rows]
        return "\n".join(lines)

    dv01_receive, dv01_pay = result["dv01_receive"], result["dv01_pay"]
    dv01 = {
        "receive_pv": dv01_receive,
        "pay_pv": dv01_pay,
        "mtm": None
        if dv01_receive is None and dv01_pay is None
        else (dv01_receive or 0.0) - (dv01_pay or 0.0),
    }
    has_stored = "stored" in result
    header = f"{'':<16}{'fintoolsom':>18}"
    header += f"{'stored':>18}{'diff':>15}" if has_stored else ""
    header += f"{'DV01 (CLP/bp)':>16}" + (f"{'diff (bp)':>12}" if has_stored else "")
    lines = [
        f"Swap {swap_id} | valuation date {t:%Y-%m-%d} | {description}",
        "CLP",
        header,
        "-" * len(header),
    ]
    for label, key in rows:
        line = f"{label:<16}{result[key]:>18,.0f}"
        if has_stored:
            diff = result[key] - result["stored"][key]
            line += f"{result['stored'][key]:>18,.0f}{diff:>15,.0f}"
        line += f"{dv01[key]:>16,.0f}" if dv01[key] is not None else f"{'n/a':>16}"
        if has_stored:
            line += f"{diff / dv01[key]:>+12.2f}" if dv01[key] else f"{'n/a':>12}"
        lines.append(line)
    if has_stored:
        lines.append(
            "diff (bp) = diff / DV01: rate move that would explain the difference (+ = fintoolsom rates higher)"
        )
    return "\n".join(lines)


def run(
    t: date,
    swap_id: int,
    collateral_cutoff_months: int = 24,
    collateral: int = 1,
    verbose: bool = True,
    live: bool = False,
) -> dict:
    """Values swap ``swap_id`` on ``t``. With ``live`` ``t`` is ignored: it values on today with Bloomberg data and does not
    compare to the stored MTMs."""
    as_of = datetime.now() if live else None
    if live:
        t = as_of.date()
    booked = get_swap(swap_id, collateral=collateral)
    inputs = get_market_and_quotes(
        t,
        fixings_start(booked.start_date, t),
        collateral_cutoff_months=collateral_cutoff_months,
        live=live,
    )
    market = inputs.market
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_curves(
            inputs.quotes,
            market,
            interp_method=InterpolationMethod.PiecewiseConstant,
        )
    for w in caught:
        if issubclass(w.category, (UserWarning, RuntimeWarning)):
            print(f"WARNING {w.category.__name__}: {str(w.message)[:500]}")

    swap = booked.swap
    collateral_index = getattr(swap, "collateral_index", None)
    result = {
        "collateral": collateral_index.name if collateral_index is not None else "none",
        "receive_pv": Calculator.get_leg_pv(
            swap, LegSide.RECEIVE, market, Currency.CLP
        ),
        "pay_pv": Calculator.get_leg_pv(swap, LegSide.PAY, market, Currency.CLP),
        "mtm": Calculator.get_swap_mtm(swap, market, Currency.CLP),
    }
    if not live:
        result["dv01_receive"] = _leg_dv01(swap, LegSide.RECEIVE, market)
        result["dv01_pay"] = _leg_dv01(swap, LegSide.PAY, market)
        stored = stored_mtm(t, [booked.receive_leg_id, booked.pay_leg_id])
        if len(stored) == 2:  # the DB stores the pay leg with a negative sign
            result["stored"] = {
                "receive_pv": stored[booked.receive_leg_id],
                "pay_pv": -stored[booked.pay_leg_id],
                "mtm": sum(stored.values()),
            }

    if verbose:
        print(
            _format_report(
                result, t, swap_id, len(inputs.quotes), list(market.curves), as_of=as_of
            )
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=date(2026, 11, 6),
        help="valuation date (default 2026-11-06; ignored with --live)",
    )
    parser.add_argument("--swap-id", type=int, default=10)
    parser.add_argument(
        "--collateral-cutoff",
        type=int,
        default=24,
        help="CHSWP tenors (months) from which the quote is SOFR collateralised",
    )
    parser.add_argument(
        "--collateral",
        type=int,
        choices=COLLATERAL_CHOICES,
        default=1,
        help="CSA of the swap: 1 = SOFR collateralised, 0 = uncollateralised (default: 1)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="current valuation from Bloomberg (bdp quotes, bdh fixings); ignores --date, compares to nothing",
    )
    args = parser.parse_args()
    run(
        args.date, args.swap_id, args.collateral_cutoff, args.collateral, live=args.live
    )
