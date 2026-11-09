"""Builds the curves from the ``fin_data`` quotes with fintoolsom.curve_builder and values a booked swap.

    python scripts/swap_valuation_from_quotes/run_valuation.py                      # swap 10 on 2026-11-06
    python scripts/swap_valuation_from_quotes/run_valuation.py --swap-id 10 --date 2026-11-06 --collateral-cutoff 18
    python scripts/swap_valuation_from_quotes/run_valuation.py --collateral 0        # value the swap uncollateralised (1 = SOFR)
    python scripts/swap_valuation_from_quotes/run_valuation.py --live                # current valuation from Bloomberg

Prints the leg PVs, the swap MTM and DV01 (CLP), next to the MTMs stored for swap 10 on 2026-11-06 (constants below). For any
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
from fintoolsom.market import Currency

from build_quotes import fixings_start, get_market_and_quotes
from build_swap import COLLATERAL_CHOICES, get_swap


# MTMs stored in derivatives.swaps_mtm for swap 10 on 2026-11-06 (CLP; the DB signs the pay leg negative), kept as constants.
_STORED_MTM = {
    (date(2026, 11, 6), "SLMNID5G"): 16_955_420_167.0,  # receive leg, fixed 4.92%
    (date(2026, 11, 6), "SLMNID5H"): -17_015_550_924.0,  # pay leg, ICP
    (date(2026, 11, 9), "SLMNID5G"): 16_966_709_576.0,  # receive leg
    (date(2026, 11, 9), "SLMNID5H"): -17_011_472_728.0,  # pay leg
}


def _stored_mtm(t: date, leg_ids: list[str]) -> dict[str, float]:
    """Stored MTM per leg id on ``t``; a leg without a stored value is left out."""
    return {
        leg_id: _STORED_MTM[(t, leg_id)]
        for leg_id in leg_ids
        if (t, leg_id) in _STORED_MTM
    }


def _format_report(
    result: dict,
    t: date,
    swap_id: int,
    n_quotes: int,
    curves: list[str],
    as_of: datetime | None = None,
) -> str:
    """Table of fintoolsom vs stored values. ``diff (bp)`` is ``diff / DV01``: the parallel rate move (bp) that explains the
    difference, positive when fintoolsom's rates would have to be higher than the stored ones. Legs use their own DV01 and the
    swap uses the net DV01 (the pay leg is floating, so its tiny DV01 inflates its bp). With ``as_of`` (live valuation) it only
    states the current PVs and MTM."""
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

    dv01 = {
        "receive_pv": result["dv01_receive"],
        "pay_pv": result["dv01_pay"],
        "mtm": result["dv01_receive"] - result["dv01_pay"],
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
        line += f"{dv01[key]:>16,.0f}"
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
    collateral_cutoff_months: int = 18,
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
        build_curves(inputs.quotes, market)
    for w in caught:
        if issubclass(w.category, (UserWarning, RuntimeWarning)):
            print(f"WARNING {w.category.__name__}: {str(w.message)[:500]}")

    swap = booked.swap
    collateral_index = getattr(swap, "collateral_index", None)
    result = {
        "collateral": collateral_index.name if collateral_index is not None else "none",
        "receive_pv": Calculator.get_leg_pv(
            swap, swap.receive_leg, market, Currency.CLP
        ),
        "pay_pv": Calculator.get_leg_pv(swap, swap.pay_leg, market, Currency.CLP),
        "mtm": Calculator.get_swap_mtm(swap, market, Currency.CLP),
    }
    if not live:
        result["dv01_receive"] = Calculator.get_leg_dv01(
            swap, swap.receive_leg, market, Currency.CLP
        )
        result["dv01_pay"] = Calculator.get_leg_dv01(
            swap, swap.pay_leg, market, Currency.CLP
        )
        stored = _stored_mtm(t, [booked.receive_leg_id, booked.pay_leg_id])
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
        default=18,
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
