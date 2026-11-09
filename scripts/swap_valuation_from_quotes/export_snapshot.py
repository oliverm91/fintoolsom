"""Creates ``local_data.sqlite``, the offline snapshot ``run_valuation.py`` runs on. The only script here that needs MySQL.

    python scripts/swap_valuation_from_quotes/export_snapshot.py [--start-date 2026-11-04] [--end-date 2026-11-06] [--swap-id 10]

``--start-date`` defaults to 2026-11-06 and ``--end-date`` to ``--start-date``. Every date in the range that has quotes can then
be valued. Needs the PT_MYSQL_* environment variables and read grants on ``fin_data`` and ``derivatives``. It copies:

* ``fin_data.financial_data``: the quote tickers for each date of the range, and the CLICP (ICP levels) and SOFRRATE ``px_last``
  fixings from ``fixings_start`` (a week before the swap start or the first date, whichever is earlier) to the last date;
* ``derivatives.swaps`` and ``derivatives.swap_legs`` of ``--swap-id``.

Running it again overwrites the snapshot. The stored MTMs are not exported: ``run_valuation.py`` keeps them as constants.
"""

import argparse
import os
import sqlite3
from datetime import date

import pandas as pd
from sqlalchemy import text

from PropTools.config.settings import (
    MYSQL_HOST_IP,
    MYSQL_HOST_PORT,
    PT_MYSQL_PWD,
    PT_MYSQL_USR,
)
from PropTools.data.MySQLWrapper import MySQLWrapper

from build_quotes import fixings_start
from local_db import DB_PATH


def _read_mysql(database: str, sql: str, **params) -> pd.DataFrame:
    with MySQLWrapper(
        MYSQL_HOST_IP,
        MYSQL_HOST_PORT,
        os.environ.get(PT_MYSQL_USR),
        os.environ.get(PT_MYSQL_PWD),
        database,
    ) as conn:
        return pd.read_sql(text(sql), conn, params=params)


def _sqlite_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Dates to ISO text and Decimals to floats, which is what SQLite and ``local_db`` expect."""
    out = df.copy()
    for column in out.columns:
        if str(out[column].dtype).startswith("datetime64"):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
            continue
        if out[column].dtype != object:
            continue
        sample = out[column].dropna()
        if sample.empty:
            continue
        if hasattr(sample.iloc[0], "isoformat"):
            out[column] = out[column].map(
                lambda v: v.isoformat()[:10] if pd.notna(v) else None
            )
        elif type(sample.iloc[0]).__name__ == "Decimal":
            out[column] = out[column].map(lambda v: float(v) if pd.notna(v) else None)
    return out


def export(start_date: date, end_date: date, swap_id: int) -> None:
    if end_date < start_date:
        raise ValueError(f"--end-date {end_date} is before --start-date {start_date}.")
    swaps = _read_mysql(
        "derivatives", "SELECT * FROM derivatives.swaps WHERE id=:id", id=swap_id
    )
    legs = _read_mysql(
        "derivatives",
        "SELECT * FROM derivatives.swap_legs WHERE fk_swap=:id",
        id=swap_id,
    )
    if swaps.empty or legs.empty:
        raise ValueError(f"Swap {swap_id} not found in derivatives.")
    fixings_from = fixings_start(
        pd.to_datetime(swaps.start_date.iloc[0]).date(), start_date
    )

    quotes = _read_mysql(
        "fin_data",
        """SELECT upload_date, date, ticker, field, value FROM fin_data.financial_data
           WHERE (date BETWEEN :d0 AND :d1 AND (ticker LIKE 'CHSWP%' OR ticker LIKE 'USOSFR%' OR ticker LIKE 'CPXOSS%'
                                                OR ticker LIKE 'CHN%' OR ticker LIKE 'USDCLP%'))
              OR (field='px_last' AND date BETWEEN :f0 AND :d1 AND (ticker LIKE 'CLICP%' OR ticker LIKE 'SOFRRATE%'))""",
        d0=start_date,
        d1=end_date,
        f0=fixings_from,
    )
    if quotes.empty:
        raise ValueError(f"No quotes in fin_data between {start_date} and {end_date}.")
    quotes["field"] = quotes[
        "field"
    ].str.lower()  # CLICP is stored as PX_LAST, the other fixings as px_last

    DB_PATH.unlink(missing_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for name, df in (
            ("financial_data", quotes),
            ("swaps", swaps),
            ("swap_legs", legs),
        ):
            _sqlite_friendly(df).to_sql(name, conn, index=False)
    quote_days = quotes[quotes.ticker.str.startswith("CHSWP")]["date"].nunique()
    print(
        f"{DB_PATH}: {len(quotes)} financial_data rows ({quote_days} quote dates in {start_date}..{end_date}, fixings from "
        f"{fixings_from}), swap {swap_id} with {len(legs)} legs."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--start-date", type=date.fromisoformat, default=date(2026, 11, 6)
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="default: --start-date",
    )
    parser.add_argument("--swap-id", type=int, default=10)
    args = parser.parse_args()
    export(args.start_date, args.end_date or args.start_date, args.swap_id)
