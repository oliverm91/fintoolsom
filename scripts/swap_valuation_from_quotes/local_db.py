"""Read access to the local SQLite snapshot (``local_data.sqlite``) the swap valuation scripts run on.

The snapshot holds the same tables as MySQL, without the schema prefix:

* ``financial_data``  (``fin_data``):    upload_date, date, ticker, field, value
* ``swaps``, ``swap_legs`` (``derivatives``)

Dates are ISO text (``YYYY-MM-DD``), so pass dates as ``date.isoformat()``. ``export_snapshot.py`` creates the file and
is the only script that connects to MySQL.
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DB_PATH = Path(__file__).with_name("local_data.sqlite")


def read_sql(sql: str, **params) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found. Create it with export_snapshot.py (needs MySQL access).")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    finally:
        engine.dispose()
