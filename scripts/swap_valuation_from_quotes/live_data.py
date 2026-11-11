"""Live Bloomberg readers for ``run_valuation.py --live``. They return the same shapes as the snapshot readers of ``build_quotes.py``.

* ``get_live_quotes_table()``: current quotes with ``bdp`` (``PropTools.data.external_data.bbg``), one row per ticker (without the
  `` Curncy`` suffix) and the columns ``px_bid``, ``px_ask``, ``maturity``, ``sw_eff_dt``, ``settle_dt``.
* ``get_live_fixings(start, t)``: the ICP levels (``CLICP Index``) and SOFR fixings (``SOFRRATE Index``) from ``start`` to ``t``
  with ``bdh``. ``bdp`` only has the latest value, and the running coupons need the history. The dates are Bloomberg's own: the
  SOFR of a day is not published until the next one, so the fixing of ``t`` is missing and the last known rate is repeated.

The tickers are those the ``fin_data`` loader stores (see ``build_quotes.py`` for what each family is).
"""

from datetime import date

import pandas as pd

from PropTools.data.external_data.bbg.bbgInterAPI import bdh, bdp

SWAP_FIELDS = ["px_bid", "px_ask", "maturity", "sw_eff_dt"]
FORWARD_FIELDS = ["px_bid", "px_ask", "settle_dt"]
SPOT_FIELDS = ["px_bid", "px_ask"]

TICKERS = {
    "CHSWP": (
        [
            f"CHSWP{c} Curncy"
            for c in (
                "A",
                "B",
                "C",
                "F",
                "I",
                "1",
                "1F",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "12",
                "15",
                "20",
            )
        ],
        SWAP_FIELDS,
    ),
    "USOSFR": (
        [
            f"USOSFR{c} Curncy"
            for c in ["1Z", "2Z", "A", "B", "C", "D", "E", "F", "I", "1F"]
            + [str(n) for n in range(1, 21)]
        ],
        SWAP_FIELDS,
    ),
    "CPXOSS": (
        [f"CPXOSS{n} Curncy" for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)],
        SWAP_FIELDS,
    ),
    "CHN": (
        [
            f"CHN{t} Curncy"
            for t in (
                "1W",
                "2W",
                "1M",
                "2M",
                "3M",
                "4M",
                "5M",
                "6M",
                "9M",
                "12M",
                "18M",
                "2Y",
            )
        ],
        FORWARD_FIELDS,
    ),
    "USDCLP": (["USDCLP Curncy"], SPOT_FIELDS),
}
FIXING_TICKERS = ("CLICP Index", "SOFRRATE Index")


def get_live_quotes_table() -> pd.DataFrame:
    frames = []
    for family, (tickers, fields) in TICKERS.items():
        response = bdp(tickers, fields)
        if isinstance(response, str) or response is None or len(response) == 0:
            raise RuntimeError(
                f"Bloomberg bdp returned no data for the {family} tickers: {response}"
            )
        response = response.copy()
        response.index = [ticker.removesuffix(" Curncy") for ticker in response.index]
        frames.append(response)
    return pd.concat(frames)


def get_live_fixings(start: date, t: date) -> pd.DataFrame:
    history = bdh(list(FIXING_TICKERS), ["px_last"], start, t)
    if isinstance(history, str) or history is None or len(history) == 0:
        raise RuntimeError(
            f"Bloomberg bdh returned no data for {FIXING_TICKERS} between {start} and {t}: {history}"
        )
    rows = []
    for ticker in FIXING_TICKERS:
        for day, value in history[(ticker, "px_last")].dropna().items():
            rows.append((pd.to_datetime(day).date(), ticker, float(value)))
    return pd.DataFrame(rows, columns=["date", "ticker", "value"]).sort_values("date")
