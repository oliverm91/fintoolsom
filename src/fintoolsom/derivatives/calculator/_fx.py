"""FX conversion at spot and FX fixings (realised or projected)."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from ...dates.date_counts import date_ordinals

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency


def _convert(
    value: float, from_currency: Currency, to_currency: Currency, market: Market
) -> float:
    """`value` in `from_currency` converted to `to_currency` at spot (``market.t``)."""
    from ...market.currencies import CurrencyPair as _CurrencyPair

    return (
        value
        * market.get_fx_rate(market.t, _CurrencyPair(from_currency, to_currency)).value
    )


def get_fx_fixings(
    base_currency: Currency,
    quote_currency: Currency,
    fixing_dates: list[date],
    market: Market,
) -> np.ndarray:
    """FX rates (quote per unit of base) fixed on `fixing_dates`.

    Let ``L = market.get_fx_known_until(pair)``: ``market.t`` for a traded
    pair, the last published date for a pair priced by an index known ahead
    of time (the UF, up to the 9th of next month once CPI is out). A fixing on
    or before ``L`` is read as is (FX history, or UF history including its
    already-published future levels); a later one is projected from the rate
    at ``L`` with forward discount factors on the funding curves::

        FX(d) = FX(L) * [df_base(d) / df_base(L)] / [df_quote(d) / df_quote(L)]

    which is ``spot * df_base(d) / df_quote(d)`` when ``L = market.t``. Under
    a common CSA the collateral adjustment of both currencies cancels in this
    ratio, so funding curves are used regardless.

    Parameters
    ----------
    base_currency, quote_currency : Currency
        Currency pair.
    fixing_dates : list of date
        Fixing dates.
    market : Market
        Market at the valuation date.

    Returns
    -------
    numpy.ndarray
        One FX rate per fixing date.

    Raises
    ------
    KeyError
        If a known fixing, or the rate at ``L``, is not loaded.
    """
    return _fx_fixings_at_ordinals(
        base_currency, quote_currency, date_ordinals(fixing_dates), market
    )


def _fx_fixings_at_ordinals(
    base_currency: Currency,
    quote_currency: Currency,
    fixings: np.ndarray,
    market: Market,
) -> np.ndarray:
    """:func:`get_fx_fixings` for dates already held as integer ordinals."""
    from ...market.currencies import CurrencyPair as _CurrencyPair

    cp = _CurrencyPair(base_currency, quote_currency)
    known_until = market.get_fx_known_until(cp)
    last_known = known_until.toordinal()
    known = fixings <= last_known
    fx = np.empty(len(fixings))
    for i in np.where(known)[0]:
        fx[i] = market.get_fx_rate(date.fromordinal(int(fixings[i])), cp).value
    if (~known).any():
        anchor = market.get_fx_rate(known_until, cp).value
        ordinals = np.concatenate(([last_known], fixings[~known]))
        df_base = market.get_discount_curve(base_currency)._dfs_at_ordinals(ordinals)
        df_quote = market.get_discount_curve(quote_currency)._dfs_at_ordinals(ordinals)
        fx[~known] = anchor * (df_base[1:] / df_base[0]) / (df_quote[1:] / df_quote[0])
    return fx
