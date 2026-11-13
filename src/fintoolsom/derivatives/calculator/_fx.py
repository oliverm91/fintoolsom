"""FX conversion and fixings, kept for the :class:`Calculator` facade.

The implementation lives in :class:`~fintoolsom.market.Market`
(:meth:`~fintoolsom.market.Market.get_future_fx_rate`,
:meth:`~fintoolsom.market.Market.convert_value_currency` and
:meth:`~fintoolsom.market.Market.convert_future_value_currency`); these
functions delegate to it.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from ...dates.date_counts import date_ordinals
from ...market.currencies import CurrencyPair

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency


def _convert(
    value: float, from_currency: Currency, to_currency: Currency, market: Market
) -> float:
    """`value` converted at spot; see :meth:`Market.convert_value_currency`."""
    return market.convert_value_currency(value, from_currency, to_currency)


def get_fx_fixings(
    base_currency: Currency,
    quote_currency: Currency,
    fixing_dates: list[date],
    market: Market,
) -> np.ndarray:
    """FX rates (quote per unit of base) fixed on `fixing_dates`, known or
    projected; see :meth:`Market.get_future_fx_rate`.

    Returns
    -------
    numpy.ndarray
        One FX rate per fixing date.
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
    return market._future_fx_rates_at_ordinals(
        CurrencyPair(base_currency, quote_currency), fixings
    )
