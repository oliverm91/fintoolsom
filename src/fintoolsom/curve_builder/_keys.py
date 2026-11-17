"""Data arrangement: the curves each quote relates to, and its pillar dates.

Pure functions of a quote or instrument (no solving): curve keys
(:func:`_curves_needed`), the date an instrument pins on a curve
(:func:`_get_maturity`; all of them and its last read date: :func:`_maturities`) and
the curves it may take its one pillar on (:func:`_primary_candidates`).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .. import settings
from ..derivatives.forwards import NDF, Forward
from ..derivatives.swaps import FloatingLeg, Swap
from ..market.currencies import Currency
from ..market.index import Index
from ..market.market import Market
from ..market.quotes import (
    CrossCurrencyFixedFloatQuote,
    CrossCurrencyFloatFloatQuote,
    ForwardPointsQuote,
    InstrumentQuote,
)

# A curve the builder solves: an Index is that index's IndexCurve (it projects the
# index and may discount); a Currency is its DiscountCurve under the market's
# risk-free index (a currency other than the risk-free index's).
CurveKey = Index | Currency

# Quote types whose instrument needs the market FX to fully materialise: XCCY (to size
# matched leg notionals) and forward-points (to build the outright strike off the spot).
_FX_DEPENDENT = (
    CrossCurrencyFixedFloatQuote,
    CrossCurrencyFloatFloatQuote,
    ForwardPointsQuote,
)


def _funding_key(risk_free_index: Index, currency: Currency) -> CurveKey:
    """Key of the uncollateralised discount curve of `currency`.

    The risk-free index for its own currency; otherwise `currency`, whose curve is
    implied from cross-currency products.
    """
    return risk_free_index if currency == risk_free_index.currency else currency


def _curve_id(key: CurveKey, risk_free_index: Index) -> str:
    """Id of the curve under `key` in ``Market.curves``."""
    return key.name if isinstance(key, Index) else f"{key.value}_{risk_free_index.name}"


def _built(market: Market, key: CurveKey):
    """Curve of `key` in `market`, or None."""
    return market.curves.get(_curve_id(key, market.risk_free_index))


def _instrument_of(quote: InstrumentQuote, market: Market):
    """Instrument described by `quote`.

    FX-dependent quotes receive `market` so their notionals or strike resolve off
    the spot; the others do not take it.

    Parameters
    ----------
    quote : InstrumentQuote
        Quote to materialise.
    market : Market
        Market at the quote date.

    Returns
    -------
    Forward, NDF or Swap
    """
    return (
        quote.get_instrument(market)
        if isinstance(quote, _FX_DEPENDENT)
        else quote.get_instrument()
    )


def _curves_needed(
    quote: InstrumentQuote, risk_free_index: Index, instrument=None
) -> frozenset[CurveKey]:
    """Every curve `quote` reads to be valued; quotes are grouped by this set.

    Each leg discounts in its settlement currency (the swap payment currency for a
    non-deliverable leg in another currency, else its own). Under collateral C the
    discount factor is ``funding(ccy) * curve(C) / funding(C.ccy)``; when
    ``ccy == C.ccy`` the funding factors cancel and only C is read. A non-deliverable leg also reads
    the funding curves of both currencies to project its FX fixings. A floating
    leg reads its index's curve.

    Parameters
    ----------
    quote : InstrumentQuote
        Quote considered.
    risk_free_index : Index
        Market funding index.
    instrument : Forward, NDF or Swap, optional
        The quote's instrument, if already built; only its legs and currencies are read.

    Returns
    -------
    frozenset of Index or Currency
    """
    if instrument is None:
        instrument = quote.get_instrument()
    collateral: Index | None = getattr(quote, "collateral_index", None)
    keys: set[CurveKey] = set()

    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            pay_ccy = _settlement_currency(instrument, leg)
            if collateral is None:
                keys.add(_funding_key(risk_free_index, pay_ccy))
            else:
                if pay_ccy != collateral.currency:
                    keys.add(_funding_key(risk_free_index, pay_ccy))
                    keys.add(_funding_key(risk_free_index, collateral.currency))
                keys.add(collateral)
            if pay_ccy != leg.currency:  # FX fixings projected on both funding curves
                keys.add(_funding_key(risk_free_index, leg.currency))
                keys.add(_funding_key(risk_free_index, pay_ccy))
            if isinstance(leg, FloatingLeg):
                keys.add(leg.index)
    else:  # Forward / NDF (a UF forward is an NDF on CLF/CLP: CLP and CLF funding)
        pair = instrument.currency_pair
        for currency in (pair.base_currency, pair.quote_currency):
            keys.add(_funding_key(risk_free_index, currency))

    return frozenset(keys)


def _primary_candidates(
    quote: InstrumentQuote, pinned: frozenset, instrument=None
) -> tuple:
    """Curves `quote` may take its one pillar on, preferred first.

    FX forwards and cross-currency swaps: the non-risk-free funding curve they pin
    (the cross-currency basis). Single-currency swaps: their floating index curve;
    a float-float basis swap lists both indexes (the builder keeps the one no IRS
    already pins). `instrument` is the quote's instrument, if already built.
    """
    if instrument is None:
        instrument = quote.get_instrument()
    funding = sorted((k for k in pinned if isinstance(k, Currency)), key=str)
    if not isinstance(instrument, Swap):
        return tuple(funding) or tuple(sorted(pinned, key=str))
    legs = (instrument.receive_leg, instrument.pay_leg)
    currencies = {leg.currency for leg in legs} | {
        _settlement_currency(instrument, leg) for leg in legs
    }
    if len(currencies) > 1 and funding:
        collateral = getattr(quote, "collateral_index", None)
        return tuple(
            sorted(funding, key=lambda k: k == getattr(collateral, "currency", None))
        )
    floats = [leg.index for leg in legs if isinstance(leg, FloatingLeg)]
    return tuple(floats) or tuple(sorted(pinned, key=str))


def _settlement_currency(swap: Swap, leg) -> Currency:
    """Currency `leg` is paid and discounted in.

    See :meth:`Swap.leg_settlement_currency`: a UF (CLF) leg always settles in CLP.
    """
    return swap.leg_settlement_currency(leg)


def _curve_currency(curve_key: CurveKey) -> Currency:
    """Currency of the flows `curve_key` discounts (an index curve: its currency)."""
    return curve_key if isinstance(curve_key, Currency) else curve_key.currency


def _get_maturity(
    curve_key: CurveKey, instrument, risk_free_index: Index
) -> date | None:
    """Pillar `instrument` pins on `curve_key`: its latest relevant date.

    Payment dates for the settlement-currency funding curve, FX fixing dates for
    the leg-currency funding curve of a non-deliverable leg, end dates for a
    floating index; the fixing date (NDF) or payment date (forward)
    for an FX forward.

    Parameters
    ----------
    curve_key : Index or Currency
        Curve considered.
    instrument : Swap or Forward
        Instrument considered.
    risk_free_index : Index
        Market funding index.

    Returns
    -------
    date or None
        None when the instrument does not touch `curve_key`.

    Raises
    ------
    TypeError
        For an unsupported instrument type.
    """
    return _maturities(instrument, risk_free_index)[0].get(curve_key)


def _maturities(instrument, risk_free_index: Index) -> tuple[dict, date]:
    """Pillar `instrument` pins on each curve it touches, and its last read date.

    The first is ``{curve key: _get_maturity(curve key, ...)}`` over the curves it
    touches; the last date is the last one it reads any curve on (payments, accrual
    ends, fixings).

    Raises
    ------
    TypeError
        For an unsupported instrument type.
    """
    pinned: dict = {}

    def add(key, d: date) -> None:
        if key not in pinned or d > pinned[key]:
            pinned[key] = d

    if isinstance(instrument, Swap):
        last: list[date] = []
        for leg in (instrument.receive_leg, instrument.pay_leg):
            pay_ccy = _settlement_currency(instrument, leg)
            add(_funding_key(risk_free_index, pay_ccy), pay := max(leg.payment_dates))
            if pay_ccy != leg.currency:
                add(
                    _funding_key(risk_free_index, leg.currency),
                    max(leg.fx_fixing_dates),
                )
            last.append(pay)
            if isinstance(leg, FloatingLeg):
                add(leg.index, end := max(leg.end_dates))
                last.append(end)
        return pinned, max(last)
    if isinstance(instrument, Forward):
        pair = instrument.currency_pair
        is_ndf = isinstance(instrument, NDF)
        d = instrument.fixing_date if is_ndf else instrument.payment_date
        for currency in (pair.base_currency, pair.quote_currency):
            add(_funding_key(risk_free_index, currency), d)
        return pinned, max(instrument.payment_date, d)
    raise TypeError(
        f"_get_maturity only supports Swap and Forward instruments, got {type(instrument).__name__}."
    )


def _table_kind(instrument) -> int:
    """Kind of `instrument` for ``_native.maturity_tables`` (0: one it does not take)."""
    if isinstance(instrument, NDF):
        return 2
    if isinstance(instrument, Forward):
        return 1
    if isinstance(instrument, Swap):
        return (
            4
            + 8 * isinstance(instrument.receive_leg, FloatingLeg)
            + 16 * isinstance(instrument.pay_leg, FloatingLeg)
        )
    return 0


def _maturity_tables(instruments: list, risk_free_index: Index) -> list:
    """:func:`_maturities` of each of `instruments`, in one C++ call on the C++ builder."""
    tables = [None] * len(instruments)
    if settings.is_cpp("curve_builder"):
        kinds = np.fromiter(map(_table_kind, instruments), np.int64, len(instruments))
        tables = settings.native().maturity_tables(
            list(instruments), kinds, risk_free_index
        )
    return [
        t if t is not None else _maturities(i, risk_free_index)
        for i, t in zip(instruments, tables)
    ]
