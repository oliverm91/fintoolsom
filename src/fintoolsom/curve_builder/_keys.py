"""Data arrangement: the curves each quote relates to, and its pillar dates.

Pure functions of a quote or instrument (no solving): curve keys
(:func:`_curves_needed`), the date an instrument pins on a curve
(:func:`_get_maturity`), its last read date (:func:`_last_date`) and the curves it
may take its one pillar on (:func:`_primary_candidates`).
"""

from __future__ import annotations

from datetime import date

from ..derivatives.forwards import NDF, Forward
from ..derivatives.swaps import FloatingLeg, Swap
from ..market.currencies import Currency
from ..market.index import Index
from ..market.market import Market
from ..market.quotes import (
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
_FX_DEPENDENT = (CrossCurrencyFloatFloatQuote, ForwardPointsQuote)


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
    quote: InstrumentQuote, risk_free_index: Index
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

    Returns
    -------
    frozenset of Index or Currency
    """
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


def _primary_candidates(quote: InstrumentQuote, pinned: frozenset) -> tuple:
    """Curves `quote` may take its one pillar on, preferred first.

    FX forwards and cross-currency swaps: the non-risk-free funding curve they pin
    (the cross-currency basis). Single-currency swaps: their floating index curve;
    a float-float basis swap lists both indexes (the builder keeps the one no IRS
    already pins).
    """
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
    candidates: set[date] = set()
    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            pay_ccy = _settlement_currency(instrument, leg)
            if _funding_key(risk_free_index, pay_ccy) == curve_key:
                candidates.add(max(leg.payment_dates))
            if (
                pay_ccy != leg.currency
                and _funding_key(risk_free_index, leg.currency) == curve_key
            ):
                candidates.add(max(leg.fx_fixing_dates))
            if isinstance(leg, FloatingLeg) and curve_key == leg.index:
                candidates.add(max(leg.end_dates))
    elif isinstance(instrument, Forward):
        pair = instrument.currency_pair
        for currency in (pair.base_currency, pair.quote_currency):
            if _funding_key(risk_free_index, currency) == curve_key:
                candidates.add(
                    instrument.fixing_date
                    if isinstance(instrument, NDF)
                    else instrument.payment_date
                )
    else:
        raise TypeError(
            f"_get_maturity only supports Swap and Forward instruments, got {type(instrument).__name__}."
        )
    return max(candidates) if candidates else None


def _last_date(instrument) -> date:
    """Last date `instrument` reads any curve on (payments, accrual ends, fixings)."""
    if isinstance(instrument, Forward):
        return max(
            instrument.payment_date,
            getattr(instrument, "fixing_date", instrument.payment_date),
        )
    dates = []
    for leg in (instrument.receive_leg, instrument.pay_leg):
        dates += list(leg.payment_dates)
        if isinstance(leg, FloatingLeg):
            dates += list(leg.end_dates)
    return max(dates)
