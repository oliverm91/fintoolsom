"""Data arrangement: the curves each quote relates to, and its pillar dates.

Pure functions of a quote or instrument (no solving): curve keys
(:func:`_curves_needed`), the date an instrument pins on a curve
(:func:`_get_maturity`), its last read date (:func:`_last_date`) and the curves it
may take its one pillar on (:func:`_primary_candidates`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..derivatives.forwards import NDF, Forward
from ..derivatives.swaps import FloatingLeg, Swap
from ..market.currencies import Currency
from ..market.index import Index
from ..market.market import CurveKey, Market, funding_key
from ..market.quotes import CrossCurrencyFloatFloatQuote, ForwardPointsQuote, InstrumentQuote

# Quote types whose instrument needs the market FX to fully materialise: XCCY (to size
# matched leg notionals) and forward-points (to build the outright strike off the spot).
_FX_DEPENDENT = (CrossCurrencyFloatFloatQuote, ForwardPointsQuote)


@dataclass(frozen=True)
class _Proj:
    """Solve key of the independent forward curve of a projection-only index.

    Discount curves are keyed by Index or Currency (``market.curves``); a
    projection-only index (e.g. a term index such as TAB 3M) is solved under this
    key and stored in ``market.projection_curves``. Both kinds go through the same
    scheduler, so a projection curve can depend on, and be needed by, other curves.
    """

    index: Index

    @property
    def currency(self) -> Currency:
        return self.index.currency


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
    return quote.get_instrument(market) if isinstance(quote, _FX_DEPENDENT) else quote.get_instrument()


def _discount_indexes(quote: InstrumentQuote, riskless_index: Index, extra: frozenset[Index]) -> set[Index]:
    """Indexes whose own curve discounts for `quote`.

    Parameters
    ----------
    quote : InstrumentQuote
        Quote considered.
    riskless_index : Index
        Market funding index.
    extra : frozenset of Index
        Other discount indexes (other quotes' collateral, index curves in market).

    Returns
    -------
    set of Index
        The riskless index, the quote's collateral and `extra`.
    """
    collateral: Index | None = getattr(quote, "collateral_index", None)
    return {riskless_index, *extra, *([collateral] if collateral is not None else [])}


def _curves_needed(
    quote: InstrumentQuote, riskless_index: Index, discount_indexes: frozenset[Index] = frozenset()
) -> frozenset[CurveKey]:
    """Every curve `quote` reads to be valued; quotes are grouped by this set.

    Each leg discounts in its settlement currency (the swap payment currency for a
    non-deliverable leg in another currency, else its own). Under collateral C the
    discount factor is ``funding(ccy) * curve(C) / funding(C.ccy)``; when
    ``ccy == C.ccy`` the funding factors cancel and only C is read. A non-deliverable leg also reads
    the funding curves of both currencies to project its FX fixings. A floating
    index is its own key when its curve discounts, else ``_Proj(index)``: a
    projection-only forward curve.

    Parameters
    ----------
    quote : InstrumentQuote
        Quote considered.
    riskless_index : Index
        Market funding index.
    discount_indexes : frozenset of Index, optional
        Other discount indexes, see :func:`_discount_indexes`.

    Returns
    -------
    frozenset of Index or Currency
    """
    instrument = quote.get_instrument()
    collateral: Index | None = getattr(quote, "collateral_index", None)
    discounting = _discount_indexes(quote, riskless_index, discount_indexes)
    keys: set[CurveKey] = set()

    if isinstance(instrument, Swap):
        for leg in (instrument.receive_leg, instrument.pay_leg):
            pay_ccy = _settlement_currency(instrument, leg)
            if collateral is None:
                keys.add(funding_key(riskless_index, pay_ccy))
            else:
                if pay_ccy != collateral.currency:
                    keys.add(funding_key(riskless_index, pay_ccy))
                    keys.add(funding_key(riskless_index, collateral.currency))
                keys.add(collateral)
            if pay_ccy != leg.currency:  # FX fixings projected on both funding curves
                keys.add(funding_key(riskless_index, leg.currency))
                keys.add(funding_key(riskless_index, pay_ccy))
            if isinstance(leg, FloatingLeg):
                keys.add(leg.index if leg.index in discounting else _Proj(leg.index))
    else:  # Forward / NDF (a UF forward is an NDF on CLF/CLP: CLP and CLF funding)
        pair = instrument.currency_pair
        for currency in (pair.base_currency, pair.quote_currency):
            keys.add(funding_key(riskless_index, currency))

    return frozenset(keys)


def _primary_candidates(quote: InstrumentQuote, pinned: frozenset, discount_indexes: frozenset) -> tuple:
    """Curves `quote` may take its one pillar on, preferred first.

    FX forwards and cross-currency swaps: the non-riskless funding curve they pin
    (the cross-currency basis). Single-currency swaps: their floating index curve;
    a float-float basis swap lists both indexes (the builder keeps the one no IRS
    already pins).
    """
    instrument = quote.get_instrument()
    funding = sorted((k for k in pinned if isinstance(k, Currency)), key=str)
    if not isinstance(instrument, Swap):
        return tuple(funding) or tuple(sorted(pinned, key=str))
    legs = (instrument.receive_leg, instrument.pay_leg)
    currencies = {leg.currency for leg in legs} | {_settlement_currency(instrument, leg) for leg in legs}
    if len(currencies) > 1 and funding:
        collateral = getattr(quote, "collateral_index", None)
        return tuple(sorted(funding, key=lambda k: k == getattr(collateral, "currency", None)))
    floats = [
        leg.index if leg.index in discount_indexes else _Proj(leg.index)
        for leg in legs if isinstance(leg, FloatingLeg)
    ]
    return tuple(floats) or tuple(sorted(pinned, key=str))


def _settlement_currency(swap: Swap, leg) -> Currency:
    """Currency `leg` is paid and discounted in.

    See :meth:`Swap.leg_settlement_currency`: a UF (CLF) leg always settles in CLP.
    """
    return swap.leg_settlement_currency(leg)


def _curve_currency(curve_key: CurveKey) -> Currency:
    """Currency of the flows `curve_key` discounts (an index curve: its currency)."""
    return curve_key if isinstance(curve_key, Currency) else curve_key.currency


def _get_maturity(curve_key: CurveKey, instrument, riskless_index: Index) -> date | None:
    """Pillar `instrument` pins on `curve_key`: its latest relevant date.

    Payment dates for the settlement-currency funding curve, FX fixing dates for
    the leg-currency funding curve of a non-deliverable leg, end dates for a
    discounting floating index; the fixing date (NDF) or payment date (forward)
    for an FX forward.

    Parameters
    ----------
    curve_key : Index or Currency
        Curve considered.
    instrument : Swap or Forward
        Instrument considered.
    riskless_index : Index
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
            if funding_key(riskless_index, pay_ccy) == curve_key:
                candidates.add(max(leg.payment_dates))
            if pay_ccy != leg.currency and funding_key(riskless_index, leg.currency) == curve_key:
                candidates.add(max(leg.fx_fixing_dates))
            if isinstance(leg, FloatingLeg) and curve_key in (leg.index, _Proj(leg.index)):
                candidates.add(max(leg.end_dates))
    elif isinstance(instrument, Forward):
        pair = instrument.currency_pair
        for currency in (pair.base_currency, pair.quote_currency):
            if funding_key(riskless_index, currency) == curve_key:
                candidates.add(instrument.fixing_date if isinstance(instrument, NDF) else instrument.payment_date)
    else:
        raise TypeError(
            f"_get_maturity only supports Swap and Forward instruments, got {type(instrument).__name__}."
        )
    return max(candidates) if candidates else None


def _last_date(instrument) -> date:
    """Last date `instrument` reads any curve on (payments, accrual ends, fixings)."""
    if isinstance(instrument, Forward):
        return max(instrument.payment_date, getattr(instrument, "fixing_date", instrument.payment_date))
    dates = []
    for leg in (instrument.receive_leg, instrument.pay_leg):
        dates += list(leg.payment_dates)
        if isinstance(leg, FloatingLeg):
            dates += list(leg.end_dates)
    return max(dates)
