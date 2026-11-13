"""Deliverable and non-deliverable FX forwards."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency
    from ..forwards.forwards import NDF, Forward


def get_forward_mtm(
    forward: Forward,
    market: Market,
    currency: Currency,
) -> float:
    """Deliverable FX forward mark-to-market.

    Domestic is the quote currency and foreign the base currency, each
    discounted under the forward's ``collateral_index`` (see
    :meth:`Market.get_dfs`; None means uncollateralised); spot is read from
    `market` at ``market.t``. A forward paid on or before ``market.t`` is worth 0.

    Parameters
    ----------
    forward : Forward
        Deliverable forward.
    market : Market
        Market at the valuation date.
    currency : Currency
        Reporting currency; the quote-currency MTM is converted at spot.

    Returns
    -------
    float
        MTM in `currency`.
    """
    cp = forward.currency_pair
    if forward.payment_date <= market.t:
        return 0.0
    collateral = forward.collateral_index
    spot = market.get_fx_rate(market.t, cp).value

    df_d = market.get_df(cp.quote_currency, forward.payment_date, collateral=collateral)
    df_f = market.get_df(cp.base_currency, forward.payment_date, collateral=collateral)
    notional_leg_vp = spot * forward.notional * df_f
    strike_leg_vp = forward.strike * forward.notional * df_d

    mtm = forward.sign * (notional_leg_vp - strike_leg_vp)
    return market.convert_value_currency(mtm, cp.quote_currency, currency)


def get_ndf_mtm(
    ndf: NDF,
    market: Market,
    currency: Currency,
) -> float:
    """Non-deliverable FX forward mark-to-market.

    The settlement FX at `fixing_date` is the realised rate
    (:meth:`Market.get_fx_rate`) once fixed, else :meth:`Market.get_future_fx_rate`
    (realised or already published, else projected on the funding curves,
    whatever the collateral) and the settlement amount is discounted from
    `payment_date` in the quote currency under the NDF's ``collateral_index``
    (see :meth:`Market.get_dfs`; None means uncollateralised). A UF forward is
    an NDF on ``CLF/CLP``. An NDF paid on or before ``market.t`` is worth 0.

    Parameters
    ----------
    ndf : NDF
        Non-deliverable forward.
    market : Market
        Market at the valuation date.
    currency : Currency
        Reporting currency; the quote-currency MTM is converted at spot.

    Returns
    -------
    float
        MTM in `currency`.
    """
    cp = ndf.currency_pair
    if ndf.payment_date <= market.t:
        return 0.0
    if ndf.fixing_date <= market.t:
        s_t = market.get_fx_rate(ndf.fixing_date, cp).value
    else:
        s_t = market.get_future_fx_rate(ndf.fixing_date, cp).value
    fv = ndf.sign * ndf.notional * (s_t - ndf.strike)
    mtm = fv * market.get_df(
        cp.quote_currency, ndf.payment_date, collateral=ndf.collateral_index
    )
    return market.convert_value_currency(mtm, cp.quote_currency, currency)
