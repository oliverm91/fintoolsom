"""Swap leg flows and present values, deliverable or not."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..swaps.legs import FixedLeg, TermRateLeg
from ..swaps.swaps import LegSide
from ._fx import _convert, _fx_fixings_at_ordinals

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency
    from ..swaps.swaps import Swap


def _leg_flows(leg, market: Market) -> tuple[np.ndarray, np.ndarray]:
    """Flows, in the leg currency, of the coupons not yet paid.

    A coupon paid on or before ``market.t`` is dropped.

    Realised fixings come from the index history (``market.get_index``).

    TermRateLeg, split by fixing date:

    * fixed (``fixing_date <= t``): the fixed rate over the whole coupon.
    * to project (``fixing_date > t``): forward accrual from the projection curve.

    OvernightLeg and XCCYFloatingLeg, split by start date:

    * running (``start <= t < payment``): realised ``[start, t]`` (index business
      days, or ICP level ratio) compounded with the projection ``[t, end]``, or
      fully realised when ``end <= t``.
    * future (``start > t``): forward accrual from the projection curve.

    Parameters
    ----------
    leg : SwapLeg
        Fixed, term-rate, overnight or cross-currency floating leg.
    market : Market
        Market at the valuation date.

    Returns
    -------
    future_payment : numpy.ndarray of bool
        Per-coupon mask of the coupons paid after ``market.t``.
    flows : numpy.ndarray
        Flows of those coupons (interest + spread + amortization).

    Raises
    ------
    KeyError
        If a realised fixing is needed and is missing from the index history.
    """
    t = market.t.toordinal()
    future_payment = leg._payment_ordinals > t
    if not future_payment.any():
        return future_payment, np.zeros(0)

    if isinstance(leg, FixedLeg):
        return future_payment, leg.flows[future_payment]

    flows = np.zeros(int(future_payment.sum()))
    # Projection goes through get_accrual (interest per unit notional) on the
    # index's own curve.
    proj_curve = market.get_index_curve(leg.index)

    if isinstance(leg, TermRateLeg):
        # Once fixing_date <= t the rate is locked, wherever start_date falls.
        fixed_mask = future_payment & (leg._fixing_ordinals <= t)
        to_proj_mask = future_payment & ~fixed_mask

        if fixed_mask.any():
            history = _index_history(leg, market)
            fixed_flows = [
                history.get_rate(leg.fixing_dates[i]).get_accrued_interest(
                    leg.residuals[i], leg.start_dates[i], leg.end_dates[i]
                )
                + leg.spreads_values[i]
                + leg.amortizations[i]
                for i in np.where(fixed_mask)[0]
            ]
            flows[fixed_mask[future_payment]] = np.array(fixed_flows)

        if to_proj_mask.any():
            accruals = proj_curve._accrual_at_ordinals(
                leg._start_ordinals[to_proj_mask], leg._end_ordinals[to_proj_mask]
            )
            flows[to_proj_mask[future_payment]] = (
                leg.residuals[to_proj_mask] * accruals
                + leg.spreads_values[to_proj_mask]
                + leg.amortizations[to_proj_mask]
            )

    else:
        future_start = leg._start_ordinals > t
        pure_future = future_payment & future_start
        current_mask = future_payment & ~future_start

        if pure_future.any():
            accruals = proj_curve._accrual_at_ordinals(
                leg._start_ordinals[pure_future], leg._end_ordinals[pure_future]
            )
            # amortizations: the notional repaid with the final coupon (final
            # exchange). Same-currency legs cancel it; cross-currency legs exchange it.
            flows[pure_future[future_payment]] = (
                leg.residuals[pure_future] * accruals
                + leg.spreads_values[pure_future]
                + leg.amortizations[pure_future]
            )

        if current_mask.any():
            idx = int(np.where(current_mask)[0][0])
            c = leg.coupons[idx]
            # Realised part from the index history: compounded on the index's own
            # business days (rate index) or the ratio of published levels (ICP).
            realised_end = min(market.t, c.end_date)
            accrued = _index_history(leg, market).get_accrued_interest(
                c.residual, c.start_date, realised_end
            )
            interest = accrued
            if realised_end < c.end_date:
                interest += (c.residual + accrued) * proj_curve.get_accrual(
                    market.t, c.end_date
                )
            flows[current_mask[future_payment]] = (
                interest + leg.spreads_values[idx] + leg.amortizations[idx]
            )

    return future_payment, flows


def _index_history(leg, market: Market):
    """History of the leg index, holding its realised fixings.

    Raises
    ------
    KeyError
        If `market` has no history for the index.
    """
    return market.get_index(leg.index.name)


def _leg_pv(leg, collateral, market: Market) -> float:
    """PV of a leg settled in its own currency.

    Parameters
    ----------
    leg : SwapLeg
        Leg to value.
    collateral : InterestIndex or None
        CSA collateral index; None means uncollateralised.
    market : Market
        Market at the valuation date.

    Returns
    -------
    float
        PV in the leg currency.
    """
    future_payment, flows = _leg_flows(leg, market)
    if not future_payment.any():
        return 0.0
    dfs = market._dfs_at_ordinals(
        leg.currency, leg._payment_ordinals[future_payment], collateral=collateral
    )
    return float(np.dot(flows, dfs))


def _leg_pv_non_deliverable(
    leg, collateral, market: Market, settlement_currency: Currency
) -> float:
    """PV of a leg settled non-deliverably in `settlement_currency`.

    Each unpaid flow is converted at its FX fixing (realised if fixed on or
    before ``market.t``, including a flow fixed but not yet paid; projected
    otherwise) and discounted from its payment date in `settlement_currency`.

    Parameters
    ----------
    leg : SwapLeg
        Leg to value.
    collateral : InterestIndex or None
        CSA collateral index; None means uncollateralised.
    market : Market
        Market at the valuation date.
    settlement_currency : Currency
        Currency the flows are paid in.

    Returns
    -------
    float
        PV in `settlement_currency`.
    """
    if leg.currency == settlement_currency:
        return _leg_pv(leg, collateral, market)
    future_payment, flows = _leg_flows(leg, market)
    if not future_payment.any():
        return 0.0
    fx = _fx_fixings_at_ordinals(
        leg.currency,
        settlement_currency,
        leg._fx_fixing_ordinals[future_payment],
        market,
    )
    dfs = market._dfs_at_ordinals(
        settlement_currency,
        leg._payment_ordinals[future_payment],
        collateral=collateral,
    )
    return float(np.sum(flows * fx * dfs))


def get_swap_mtm(swap: Swap, market: Market, currency: Currency) -> float:
    """Swap mark-to-market in the reporting `currency`.

    Each leg settles in :meth:`Swap.leg_settlement_currency`. A leg settled in
    its own currency is valued there; otherwise (non-deliverable swap, or a
    UF leg, always paid in CLP) every flow is converted at its FX fixing and
    discounted in the settlement currency. Each leg PV is then converted to
    `currency` at spot. Only the final notional exchange (the last coupon's
    amortization) is valued; there is no initial exchange.

    Parameters
    ----------
    swap : Swap
        Swap to value.
    market : Market
        Market at the valuation date.
    currency : Currency
        Reporting currency.

    Returns
    -------
    float
        PV of the received leg minus PV of the paid leg, in `currency`.
    """
    return get_leg_pv(swap, LegSide.RECEIVE, market, currency) - get_leg_pv(
        swap, LegSide.PAY, market, currency
    )


def get_leg_pv(swap: Swap, side: LegSide, market: Market, currency: Currency) -> float:
    """PV of one leg of `swap`, in the reporting `currency`.

    The leg settles in :meth:`Swap.leg_settlement_currency` and is discounted
    under the swap's collateral, as in :func:`get_swap_mtm`.

    Parameters
    ----------
    swap : Swap
        Swap to value one leg of.
    side : LegSide
        Which leg: ``LegSide.RECEIVE`` or ``LegSide.PAY``.
    market : Market
        Market at the valuation date.
    currency : Currency
        Reporting currency.

    Returns
    -------
    float
        PV of the unpaid flows, positive for a leg that pays to its holder.
    """
    leg = swap.leg(side)
    settlement = swap.leg_settlement_currency(leg)
    pv = _leg_pv_non_deliverable(leg, swap.collateral_index, market, settlement)
    return _convert(pv, settlement, currency, market)
