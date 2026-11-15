"""Swap leg flows and present values, deliverable or not."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..swaps.legs import FixedLeg, TermRateLeg
from ..swaps.swaps import LegSide
from ...market.currencies import CurrencyPair

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency
    from ..swaps.swaps import Swap


def _leg_flows(leg, market: Market) -> tuple[np.ndarray, np.ndarray]:
    """Flows, in the leg currency, of the coupons not yet paid.

    A coupon paid on or before ``market.t`` is dropped.

    Realised fixings come from the index history (``market.get_index``).

    OvernightLeg and XCCYFloatingLeg, split by start date:

    * running (``start <= t < payment``): realised ``[start, t]`` (index business
      days, or ICP level ratio) compounded with the projection ``[t, end]``, or
      fully realised when ``end <= t``.
    * future (``start > t``): forward accrual from the projection curve.

    TermRateLeg, split by fixing date:

    * fixed (``fixing_date <= t``): the fixed rate over the whole coupon.
    * to project (``fixing_date > t``): forward accrual from the projection curve.

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

    # Projection goes through the accrual (interest per unit notional) on the
    # index's own curve.
    proj_curve = market.get_index_curve(leg.index)

    # Overnight and cross-currency legs first: most swaps are not on a term rate.
    if not isinstance(leg, TermRateLeg):
        # Running: started, not yet paid. With a payment lag there can be two: one
        # ended (fully realised) awaiting payment, and the next one accruing.
        running = future_payment & (leg._start_ordinals <= t)
        if not running.any():  # nothing started yet: every unpaid coupon projects
            return future_payment, _projected_flows(leg, proj_curve, future_payment)
        pure_future = future_payment & ~running

        flows = np.zeros(int(future_payment.sum()))
        if pure_future.any():
            flows[pure_future[future_payment]] = _projected_flows(
                leg, proj_curve, pure_future
            )
        flows[running[future_payment]] = _running_flows(
            leg, market, proj_curve, running
        )
        return future_payment, flows

    # Term-rate leg: once fixing_date <= t the rate is locked, wherever
    # start_date falls.
    set_mask = future_payment & (leg._fixing_ordinals <= t)
    if not set_mask.any():  # nothing fixed yet: every unpaid coupon projects
        return future_payment, _projected_flows(leg, proj_curve, future_payment)
    to_proj_mask = future_payment & ~set_mask

    flows = np.zeros(int(future_payment.sum()))
    history = _index_history(leg, market)
    flows[set_mask[future_payment]] = [
        history.get_rate(leg.fixing_dates[i]).get_accrued_interest(
            leg.residuals[i], leg.start_dates[i], leg.end_dates[i]
        )
        + leg.fixed_flows[i]
        for i in np.flatnonzero(set_mask)
    ]
    if to_proj_mask.any():
        flows[to_proj_mask[future_payment]] = _projected_flows(
            leg, proj_curve, to_proj_mask
        )
    return future_payment, flows


def _projected_flows(leg, proj_curve, mask: np.ndarray) -> np.ndarray:
    """Flows of the `mask` coupons, interest projected over their whole period.

    ``fixed_flows`` holds the spread and the notional repaid with the final
    coupon (final exchange): same-currency legs cancel it, cross-currency legs
    exchange it.
    """
    accruals = proj_curve._accrual_at_ordinals(
        leg._start_ordinals[mask], leg._end_ordinals[mask]
    )
    return leg.residuals[mask] * accruals + leg.fixed_flows[mask]


def _running_flows(leg, market: Market, proj_curve, mask: np.ndarray) -> np.ndarray:
    """Flows of the started, unpaid overnight coupons in `mask`.

    The realised part ``[start, min(t, end)]`` comes from the index history
    (compounded on the index's business days, or the ratio of published ICP
    levels), compounded with the projection over ``[t, end]``, which is zero
    for a coupon already ended.

    A scalar loop: `mask` holds one or two coupons, too few for array
    projection to beat scalar ``get_accrual`` (measured ~8 us slower per call).
    """
    t = market.t
    history = _index_history(leg, market)
    flows = []
    for i in np.flatnonzero(mask):
        start, end, residual = leg.start_dates[i], leg.end_dates[i], leg.residuals[i]
        realised_end = min(t, end)
        interest = history.get_accrued_interest(residual, start, realised_end)
        if realised_end < end:
            interest += (residual + interest) * proj_curve.get_accrual(t, end)
        flows.append(interest + leg.fixed_flows[i])
    return np.array(flows)


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
    fx = market._future_fx_rates_at_ordinals(
        CurrencyPair(leg.currency, settlement_currency),
        leg._fx_fixing_ordinals[future_payment],
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
    return market.convert_value_currency(pv, settlement, currency)
