"""Swap leg flows and present values, deliverable or not."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..swaps.legs import FixedLeg, FloatingLeg, TermRateLeg
from ..swaps.swaps import LegSide
from ...market.currencies import CurrencyPair

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency
    from ..swaps.swaps import Swap


def _sum_products(a: np.ndarray, b: np.ndarray) -> float:
    """``sum(a * b)``, correctly rounded (:func:`math.fsum`).

    PVs sum with ``math.fsum`` rather than ``np.dot``: BLAS kernels choose their own
    summation order (and fused multiply-adds), so ``np.dot`` cannot be reproduced
    exactly elsewhere; the correctly rounded sum is the same in the C++ backend.
    """
    return math.fsum((a * b).tolist())


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

    # Projection goes through the market's accrual (interest per unit notional) of
    # the leg's index, on that index's curve.

    # Overnight and cross-currency legs first: most swaps are not on a term rate.
    if not isinstance(leg, TermRateLeg):
        # Running: started, not yet paid. With a payment lag there can be two: one
        # ended (fully realised) awaiting payment, and the next one accruing.
        running = future_payment & (leg._start_ordinals <= t)
        if not running.any():  # nothing started yet: every unpaid coupon projects
            return future_payment, _projected_flows(leg, market, future_payment)
        pure_future = future_payment & ~running

        flows = np.zeros(int(future_payment.sum()))
        if pure_future.any():
            flows[pure_future[future_payment]] = _projected_flows(
                leg, market, pure_future
            )
        flows[running[future_payment]] = _running_flows(leg, market, running)
        return future_payment, flows

    # Term-rate leg: once fixing_date <= t the rate is locked, wherever
    # start_date falls.
    set_mask = future_payment & (leg._fixing_ordinals <= t)
    if not set_mask.any():  # nothing fixed yet: every unpaid coupon projects
        return future_payment, _projected_flows(leg, market, future_payment)
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
            leg, market, to_proj_mask
        )
    return future_payment, flows


def _projected_flows(leg, market: Market, mask: np.ndarray) -> np.ndarray:
    """Flows of the `mask` coupons, interest projected over their whole period.

    The index accrues over each coupon's projection window: its period, or for a
    term-rate coupon its fixing window, whose forward is then accrued over the
    period (:meth:`FloatingLeg._projected_interest`). One vectorised curve read for
    all the coupons.

    ``fixed_flows`` holds the spread and the notional repaid with the final
    coupon (final exchange): same-currency legs cancel it, cross-currency legs
    exchange it.
    """
    accruals = market._accrual_at_ordinals(
        leg.index, leg._window_start_ordinals[mask], leg._window_end_ordinals[mask]
    )
    return (
        leg.residuals[mask] * leg._projected_interest(accruals, mask)
        + leg.fixed_flows[mask]
    )


def _running_flows(leg, market: Market, mask: np.ndarray) -> np.ndarray:
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
            interest += (residual + interest) * market.get_accrual(leg.index, t, end)
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
    if (
        isinstance(leg, FloatingLeg)
        and leg._par_eligible
        and _nothing_realised(leg, market.t.toordinal())
        and _discounted_on_index_curve(leg, collateral, market)
    ):
        return _par_leg_pv(leg, market)
    return _flows_pv(leg, collateral, market)


def _flows_pv(leg, collateral, market: Market) -> float:
    """PV of a leg settled in its own currency: every unpaid flow discounted."""
    future_payment, flows = _leg_flows(leg, market)
    if not future_payment.any():
        return 0.0
    dfs = market._dfs_at_ordinals(
        leg.currency, leg._payment_ordinals[future_payment], collateral=collateral
    )
    return _sum_products(flows, dfs)


def _nothing_realised(leg, t: int) -> bool:
    """No coupon of the floating `leg` has started accruing (overnight) or fixed
    (term rate) by the ordinal `t`: every flow is projected."""
    first = (
        leg._fixing_ordinals if isinstance(leg, TermRateLeg) else leg._start_ordinals
    )
    return first.size > 0 and first[0] > t


def _discounted_on_index_curve(leg, collateral, market: Market) -> bool:
    """The curve discounting the leg is the one projecting its index.

    As :meth:`Market._dfs_at_ordinals` resolves it: the collateral's curve when the
    collateral is in the leg currency, else (uncollateralised) the risk-free index's
    curve when the risk-free index is in the leg currency. Curves are stored by index
    name.
    """
    index = collateral if collateral is not None else market.risk_free_index
    return (
        index is not None
        and index.currency == leg.currency
        and index.name == leg.index.name
    )


def _par_leg_pv(leg, market: Market) -> float:
    """PV of a par-eligible floating leg discounted on its index curve, before any
    coupon is realised.

    With ``df`` the index curve, coupon ``i`` pays at its end ``e_i``
    ``residual_i * (df(s_i) / df(e_i) - 1) + spread_i + amortization_i``. Since
    ``s_{i+1} = e_i`` and ``residual_{i+1} = residual_i - amortization_i``, the index
    interest and amortizations discount to ``residuals[0] * df(s_0)``; only the
    spreads need the end-date dfs.
    """
    curve = market.get_index_curve(leg.index)
    pv = leg.residuals[0] * curve._dfs_at_ordinals(leg._start_ordinals[:1])[0]
    if leg._has_spread:
        pv += _sum_products(
            leg.spreads_values, curve._dfs_at_ordinals(leg._end_ordinals)
        )
    return float(pv)


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
    return math.fsum((flows * fx * dfs).tolist())


def get_swap_mtm(swap: Swap, market: Market, currency: Currency) -> float:
    """Swap mark-to-market in the reporting `currency`.

    Each leg settles in :meth:`Swap.leg_settlement_currency`. A leg settled in
    its own currency is valued there; otherwise (non-deliverable swap, or a
    UF leg, always paid in CLP) every flow is converted at its FX fixing and
    discounted in the settlement currency. Each leg PV is then converted to
    `currency` at spot. The final notional exchange is the last coupon's
    amortization; a deliverable swap with ``initial_exchange`` also pays each
    leg's first residual on its first start date (:func:`_initial_exchange_pv`).

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
    if swap.initial_exchange:  # deliverable: settlement is the leg currency
        pv = _with_initial_exchange(pv, leg, swap.collateral_index, market)
    return market.convert_value_currency(pv, settlement, currency)


def _initial_exchange_pv(leg, collateral, market: Market) -> float | None:
    """PV, in the leg currency, of the leg paying its first residual on its first
    start date (a deliverable swap's initial exchange); None once that date is on or
    before ``market.t``: paid, like a coupon."""
    start = leg._start_ordinals[:1]
    if start.size == 0 or start[0] <= market.t.toordinal():
        return None
    df = market._dfs_at_ordinals(leg.currency, start, collateral=collateral)[0]
    return -float(leg.residuals[0]) * df


def _with_initial_exchange(pv: float, leg, collateral, market: Market) -> float:
    """`pv` plus the leg's unpaid initial exchange. ``ValuationBatch`` adds it the
    same way, after the leg's flows, so the two agree to the bit."""
    exchange = _initial_exchange_pv(leg, collateral, market)
    return pv if exchange is None else float(pv + exchange)
