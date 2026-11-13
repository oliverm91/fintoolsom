"""Swap leg risk: duration and DV01."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from ..swaps.legs import FixedLeg, FloatingLeg, TermRateLeg
from ._swap_pv import _leg_flows, get_leg_pv

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency
    from ..swaps.swaps import LegSide, Swap


def _rate_sensitive_flows(leg, market: Market) -> tuple[list[date], np.ndarray]:
    """Flows of `leg` that stay fixed under a parallel rate shift, with their dates.

    A fixed leg: all unpaid flows. A floating leg is replaced by its
    floating-rate-note equivalent: coupons whose rate is already set (fixed
    term coupons, the running overnight coupon) are kept; from the first
    unset coupon on, the projected interest and the amortizations are
    replaced by the residual notional at that coupon's start (the next reset),
    and only the spread flows remain.
    """
    future_payment, flows = _leg_flows(leg, market)
    idx = np.where(future_payment)[0]
    if isinstance(leg, FixedLeg) or len(idx) == 0:
        return [leg.payment_dates[i] for i in idx], flows

    if isinstance(leg, TermRateLeg):
        is_set = [leg.fixing_dates[i] <= market.t for i in idx]
    else:
        is_set = [leg.start_dates[i] <= market.t for i in idx]
    unset = [k for k, s in enumerate(is_set) if not s]
    if not unset:
        return [leg.payment_dates[i] for i in idx], flows

    first = unset[0]
    dates_out = [leg.payment_dates[i] for i in idx[:first]]
    flows_out = list(flows[:first])
    reset = idx[first]
    dates_out.append(leg.start_dates[reset])
    flows_out.append(leg.residuals[reset])
    for i in idx[first:]:
        dates_out.append(leg.payment_dates[i])
        flows_out.append(leg.spreads_values[i])
    return dates_out, np.array(flows_out, dtype=float)


def get_leg_duration(swap: Swap, side: LegSide, market: Market) -> float:
    """Macaulay duration of one leg of `swap`, in act/365 years.

    ``sum(tau_i * PV_i) / sum(PV_i)`` over the leg's rate-sensitive flows
    (:func:`_rate_sensitive_flows`), discounted in the leg currency under the
    swap collateral. A fixed leg gets its bond-like duration; a floating leg,
    about the time to its next reset (plus its spread annuity), since its
    projected coupons move with the curve.

    Parameters
    ----------
    swap : Swap
        Swap holding the leg (collateral).
    side : LegSide
        Which leg: ``LegSide.RECEIVE`` or ``LegSide.PAY``.
    market : Market
        Market at the valuation date.

    Returns
    -------
    float
        Duration in years; 0 for a leg with no unpaid flow.
    """
    leg = swap.leg(side)
    pay_dates, flows = _rate_sensitive_flows(leg, market)
    if not pay_dates:
        return 0.0
    dfs = market.get_dfs(leg.currency, pay_dates, collateral=swap.collateral_index)
    pvs = flows * dfs
    taus = np.array([(d - market.t).days / 365 for d in pay_dates])
    total = float(np.sum(pvs))
    return float(np.sum(taus * pvs) / total) if total != 0 else 0.0


def get_leg_dv01(
    swap: Swap, side: LegSide, market: Market, currency: Currency
) -> float:
    """DV01 estimate of one fixed leg: ``-PV * duration / 10_000``, in `currency`.

    The change of the leg PV for a 1bp parallel rise of its rates. For a leg
    near par, ``PV ~ notional``, so this is ``-notional * duration / 10_000``.

    Parameters
    ----------
    swap : Swap
        Swap holding the leg.
    side : LegSide
        Which leg: ``LegSide.RECEIVE`` or ``LegSide.PAY``; must be a fixed leg.
    market : Market
        Market at the valuation date.
    currency : Currency
        Reporting currency.

    Returns
    -------
    float

    Raises
    ------
    NotImplementedError
        For a floating leg.
    """
    leg = swap.leg(side)
    if isinstance(leg, FloatingLeg):
        raise NotImplementedError(
            f"DV01 is only implemented for fixed legs, got {type(leg).__name__}."
        )
    pv = get_leg_pv(swap, side, market, currency)
    return -pv * get_leg_duration(swap, side, market) / 10_000
