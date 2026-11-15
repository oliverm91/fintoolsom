"""Present values of simplified swaps (:mod:`fintoolsom.derivatives.swaps.simplified`).

Every leg settles in its own currency and pays on its accrual end dates, so a
leg PV is its unpaid flows discounted in the leg currency; no FX fixing is
projected. A non-started swap also skips the split into paid, running and
future coupons: every coupon is projected and discounted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..swaps.legs import FixedLeg
from ._swap_pv import _leg_pv

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency
    from ..swaps.simplified import NonStartedSimplifiedSwap, SimplifiedSwap


def get_simplified_swap_mtm(
    swap: SimplifiedSwap, market: Market, currency: Currency
) -> float:
    """Mark-to-market of a :class:`SimplifiedSwap` in the reporting `currency`.

    Each leg's unpaid flows (:func:`_leg_flows`) are discounted in the leg
    currency under the swap collateral, and the leg PV is converted at spot.

    Parameters
    ----------
    swap : SimplifiedSwap
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
    collateral = swap.collateral_index
    receive, pay = swap.receive_leg, swap.pay_leg
    return market.convert_value_currency(
        _leg_pv(receive, collateral, market), receive.currency, currency
    ) - market.convert_value_currency(
        _leg_pv(pay, collateral, market), pay.currency, currency
    )


def _non_started_leg_pv(leg, collateral, market: Market) -> float:
    """PV of a leg none of whose coupons has started, in the leg currency.

    A floating leg projects every coupon over ``[start, end]`` on its index curve.
    """
    if isinstance(leg, FixedLeg):
        flows = leg.flows
    else:
        accruals = market.get_index_curve(leg.index)._accrual_at_ordinals(
            leg._start_ordinals, leg._end_ordinals
        )
        flows = leg.residuals * accruals + leg.fixed_flows
    dfs = market._dfs_at_ordinals(
        leg.currency, leg._payment_ordinals, collateral=collateral
    )
    return float(np.dot(flows, dfs))


def get_non_started_swap_mtm(
    swap: NonStartedSimplifiedSwap, market: Market, currency: Currency
) -> float:
    """Mark-to-market of a :class:`NonStartedSimplifiedSwap` in `currency`.

    As :func:`get_simplified_swap_mtm`, with every coupon unpaid and projected.

    Parameters
    ----------
    swap : NonStartedSimplifiedSwap
        Swap to value.
    market : Market
        Market at the valuation date, on or before ``swap.effective_date``.
    currency : Currency
        Reporting currency.

    Returns
    -------
    float
        PV of the received leg minus PV of the paid leg, in `currency`.

    Raises
    ------
    ValueError
        If ``market.t`` is after ``swap.effective_date``: a coupon may have
        started or been paid. Curves would still return discount factors there
        (an index curve's spot segment reaches back past its curve date), so the
        value would be wrong without an error.
    """
    if market.t > swap.effective_date:
        raise ValueError(
            f"A NonStartedSimplifiedSwap effective {swap.effective_date} cannot be "
            f"valued on {market.t}, after it starts: use a SimplifiedSwap or a Swap."
        )
    collateral = swap.collateral_index
    receive, pay = swap.receive_leg, swap.pay_leg
    return market.convert_value_currency(
        _non_started_leg_pv(receive, collateral, market), receive.currency, currency
    ) - market.convert_value_currency(
        _non_started_leg_pv(pay, collateral, market), pay.currency, currency
    )
