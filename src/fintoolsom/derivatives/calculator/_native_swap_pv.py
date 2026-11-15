"""Swap leg PVs on the C++ backend (``fintoolsom._native.SwapLegCore``).

Each function mirrors one in :mod:`._swap_pv` or :mod:`._simplified_swap_pv` and gives
the same result to the bit. Python keeps what reads the market: the curves (looked up
in the same order, so a missing one raises the same error), the realised fixings of
started coupons and the known FX fixings. C++ does the arithmetic on the curves:
projected accruals, flows, discount factors (straight from a curve's C++ core when
it has one), FX projection, the par shortcut and the correctly rounded sums.

A leg uses this module when it was created under the ``"cpp"`` backend of
``"derivatives"`` (:mod:`fintoolsom.settings`).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from ... import settings
from ...market.currencies import CurrencyPair
from ..swaps.legs import FixedLeg, TermRateLeg
from . import _swap_pv as _py

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency

_NO_INDEX = np.empty(0, dtype=np.int64)
_NO_FLOWS = np.empty(0)

# fintoolsom._native.SwapLegCore kinds.
_FIXED, _OVERNIGHT, _TERM_RATE = 0, 1, 2


def _core(leg):
    """The leg's SwapLegCore, built on first use (legs are immutable)."""
    core = leg._core
    if core is None:
        fixed = isinstance(leg, FixedLeg)
        term = isinstance(leg, TermRateLeg)
        core = leg._core = settings.native().SwapLegCore(
            _FIXED if fixed else _TERM_RATE if term else _OVERNIGHT,
            leg._start_ordinals,
            leg._end_ordinals,
            leg._payment_ordinals,
            leg._fx_fixing_ordinals,
            leg._fixing_ordinals if term else _NO_INDEX,
            np.asarray(leg.residuals, dtype=float),
            np.asarray(leg.flows if fixed else leg.fixed_flows, dtype=float),
            _NO_FLOWS if fixed else np.asarray(leg.spreads_values, dtype=float),
            False if fixed else leg._par_eligible,
            False if fixed else leg._has_spread,
            leg.currency,
            None if fixed else leg.index,
        )
    return core


def _discount_curves(market: Market, currency: Currency, collateral) -> tuple:
    """Curves of :meth:`Market._dfs_at_ordinals`, resolved in its order: one curve,
    or ``(a, b, c)`` for ``a * b / c``."""
    if collateral is None:
        return (market.get_discount_curve(currency),)
    if collateral.currency == currency:
        return (market.get_index_curve(collateral),)
    return (
        market.get_discount_curve(currency),
        market.get_index_curve(collateral),
        market.get_discount_curve(collateral.currency),
    )


def _projection(leg, core, market: Market, t: int):
    """Projection curve and known flows of an unpaid leg, as :func:`._swap_pv._leg_flows`
    reads them: the index curve first, then the realised fixings (from the history)
    of the coupons started (overnight) or fixed (term rate) by `t`."""
    if isinstance(leg, FixedLeg):
        return None, _NO_INDEX, _NO_FLOWS
    proj = market.get_index_curve(leg.index)
    if not core.has_realised(t):
        return proj, _NO_INDEX, _NO_FLOWS
    future_payment = leg._payment_ordinals > t
    if not isinstance(leg, TermRateLeg):
        running = future_payment & (leg._start_ordinals <= t)
        return (
            proj,
            np.flatnonzero(running).astype(np.int64),
            _py._running_flows(leg, market, proj, running),
        )
    set_mask = future_payment & (leg._fixing_ordinals <= t)
    history = _py._index_history(leg, market)
    index = np.flatnonzero(set_mask).astype(np.int64)
    flows = [
        history.get_rate(leg.fixing_dates[i]).get_accrued_interest(
            leg.residuals[i], leg.start_dates[i], leg.end_dates[i]
        )
        + leg.fixed_flows[i]
        for i in index
    ]
    return proj, index, np.array(flows, dtype=float)


def leg_pv(leg, collateral, market: Market, *, assume_par: bool = False) -> float:
    """:func:`._swap_pv._leg_pv`, or with `assume_par`
    :func:`._simplified_swap_pv._simplified_leg_pv` (eligibility not checked)."""
    core = _core(leg)
    t = market.t.toordinal()
    if (
        not isinstance(leg, FixedLeg)
        and (assume_par or leg._par_eligible)
        and core.nothing_realised(t)
        and _py._discounted_on_index_curve(leg, collateral, market)
    ):
        return core.par_pv(market.get_index_curve(leg.index))
    if not core.has_unpaid(t):
        return 0.0
    proj, index, flows = _projection(leg, core, market, t)
    return core.pv(
        t, proj, index, flows, *_discount_curves(market, leg.currency, collateral)
    )


def leg_pv_non_deliverable(
    leg, collateral, market: Market, settlement_currency: Currency
) -> float:
    """:func:`._swap_pv._leg_pv_non_deliverable` for a leg settled in another currency."""
    core = _core(leg)
    t = market.t.toordinal()
    if not core.has_unpaid(t):
        return 0.0
    proj, index, flows = _projection(leg, core, market, t)
    # Market._future_fx_rates_at_ordinals: known fixings, then the funding curves.
    pair = CurrencyPair(leg.currency, settlement_currency)
    known_until = market.get_fx_known_until(pair)
    last_known = known_until.toordinal()
    known_ordinals, any_unknown = core.known_fx_fixings(t, last_known)
    known_fx = np.array(
        [
            market.get_fx_rate(date.fromordinal(int(o)), pair).value
            for o in known_ordinals
        ],
        dtype=float,
    )
    anchor, base, quote = 0.0, None, None
    if any_unknown:
        anchor = market.get_fx_rate(known_until, pair).value
        base = market.get_discount_curve(leg.currency)
        quote = market.get_discount_curve(settlement_currency)
    return core.pv_non_deliverable(
        t,
        proj,
        index,
        flows,
        last_known,
        known_fx,
        anchor,
        base,
        quote,
        *_discount_curves(market, settlement_currency, collateral),
    )


def non_started_leg_pv(leg, collateral, market: Market) -> float:
    """:func:`._simplified_swap_pv._non_started_leg_pv`."""
    core = _core(leg)
    if isinstance(leg, FixedLeg):
        proj = None
    elif _py._discounted_on_index_curve(leg, collateral, market):
        return core.par_pv(market.get_index_curve(leg.index))
    else:
        proj = market.get_index_curve(leg.index)
    return core.non_started_pv(
        proj, *_discount_curves(market, leg.currency, collateral)
    )


def leg_flows(leg, market: Market) -> tuple[np.ndarray, np.ndarray]:
    """:func:`._swap_pv._leg_flows`."""
    core = _core(leg)
    t = market.t.toordinal()
    future_payment = leg._payment_ordinals > t
    if not future_payment.any():
        return future_payment, np.zeros(0)
    proj, index, flows = _projection(leg, core, market, t)
    return future_payment, core.unpaid_flows(t, proj, index, flows)
