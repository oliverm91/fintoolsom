"""Initial guesses and fixed short ends of the curves a group solves.

The seeds only need to be close (the solve reprices exactly); good seeds keep the
root solves short and in the region where valuation is well behaved.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

import numpy as np

from fintoolsom.rates.Rates import RateConvention  # type: ignore[import-untyped]

from ..derivatives.forwards import Forward
from ..derivatives.swaps import FixedLeg, FloatingLeg, Swap
from ..market.index import Index
from ..market.index_history import OvernightHistory
from ..market.market import CurveKey, Market, funding_key
from ._errors import InsufficientQuotesError
from ._keys import _curve_currency, _get_maturity

_DAY_BASE = 365


class _Seeder:
    """Seeds and anchors for the curves of `market`, as of its date.

    Parameters
    ----------
    market : Market
        Market being built.
    riskless_index : Index
        Market funding index.
    """

    def __init__(self, market: Market, riskless_index: Index):
        self.market = market
        self.t = market.t
        self.riskless_index = riskless_index

    def tau(self, d: date) -> float:
        """Act/365 year fraction from the curve date to `d`."""
        return (d - self.t).days / _DAY_BASE

    def seed_free_pillars(self, instruments, fixed_pillars, free_dates, flat_guess):
        """Initial discount factors of the free pillars, read off the quotes.

        Each free pillar gets a continuously compounded zero rate implied by an
        instrument that pins it, then the remaining ones are filled in:

        1. A fixed leg in the curve's currency: its coupons compounded from the
           curve date to the leg end (par rate ~ zero rate).
        2. An FX forward or NDF whose other currency's curve is known (built, or
           seeded in step 1): covered interest parity on its strike.
        3. A float-float (basis) swap whose other curves are known: each floating
           leg is worth about par + (z_index + spread - z_discount) * annuity, and
           the swap is at par.
        4. Other pillars: zero rate interpolated linearly in time between the
           known points of the same curve (short-end anchor, existing pillars,
           steps 1-3), flat beyond them; the flat guess of
           :meth:`short_end_anchor` when the curve has no point at all; an existing
           curve's own extrapolation when it is being extended.

        The seed only needs to be close: the solve then reprices exactly.

        Parameters
        ----------
        instruments : list
            The group's instruments.
        fixed_pillars : dict
            Curve key to fixed ``(date, df)`` pillars.
        free_dates : dict
            Curve key to free pillar dates.
        flat_guess : dict
            New curve key to its flat fallback rate (annual compounding).

        Returns
        -------
        dict
            Curve key to initial dfs of its free pillars, in `free_dates` order.
        """
        implied: dict[CurveKey, dict[date, float]] = defaultdict(dict)

        def pinned_free_date(curve, instr) -> date | None:
            m = _get_maturity(curve, instr, self.riskless_index)
            return m if m is not None and m in free_dates[curve] and m not in implied[curve] else None

        # 1. Fixed legs.
        for instr in instruments:
            if not isinstance(instr, Swap):
                continue
            for leg in (instr.receive_leg, instr.pay_leg):
                if not isinstance(leg, FixedLeg) or (z := self.fixed_leg_zero(leg)) is None:
                    continue
                for curve in free_dates:
                    if _curve_currency(curve) == leg.currency and (m := pinned_free_date(curve, instr)):
                        implied[curve][m] = z

        # 2. FX forwards, against the other currency's known curve.
        for instr in instruments:
            if isinstance(instr, Forward):
                self.forward_zero(instr, implied, fixed_pillars, free_dates, pinned_free_date)

        # 3. Float-float (basis) swaps, against the other curves' known rates.
        for instr in instruments:
            if isinstance(instr, Swap) and all(
                isinstance(leg, FloatingLeg) for leg in (instr.receive_leg, instr.pay_leg)
            ):
                self.basis_zero(instr, implied, fixed_pillars, free_dates, pinned_free_date)

        # 4. Fill the rest.
        seeds: dict[CurveKey, list[float]] = {}
        for curve, dates in free_dates.items():
            built = self.market.curves.get(curve)
            points = self.zero_points(curve, implied, fixed_pillars)
            dfs = []
            for d in dates:
                if d in implied[curve]:
                    df = math.exp(-implied[curve][d] * self.tau(d))
                elif built is not None:
                    df = built.get_df(d)
                elif points:
                    taus, zs = zip(*points)
                    df = math.exp(-float(np.interp(self.tau(d), taus, zs)) * self.tau(d))
                else:
                    df = self.seed_df(flat_guess[curve], d)
                dfs.append(df)
            seeds[curve] = dfs
        return seeds

    def fixed_leg_zero(self, leg: FixedLeg) -> float | None:
        """Zero rate (continuous, act/365) to the leg end, compounding its coupons.

        The first coupon's rate also covers any gap between the curve date and the
        leg start. None when the leg has no coupon left or the rate is unusable.
        """
        coupons = [c for c in leg.coupons if c.end_date > self.t]
        if not coupons:
            return None
        wf = 1.0
        if coupons[0].start_date > self.t:
            wf *= float(coupons[0].rate.get_wealth_factor(self.t, coupons[0].start_date))
        for c in coupons:
            wf *= float(c.rate.get_wealth_factor(max(c.start_date, self.t), c.end_date))
        tau = self.tau(coupons[-1].end_date)
        return math.log(wf) / tau if wf > 0 and math.isfinite(wf) and tau > 0 else None

    def forward_zero(self, fwd: Forward, implied, fixed_pillars, free_dates, pinned_free_date) -> None:
        """Seed one funding curve of `fwd` by covered interest parity.

        With ``L`` the date the pair's FX is known until and ``T`` the pinned date,
        ``strike / FX(L) = [df_b(T)/df_b(L)] / [df_q(T)/df_q(L)]``. The unknown
        side is taken flat over ``[L, T]`` and that zero rate is extended back to
        the curve date. Skipped when neither or both sides are known.
        """
        pair = fwd.currency_pair
        keys = (
            funding_key(self.riskless_index, pair.base_currency),
            funding_key(self.riskless_index, pair.quote_currency),
        )
        known_until = self.market.get_fx_known_until(pair)
        try:
            fx_known = self.market.get_fx_rate(known_until, pair).value
        except KeyError:
            return
        for unknown, known, sign in ((keys[0], keys[1], 1.0), (keys[1], keys[0], -1.0)):
            if unknown not in free_dates or not (m := pinned_free_date(unknown, fwd)):
                continue
            span = self.tau(m) - self.tau(known_until)
            log_known = self.log_df_change(known, known_until, m, implied, fixed_pillars, free_dates)
            if span <= 0 or log_known is None or fwd.strike <= 0:
                continue
            log_ratio = math.log(fwd.strike / fx_known)
            # base unknown: -z*span = log_ratio + log_known; quote unknown: -z*span = log_known - log_ratio.
            implied[unknown][m] = -(sign * log_ratio + log_known) / span
            return

    def basis_zero(self, swap: Swap, implied, fixed_pillars, free_dates, pinned_free_date) -> None:
        """Seed one curve of a float-float swap from its spreads.

        At the pinned date, ``z_I_r + s_r - z_D_r = z_I_p + s_p - z_D_p`` (index
        ``I`` and funding discount curve ``D`` of each leg), solved for the single
        free curve in it. Collateral is ignored: it only shifts the seed. Skipped
        when there is not exactly one unknown or another curve is not known.
        """
        coef: dict[CurveKey, float] = defaultdict(float)  # coefficient of each zero rate
        spreads = 0.0
        for leg, side in ((swap.receive_leg, 1.0), (swap.pay_leg, -1.0)):
            coef[leg.index] += side
            coef[funding_key(self.riskless_index, swap.leg_settlement_currency(leg))] -= side
            spreads += side * getattr(leg.spread, "rate", 0.0)
        unknowns = [
            (curve, m)
            for curve, k in coef.items()
            if k != 0 and curve in free_dates and (m := pinned_free_date(curve, swap))
        ]
        if len(unknowns) != 1:
            return
        unknown, m = unknowns[0]
        tau = self.tau(m)
        total = spreads
        for curve, k in coef.items():
            if curve == unknown or k == 0:
                continue
            log_df = self.log_df_change(curve, self.t, m, implied, fixed_pillars, free_dates)
            if log_df is None:
                return
            total += k * (-log_df / tau)
        implied[unknown][m] = -total / coef[unknown]

    def log_df_change(self, curve, start, end, implied, fixed_pillars, free_dates) -> float | None:
        """``log(df(end) / df(start))`` on a curve already known, else None."""
        if curve in self.market.curves and curve not in free_dates:
            built = self.market.curves[curve]
            return math.log(built.get_df(end)) - math.log(built.get_df(start))
        if curve in free_dates and implied[curve]:  # being solved: use its step-1 seeds.
            taus, zs = zip(*self.zero_points(curve, implied, fixed_pillars))
            def log_df(d: date) -> float:
                return -float(np.interp(self.tau(d), taus, zs)) * self.tau(d)
            return log_df(end) - log_df(start)
        if curve in self.market.curves:  # extended curve with nothing implied: extrapolate.
            built = self.market.curves[curve]
            return math.log(built.get_df(end)) - math.log(built.get_df(start))
        return None

    def zero_points(self, curve, implied, fixed_pillars) -> list[tuple[float, float]]:
        """Sorted ``(tau, zero rate)`` points known on `curve`: fixed pillars and seeds."""
        points = {
            self.tau(d): -math.log(df) / self.tau(d)
            for d, df in fixed_pillars.get(curve, ())
            if d > self.t
        }
        points.update({self.tau(d): z for d, z in implied[curve].items()})
        return sorted(points.items())

    def short_end_anchor(self, curve: CurveKey) -> tuple[tuple[date, float] | None, float]:
        """Fixed short-end pillar and rate guess for a new curve.

        An overnight index curve is anchored at its next business day with the
        rate from :meth:`OvernightHistory.spot_overnight_rate` (today's fixing, or
        the latest realised one-day rate of a price index such as ICP).

        Parameters
        ----------
        curve : Index or Currency
            New curve.

        Returns
        -------
        anchor : tuple of (date, float) or None
            Fixed pillar; None when the curve has no overnight history.
        guess : float
            Flat rate used to seed the free pillars (4% without an anchor).
        """
        if isinstance(curve, Index):
            curve_index = curve
            try:
                history = self.market.get_index(curve_index.name)
                if isinstance(history, OvernightHistory):
                    rate = history.spot_overnight_rate(self.t).copy()
                    anchor_date = curve_index.get_maturity(self.t)  # overnight: next business day.
                    rate.convert_rate_convention(RateConvention(), self.t, anchor_date)
                    return (anchor_date, self.seed_df(rate.rate, anchor_date)), rate.rate
            except KeyError:
                pass
        return None, 0.04

    def seed_df(self, rate: float, maturity: date) -> float:
        """Annually compounded act/365 discount factor at `rate` to `maturity`."""
        return (1 + rate) ** (-(maturity - self.t).days / _DAY_BASE)

    def projection_anchor(self, index: Index) -> tuple[list[date], list[float]]:
        """Fixed spot segment ``[t + spot_lag, maturity]`` of a new projection curve.

        Its forward comes from the index fixing of the curve date.

        Raises
        ------
        InsufficientQuotesError
            If the index has no rate history.
        """
        anchor_start = index.calendar.add_business_days(self.t, getattr(index, "spot_lag", 0))
        anchor_end = index.get_maturity(anchor_start)
        try:
            history = self.market.get_index(index.name)
        except KeyError:
            history = None
        if not hasattr(history, "get_rate"):
            raise InsufficientQuotesError(
                f"Projection index '{index.name}' has no rate history to anchor its short end."
            )
        wf = history.get_rate(self.t).get_wealth_factor(anchor_start, anchor_end)  # type: ignore[union-attr]
        return [anchor_start, anchor_end], [math.log(wf) / ((anchor_end - anchor_start).days / _DAY_BASE)]
