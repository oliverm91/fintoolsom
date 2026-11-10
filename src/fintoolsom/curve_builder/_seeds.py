"""Initial guesses and fixed short ends of the curves a group solves.

The seeds only need to be close (the solve reprices exactly); good seeds keep the
root solves short and in the region where valuation is well behaved.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np

from fintoolsom.rates.Rates import RateConvention  # type: ignore[import-untyped]

from ..derivatives.forwards import Forward
from ..derivatives.swaps import FixedLeg, FloatingLeg, Swap
from ..market.index import Index, OvernightIndex
from ..market.index_history import OvernightHistory
from ..market.market import Market
from ..rates import DAY_BASE, IndexCurve, Rate
from ._errors import InsufficientQuotesError
from ._keys import CurveKey, _built, _curve_currency, _funding_key, _get_maturity

_DAY_BASE = DAY_BASE
# Flat seed rate (annual compounding) of a new curve with no anchor.
_FLAT_GUESS = 0.04


@dataclass
class _IndexFixed:
    """Fixed part of an index curve being solved: its knots, forwards and spot rate.

    ``knots[0]`` is the effective date; ``forwards`` has one entry per segment
    (none for a new overnight curve, whose first segment is the first unknown).
    """

    knots: list[date]
    forwards: list[float]
    spot_rate: Rate | None = None

    def knot_dfs(self, t: date, index: Index) -> list[tuple[date, float]]:
        """``(knot, df)`` of the fixed knots after `t`."""
        if len(self.knots) >= 2:
            dfs = IndexCurve(
                t, index, self.knots, self.forwards, self.spot_rate, precalculate=False
            ).get_dfs(self.knots)
            return [(k, float(df)) for k, df in zip(self.knots, dfs) if k > t]
        eff = self.knots[0]
        if eff <= t:
            return []
        df = (
            1.0
            if self.spot_rate is None
            else float(self.spot_rate.get_discount_factor(t, eff))
        )
        return [(eff, df)]

    def forwards_to(
        self, t: date, index: Index, dates: list[date], dfs: list[float]
    ) -> list[float]:
        """Segment forwards, from the last fixed knot through `dates`, matching `dfs`."""
        points = self.knot_dfs(t, index)
        prev_d, prev_log = self.knots[-1], math.log(points[-1][1]) if points else 0.0
        out = []
        for d, df in zip(dates, dfs):
            log_df = math.log(df)
            out.append((prev_log - log_df) / ((d - prev_d).days / _DAY_BASE))
            prev_d, prev_log = d, log_df
        return out


class _Seeder:
    """Seeds and anchors for the curves of `market`, as of its date.

    Parameters
    ----------
    market : Market
        Market being built.
    risk_free_index : Index
        Market funding index.
    """

    def __init__(self, market: Market, risk_free_index: Index):
        self.market = market
        self.t = market.t
        self.risk_free_index = risk_free_index

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
           :meth:`index_anchor` (or 4%) when the curve has no point at all; an existing
           curve's own extrapolation when it is being extended.

        The seed only needs to be close: the solve then reprices exactly.

        Parameters
        ----------
        instruments : list
            The group's instruments.
        fixed_pillars : dict
            Curve key to fixed ``(date, df)`` points (an index curve: at its knots).
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
            m = _get_maturity(curve, instr, self.risk_free_index)
            return (
                m
                if m is not None and m in free_dates[curve] and m not in implied[curve]
                else None
            )

        # 1. Fixed legs.
        for instr in instruments:
            if not isinstance(instr, Swap):
                continue
            for leg in (instr.receive_leg, instr.pay_leg):
                if (
                    not isinstance(leg, FixedLeg)
                    or (z := self.fixed_leg_zero(leg)) is None
                ):
                    continue
                for curve in free_dates:
                    if _curve_currency(curve) == leg.currency and (
                        m := pinned_free_date(curve, instr)
                    ):
                        implied[curve][m] = z

        # 2. FX forwards, against the other currency's known curve.
        for instr in instruments:
            if isinstance(instr, Forward):
                self.forward_zero(
                    instr, implied, fixed_pillars, free_dates, pinned_free_date
                )

        # 3. Float-float (basis) swaps, against the other curves' known rates.
        for instr in instruments:
            if isinstance(instr, Swap) and all(
                isinstance(leg, FloatingLeg)
                for leg in (instr.receive_leg, instr.pay_leg)
            ):
                self.basis_zero(
                    instr, implied, fixed_pillars, free_dates, pinned_free_date
                )

        # 4. Fill the rest.
        seeds: dict[CurveKey, list[float]] = {}
        for curve, dates in free_dates.items():
            built = _built(self.market, curve)
            points = self.zero_points(curve, implied, fixed_pillars)
            dfs = []
            for d in dates:
                if d in implied[curve]:
                    df = math.exp(-implied[curve][d] * self.tau(d))
                elif built is not None:
                    df = built.get_df(d)
                elif points:
                    taus, zs = zip(*points)
                    df = math.exp(
                        -float(np.interp(self.tau(d), taus, zs)) * self.tau(d)
                    )
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
            wf *= float(
                coupons[0].rate.get_wealth_factor(self.t, coupons[0].start_date)
            )
        for c in coupons:
            wf *= float(c.rate.get_wealth_factor(max(c.start_date, self.t), c.end_date))
        tau = self.tau(coupons[-1].end_date)
        return math.log(wf) / tau if wf > 0 and math.isfinite(wf) and tau > 0 else None

    def forward_zero(
        self, fwd: Forward, implied, fixed_pillars, free_dates, pinned_free_date
    ) -> None:
        """Seed one funding curve of `fwd` by covered interest parity.

        With ``L`` the date the pair's FX is known until and ``T`` the pinned date,
        ``strike / FX(L) = [df_b(T)/df_b(L)] / [df_q(T)/df_q(L)]``. The unknown
        side is taken flat over ``[L, T]`` and that zero rate is extended back to
        the curve date. Skipped when neither or both sides are known.
        """
        pair = fwd.currency_pair
        keys = (
            _funding_key(self.risk_free_index, pair.base_currency),
            _funding_key(self.risk_free_index, pair.quote_currency),
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
            log_known = self.log_df_change(
                known, known_until, m, implied, fixed_pillars, free_dates
            )
            if span <= 0 or log_known is None or fwd.strike <= 0:
                continue
            log_ratio = math.log(fwd.strike / fx_known)
            # base unknown: -z*span = log_ratio + log_known; quote unknown: -z*span = log_known - log_ratio.
            implied[unknown][m] = -(sign * log_ratio + log_known) / span
            return

    def basis_zero(
        self, swap: Swap, implied, fixed_pillars, free_dates, pinned_free_date
    ) -> None:
        """Seed one curve of a float-float swap from its spreads.

        At the pinned date, ``z_I_r + s_r - z_D_r = z_I_p + s_p - z_D_p`` (index
        ``I`` and funding discount curve ``D`` of each leg), solved for the single
        free curve in it. Collateral is ignored: it only shifts the seed. Skipped
        when there is not exactly one unknown or another curve is not known.
        """
        coef: dict[CurveKey, float] = defaultdict(
            float
        )  # coefficient of each zero rate
        spreads = 0.0
        for leg, side in ((swap.receive_leg, 1.0), (swap.pay_leg, -1.0)):
            coef[leg.index] += side
            coef[
                _funding_key(self.risk_free_index, swap.leg_settlement_currency(leg))
            ] -= side
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
            log_df = self.log_df_change(
                curve, self.t, m, implied, fixed_pillars, free_dates
            )
            if log_df is None:
                return
            total += k * (-log_df / tau)
        implied[unknown][m] = -total / coef[unknown]

    def log_df_change(
        self, curve, start, end, implied, fixed_pillars, free_dates
    ) -> float | None:
        """``log(df(end) / df(start))`` on a curve already known, else None."""
        built = _built(self.market, curve)
        if built is not None and curve not in free_dates:
            return math.log(built.get_df(end)) - math.log(built.get_df(start))
        if (
            curve in free_dates and implied[curve]
        ):  # being solved: use its step-1 seeds.
            taus, zs = zip(*self.zero_points(curve, implied, fixed_pillars))

            def log_df(d: date) -> float:
                return -float(np.interp(self.tau(d), taus, zs)) * self.tau(d)

            return log_df(end) - log_df(start)
        if built is not None:  # extended curve with nothing implied: extrapolate.
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

    def index_anchor(self, index: Index) -> tuple[_IndexFixed, float]:
        """Fixed short end of a new index curve, and a flat rate to seed it with.

        * Overnight history: effective date the next business day, today's overnight
          rate (:meth:`OvernightHistory.spot_overnight_rate`) as the spot rate.
        * Term-rate history: effective date ``t + spot_lag`` and a first segment
          ``[effective, effective + term]`` fixed at today's fixing; no spot rate.
        * No usable history, overnight index: effective date the curve date.

        Raises
        ------
        InsufficientQuotesError
            For a non-overnight index with no rate history.
        """
        try:
            history = self.market.get_index(index.name)
        except KeyError:
            history = None
        if isinstance(history, OvernightHistory):
            try:
                spot = history.spot_overnight_rate(self.t).copy()
            except KeyError:  # not enough fixings to imply today's rate
                return _IndexFixed([self.t], []), _FLAT_GUESS
            effective = index.get_maturity(self.t)  # overnight: next business day.
            annual = spot.copy()
            annual.convert_rate_convention(RateConvention(), self.t, effective)
            return _IndexFixed([effective], [], spot), annual.rate
        if hasattr(history, "get_rate"):
            effective = index.calendar.add_business_days(
                self.t, getattr(index, "spot_lag", 0)
            )
            end = index.get_maturity(effective)
            wf = history.get_rate(self.t).get_wealth_factor(effective, end)  # type: ignore[union-attr]
            forward = math.log(wf) / ((end - effective).days / _DAY_BASE)
            return _IndexFixed([effective, end], [forward]), forward
        if isinstance(index, OvernightIndex):
            return _IndexFixed([self.t], []), _FLAT_GUESS
        raise InsufficientQuotesError(
            f"Index '{index.name}' has no rate history to anchor its short end."
        )

    def seed_df(self, rate: float, maturity: date) -> float:
        """Annually compounded act/365 discount factor at `rate` to `maturity`."""
        return (1 + rate) ** (-(maturity - self.t).days / _DAY_BASE)
