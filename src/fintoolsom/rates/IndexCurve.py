"""Curve of an interest index, stored as forward rates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np

from .Curve import DAY_BASE, DateArg, InterpolationMethod
from .PillarCurve import PillarCurve
from .Rates import Rate

if TYPE_CHECKING:
    from ..market.index import Index


@dataclass
class IndexCurve(PillarCurve):
    """Curve of `index`: it projects the index and can discount with it.

    The state is one continuously compounded (Act/365) forward rate per segment
    ``[knot_dates[i], knot_dates[i + 1]]``; discount factors are derived from it::

        df(d) = df_spot * exp(-∫_{effective_date}^{d} f)

    ``effective_date`` is ``knot_dates[0]``: the next business day for an overnight
    index, ``t + spot_lag`` for a term index. ``spot_rate`` covers ``[curve_date,
    effective_date]`` in its own convention (``df_spot``); without one the first
    forward is extended back to ``curve_date``. Past the last knot the last forward
    is extended flat.

    Parameters
    ----------
    curve_date : date
        Curve date (df = 1 there).
    index : Index
        Index the curve belongs to; its name is the curve id.
    knot_dates : list of date
        Strictly increasing, the first one on or after `curve_date`.
    forward_rates : array-like
        Average forward of each segment: ``len(knot_dates) - 1`` values.
    spot_rate : Rate, optional
        Rate from `curve_date` to the effective date (e.g. today's overnight fixing).
    interpolation_method : InterpolationMethod, optional
        ``PiecewiseConstant`` by default.
    precalculate : bool, optional
        Keyword only. Store the df of every day up to the last knot on creation (the
        default). Creation is slower and lookups faster; False skips it for a curve
        that is created many times and read little. :meth:`precalculate_dfs` builds
        the grid later.
    """

    index: "Index"
    knot_dates: list[date]
    forward_rates: np.ndarray
    spot_rate: Rate | None = None
    interpolation_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant

    def _log_df_first_knot(self) -> float:
        if self.spot_rate is None or self.knot_dates[0] == self.curve_date:
            return super()._log_df_first_knot()
        return math.log(
            float(
                self.spot_rate.get_discount_factor(self.curve_date, self.knot_dates[0])
            )
        )

    # ── Curve ────────────────────────────────────────────────────────────────

    def get_id(self) -> str:
        return self.index.name

    @property
    def effective_date(self) -> date:
        """First knot: where the forwards start."""
        return self.knot_dates[0]

    def _dfs_from_days(self, days: np.ndarray) -> np.ndarray:
        dfs = super()._dfs_from_days(days)
        if self.spot_rate is not None and self.knot_dates[0] > self.curve_date:
            spot = days / DAY_BASE < self._knot_tau[0]
            if spot.any():
                dfs[spot] = self.spot_rate.get_discount_factor(
                    self.curve_date,
                    [self.curve_date + timedelta(days=int(n)) for n in days[spot]],
                )
        return dfs

    def parallel_bump_bps(self, bps: float) -> IndexCurve:
        """Copy with every forward, and the spot segment, shifted by `bps` (continuous).

        The spot rate keeps its convention: its value moves so that the df to the
        effective date shifts exactly like a continuous zero rate.
        """
        shift = bps / 10_000
        spot = self.spot_rate
        if spot is not None and self.knot_dates[0] > self.curve_date:
            rc = spot.rate_convention
            df = math.exp(self._log_df_effective - shift * float(self._knot_tau[0]))
            yf = rc.day_count_convention.get_time_fraction(
                self.curve_date, self.knot_dates[0], rc.time_fraction_base
            )
            spot = Rate(
                rc.copy(), float(rc.interest_convention.get_rate_from_df(df, yf))
            )
        return IndexCurve(
            self.curve_date,
            self.index,
            list(self.knot_dates),
            self.forward_rates + shift,
            spot,
            self.interpolation_method,
            precalculate=self.precalculate,
        )

    def get_aged_curve(self, date: date) -> IndexCurve:
        """The aged curve, without the spot rate: its first knot is the curve date."""
        aged = super().get_aged_curve(date)
        aged.spot_rate = None
        return aged

    def copy(self) -> IndexCurve:
        return IndexCurve(
            self.curve_date,
            self.index,
            list(self.knot_dates),
            self.forward_rates.copy(),
            None if self.spot_rate is None else self.spot_rate.copy(),
            self.interpolation_method,
            precalculate=self.precalculate,
        )

    # ── Projection ───────────────────────────────────────────────────────────

    def get_accrual(self, start: DateArg, end: DateArg) -> float | np.ndarray:
        """Interest per unit notional over ``[start, end]``: wealth factor - 1."""
        return self.get_wealth_factor(start, end) - 1

    def _accrual_at_ordinals(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """:meth:`get_accrual` for aligned dates already held as integer ordinals."""
        return 1.0 / (self._dfs_at_ordinals(ends) / self._dfs_at_ordinals(starts)) - 1

    def __str__(self) -> str:
        points = ", ".join(
            f"({d}|{f:.6f})" for d, f in zip(self.knot_dates[1:], self.forward_rates)
        )
        return (
            f"IndexCurve {self.index.name} ({self.interpolation_method.name}) date {self.curve_date} "
            f"effective {self.effective_date} [{points}]"
        )
