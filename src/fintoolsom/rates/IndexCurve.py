"""Curve of an interest index, stored as forward rates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline

from .Curve import (
    DAY_BASE,
    DailyGridCurve,
    DateArg,
    InterpolationMethod,
    year_fractions,
)
from .Rates import Rate

if TYPE_CHECKING:
    from ..market.index import Index


@dataclass
class IndexCurve(DailyGridCurve):
    """Curve of `index`: it projects the index and can discount with it.

    The state is one continuously compounded (Act/365) forward rate per segment
    ``[knot_dates[i], knot_dates[i + 1]]``; discount factors are derived from it::

        df(d) = df_spot * exp(-∫_{effective_date}^{d} f)

    ``effective_date`` is ``knot_dates[0]``: the next business day for an overnight
    index, ``t + spot_lag`` for a term index. ``spot_rate`` covers ``[curve_date,
    effective_date]`` in its own convention (``df_spot``); without one the first
    forward is extended back to ``curve_date``. Past the last knot the last forward
    is extended flat.

    What ``forward_rates[i]`` means follows the interpolation:

    * ``PiecewiseConstant``: the forward over the whole segment.
    * ``CubicSpline``: the segment's average forward. The dfs at the knots are those
      of ``PiecewiseConstant``, joined by a natural spline on ``log(df)``.
    * ``Linear``: the forward at ``knot_dates[i + 1]``, the end of the segment. The
      forward is linear between knots and flat over the first segment.

    Parameters
    ----------
    curve_date : date
        Curve date (df = 1 there).
    index : Index
        Index the curve belongs to; its name is the curve id.
    knot_dates : list of date
        Strictly increasing, the first one on or after `curve_date`.
    forward_rates : array-like
        One forward per segment: ``len(knot_dates) - 1`` values.
    spot_rate : Rate, optional
        Rate from `curve_date` to the effective date (e.g. today's overnight fixing).
    interpolation_method : InterpolationMethod, optional
        Forward-space methods only; ``PiecewiseConstant`` by default.
    precalculate : bool, optional
        Keyword only. Store the df of every day up to the last knot on creation (the
        default). Creation is slower and lookups faster; False skips it for a curve
        that is created many times and read little. :meth:`precalculate_dfs` builds
        the grid later.
    """

    allowed_methods: ClassVar[frozenset[InterpolationMethod]] = frozenset(
        {
            InterpolationMethod.PiecewiseConstant,
            InterpolationMethod.Linear,
            InterpolationMethod.CubicSpline,
        }
    )

    curve_date: date
    index: "Index"
    knot_dates: list[date]
    forward_rates: np.ndarray
    spot_rate: Rate | None = None
    interpolation_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant
    precalculate: bool = field(default=True, kw_only=True, repr=False, compare=False)

    _knot_tau: np.ndarray = field(init=False, repr=False)
    _cum: np.ndarray = field(init=False, repr=False)
    _log_df_effective: float = field(init=False, repr=False)
    _forward_integral: BSpline | CubicSpline | None = field(
        init=False, repr=False, default=None
    )
    _last_forward: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        self._check_method()
        self.knot_dates = list(self.knot_dates)
        self.forward_rates = np.asarray(self.forward_rates, dtype=float)
        if len(self.knot_dates) < 2:
            raise ValueError(
                f"An IndexCurve needs at least 2 knots (one segment); got {len(self.knot_dates)}."
            )
        if len(self.forward_rates) != len(self.knot_dates) - 1:
            raise ValueError(
                f"forward_rates must have one entry per segment: expected {len(self.knot_dates) - 1}, "
                f"got {len(self.forward_rates)}."
            )
        if any(b <= a for a, b in zip(self.knot_dates, self.knot_dates[1:])):
            raise ValueError(
                f"knot_dates must be strictly increasing; got {self.knot_dates}."
            )
        if self.knot_dates[0] < self.curve_date:
            raise ValueError(
                f"The effective date {self.knot_dates[0]} is before the curve date {self.curve_date}."
            )
        self._knot_tau = year_fractions(self.curve_date, self.knot_dates)
        self._cum = np.concatenate(
            ([0.0], np.cumsum(self.forward_rates * np.diff(self._knot_tau)))
        )
        self._forward_integral = self._build_forward_integral()
        if self._forward_integral is not None:
            self._last_forward = float(self._forward_integral(self._knot_tau[-1], 1))
        if self.knot_dates[0] == self.curve_date:
            self._log_df_effective = 0.0
        elif self.spot_rate is not None:
            self._log_df_effective = math.log(
                float(
                    self.spot_rate.get_discount_factor(
                        self.curve_date, self.knot_dates[0]
                    )
                )
            )
        else:
            self._log_df_effective = -float(self.forward_rates[0]) * float(
                self._knot_tau[0]
            )
        if self.precalculate:
            self.precalculate_dfs()

    def _build_forward_integral(self) -> BSpline | CubicSpline | None:
        """Integral of the forward from the first knot, as a scipy spline over the knots.

        None when the forward is constant per segment, which ``_cum`` already covers.
        """
        if (
            self.interpolation_method is InterpolationMethod.PiecewiseConstant
            or len(self.forward_rates) == 1
        ):
            return None
        if self.interpolation_method is InterpolationMethod.CubicSpline:
            return CubicSpline(self._knot_tau, self._cum, bc_type="natural")
        at_knots = np.concatenate((self.forward_rates[:1], self.forward_rates))
        return make_interp_spline(self._knot_tau, at_knots, k=1).antiderivative()

    @classmethod
    def from_zero_coupon_curve(cls, zcc, index: "Index") -> IndexCurve:
        """Piecewise-constant forwards reproducing a log-linear ZeroCouponCurve.

        Knots at the curve date and at each pillar; equal to the legacy curve up to
        its last pillar when it interpolates log-linearly (exactly the same shape).
        """
        knots = [zcc.curve_date, *zcc.dates]
        log_dfs = np.concatenate(([0.0], np.log(np.asarray(zcc.dfs, dtype=float))))
        tau = year_fractions(zcc.curve_date, knots)
        return cls(zcc.curve_date, index, knots, -np.diff(log_dfs) / np.diff(tau))

    # ── Curve ────────────────────────────────────────────────────────────────

    def get_id(self) -> str:
        return self.index.name

    @property
    def effective_date(self) -> date:
        """First knot: where the forwards start."""
        return self.knot_dates[0]

    @property
    def last_date(self) -> date:
        return self.knot_dates[-1]

    def _dfs_from_days(self, days: np.ndarray) -> np.ndarray:
        tau = days / DAY_BASE
        dfs = np.exp(self._log_df_effective - self._integrate_forward(tau))
        if self.spot_rate is not None and self.knot_dates[0] > self.curve_date:
            spot = tau < self._knot_tau[0]
            if spot.any():
                dfs[spot] = self.spot_rate.get_discount_factor(
                    self.curve_date,
                    [self.curve_date + timedelta(days=int(n)) for n in days[spot]],
                )
        return dfs

    def _integrate_forward(self, tau: np.ndarray) -> np.ndarray:
        """Integral of the forward from the first knot to each of `tau`, flat outside."""
        if self._forward_integral is None:
            # Segment of each date, searched among the inner knots only: before the
            # first one is the first forward, past the last one the last forward.
            idx = np.searchsorted(self._knot_tau[1:-1], tau, side="right")
            return self._cum[idx] + self.forward_rates[idx] * (
                tau - self._knot_tau[idx]
            )
        first, last = self._knot_tau[0], self._knot_tau[-1]
        return (
            self._forward_integral(np.clip(tau, first, last))
            + self.forward_rates[0] * np.minimum(tau - first, 0.0)
            + self._last_forward * np.maximum(tau - last, 0.0)
        )

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
        """The curve from `date`: knots are `date` and the knots after it.

        The spot rate is dropped: the first knot is the new curve date. If `date`
        falls before the effective date, the first forward is the average one over
        ``[date, effective_date]``.

        * ``PiecewiseConstant``: exact; a segment cut by `date` keeps its forward.
        * ``CubicSpline``: the dfs at the knots are exact (average forwards), but the
          natural spline is refitted on the new knots, so it differs between them.
        * ``Linear``: each remaining knot keeps its forward, so later segments are
          exact. The first one becomes flat at its end forward, since the forward at
          `date` cannot be stored.
        """
        self._check_aging_date(date)
        knots = [date, *(k for k in self.knot_dates if k > date)]
        if self.interpolation_method is InterpolationMethod.Linear:
            first = len(self.knot_dates) - (len(knots) - 1)
            forwards = np.array(
                [
                    self.forward_rates[max(i - 1, 0)]
                    for i in range(first, len(self.knot_dates))
                ]
            )
        else:
            log_dfs = np.log(self.get_dfs(knots))
            forwards = -np.diff(log_dfs) / np.diff(year_fractions(date, knots))
        return IndexCurve(
            date,
            self.index,
            knots,
            forwards,
            None,
            self.interpolation_method,
            precalculate=self.precalculate,
        )

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
