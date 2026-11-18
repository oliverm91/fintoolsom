"""Curves parametrised by forward rates between dates, with an interpolation."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from functools import cache
from typing import ClassVar, Self

import numpy as np
from scipy.interpolate import (
    BSpline,
    CubicSpline,
    PchipInterpolator,
    PPoly,
    make_interp_spline,
)

from ._convex_monotone import convex_monotone_forward
from .Curve import DAY_BASE, DailyGridCurve, InterpolationMethod, year_fractions
from .Rates import Rate

# Every pillar curve has at least two segments.
MIN_KNOTS = 3


@cache
def _warn_piecewise_linear() -> None:
    """Warn about ``PiecewiseLinear``, only the first time a curve uses it."""
    warnings.warn(
        "InterpolationMethod.PiecewiseLinear is unstable: each knot forward is "
        "2 * segment average - previous knot forward, so forwards zig-zag and a "
        "change in one segment carries to every later knot without decaying. "
        "Hagan & West (2006), 'Interpolation Methods for Curve Construction', "
        "Applied Mathematical Finance 13(2), p. 97, reject the method for this "
        "reason. PiecewiseConstant is the stable alternative.",
        UserWarning,
        stacklevel=2,
    )


@dataclass
class PillarCurve(DailyGridCurve):
    """Curve whose parameters are one forward rate per segment between knots.

    The state is ``forward_rates[i]``, the average forward, continuously compounded
    (Act/365), over the segment ``[knot_dates[i], knot_dates[i + 1]]``. Discount
    factors are only ever a result of it::

        df(t) = df(knot_dates[0]) * exp(-∫ f)

    ``df(knot_dates[0])`` comes from `spot_rate`, in its own convention, when the
    first knot is after the curve date (e.g. today's overnight fixing up to the next
    business day, or a term index's spot lag); otherwise the first forward is
    extended back to the curve date. Past the last knot the last forward is extended
    flat.

    A curve has no role of its own: the :class:`~fintoolsom.market.Market` key it is
    stored under (an ``Index`` or a ``Currency``) says what it projects or discounts.
    :meth:`from_dfs` and :meth:`from_zero_rates` build one from dfs or zero rates at
    pillar dates.

    The dfs at the knots therefore do not depend on the interpolation, which only
    chooses the function integrated between knots:

    * ``PiecewiseConstant``: constant over each segment.
    * ``PiecewiseLinear``: continuous and linear between knots, flat over the first
      segment. Forwards zig-zag, so the first curve created with it warns; see Hagan
      & West (2006), "Interpolation Methods for Curve Construction", p. 97.
    * ``CubicSpline``: from a natural spline on ``log(df)`` through the knots.
    * ``HermiteCubicSpline``: from a monotone cubic on the dfs at the knots.
    * ``ConvexMonotone``: continuous and quadratic by pieces, from the monotone convex
      method of Patrick S. Hagan and Graeme West (2006) with Simon Ibbotson's QuantLib
      enhancements (quadraticity 0.3, monotonicity 0.7).

    Parameters
    ----------
    curve_date : date
        Curve date (df = 1 there).
    knot_dates : list of date
        At least 3, strictly increasing, the first one on or after `curve_date`.
    forward_rates : array-like
        One forward per segment: ``len(knot_dates) - 1`` values.
    interpolation_method : InterpolationMethod
        How the forward varies between knots; one of ``allowed_methods``.
    spot_rate : Rate, optional
        Rate from `curve_date` to the first knot. Ignored when the first knot is the
        curve date.
    precalculate : bool, optional
        Keyword only. Store the df of every day up to the last knot on creation (the
        default). Creation is slower and lookups faster; False skips it for a curve
        that is created many times and read little. :meth:`precalculate_dfs` builds
        the grid later.
    """

    allowed_methods: ClassVar[frozenset[InterpolationMethod]] = frozenset(
        {
            InterpolationMethod.PiecewiseConstant,
            InterpolationMethod.PiecewiseLinear,
            InterpolationMethod.HermiteCubicSpline,
            InterpolationMethod.CubicSpline,
            InterpolationMethod.ConvexMonotone,
        }
    )

    curve_date: date
    knot_dates: list[date]
    forward_rates: np.ndarray
    interpolation_method: InterpolationMethod
    spot_rate: Rate | None = None
    precalculate: bool = field(default=True, kw_only=True, repr=False, compare=False)

    # Derived state is annotated where it is assigned, not in the class body: an
    # annotation there would be a dataclass field, i.e. a constructor parameter.
    def __post_init__(self):
        self._check_method()
        self.knot_dates = list(self.knot_dates)
        self.forward_rates = np.asarray(self.forward_rates, dtype=float)
        self._check_knots()
        self._knot_tau: np.ndarray = year_fractions(self.curve_date, self.knot_dates)
        self._cum: np.ndarray = np.concatenate(
            ([0.0], np.cumsum(self.forward_rates * np.diff(self._knot_tau)))
        )
        self._log_df_effective: float = self._log_df_first_knot()
        self._forward_integral: Callable[[np.ndarray], np.ndarray] = (
            self._build_forward_integral()
        )
        if self.precalculate:
            self.precalculate_dfs()

    def _set_forward_rates(self, forward_rates) -> None:
        """Replace the forwards in place, the curve then being the one
        ``__post_init__`` would build on them (knots, method and spot kept).

        For the curve builder's solver, which updates its curves on every iteration:
        it skips the construction and the knot checks. The daily grid is recomputed if
        the curve keeps one. Only for curves nothing else holds: a shared curve would
        change for all.
        """
        forward_rates = np.asarray(forward_rates, dtype=float)
        if forward_rates.shape != self.forward_rates.shape:
            raise ValueError(
                f"forward_rates must have one entry per segment: expected "
                f"{len(self.forward_rates)}, got {len(forward_rates)}."
            )
        self.forward_rates = forward_rates
        self._cum = np.concatenate(
            ([0.0], np.cumsum(forward_rates * np.diff(self._knot_tau)))
        )
        self._log_df_effective = self._log_df_first_knot()
        self._forward_integral = self._build_forward_integral()
        self._grid, self._grid_horizon = None, -1
        if self.precalculate:
            self.precalculate_dfs()

    # ── Construction from pillars ────────────────────────────────────────────

    @classmethod
    def from_dfs(
        cls,
        curve_date: date,
        dates: Sequence[date],
        dfs: Sequence[float],
        interpolation_method: InterpolationMethod,
        *,
        precalculate: bool = True,
    ) -> Self:
        """The curve whose dfs at `dates` (after `curve_date`, ascending) are `dfs`.

        The knots are `curve_date` and `dates`, so at least 2 dates are needed. With
        ``PiecewiseConstant`` the dfs between pillars are log-linear.
        """
        if len(dates) != len(dfs):
            raise ValueError(
                f"dates and dfs differ in length: {len(dates)} != {len(dfs)}."
            )
        knots = [curve_date, *dates]
        log_dfs = np.log(np.concatenate(([1.0], np.asarray(dfs, dtype=float))))
        forwards = -np.diff(log_dfs) / np.diff(year_fractions(curve_date, knots))
        return cls(
            curve_date, knots, forwards, interpolation_method, precalculate=precalculate
        )

    @classmethod
    def from_zero_rates(
        cls,
        curve_date: date,
        dates: Sequence[date],
        rates: Sequence[Rate],
        interpolation_method: InterpolationMethod,
        *,
        precalculate: bool = True,
    ) -> Self:
        """The curve whose zero rates from `curve_date` to `dates` are `rates`.

        Each rate is read in its own convention, so they need not share one.
        """
        if len(dates) != len(rates):
            raise ValueError(
                f"dates and rates differ in length: {len(dates)} != {len(rates)}."
            )
        dfs = [
            float(r.get_discount_factor(curve_date, d)) for r, d in zip(rates, dates)
        ]
        return cls.from_dfs(
            curve_date, dates, dfs, interpolation_method, precalculate=precalculate
        )

    # ── Curve ────────────────────────────────────────────────────────────────

    @property
    def last_date(self) -> date:
        return self.knot_dates[-1]

    @property
    def effective_date(self) -> date:
        """First knot: where the forwards start."""
        return self.knot_dates[0]

    def parallel_bump_bps(self, bps: float) -> Self:
        """Copy with every forward, and the spot segment, shifted by `bps` (continuous).

        The spot rate keeps its convention: its value moves so that the df to the
        first knot shifts exactly like a continuous zero rate.
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
        return replace(
            self,
            knot_dates=list(self.knot_dates),
            forward_rates=self.forward_rates + shift,
            spot_rate=spot,
        )

    def copy(self) -> Self:
        return replace(
            self,
            knot_dates=list(self.knot_dates),
            forward_rates=self.forward_rates.copy(),
            spot_rate=None if self.spot_rate is None else self.spot_rate.copy(),
        )

    def get_aged_curve(self, date: date) -> Self:
        """The curve from `date`: knots are `date` and the knots after it.

        The forwards are the average ones between those knots, so the dfs at them are
        ``df(knot) / df(date)`` whatever the interpolation. Between knots
        ``PiecewiseConstant`` is exact too; the other methods are refitted on the
        remaining knots, so they differ there. `date` must be before the second-to-last
        knot, so that the aged curve keeps two segments. The aged curve has no spot
        rate: its first knot is its curve date.
        """
        self._check_aging_date(date)
        if date >= self.knot_dates[-2]:
            raise ValueError(
                f"Date {date} must be before the second-to-last knot "
                f"{self.knot_dates[-2]}: an aged curve needs at least 2 segments."
            )
        knots = [date, *(k for k in self.knot_dates if k > date)]
        log_dfs = np.log(self.get_dfs(knots))
        return replace(
            self,
            curve_date=date,
            knot_dates=knots,
            forward_rates=-np.diff(log_dfs) / np.diff(year_fractions(date, knots)),
            spot_rate=None,
        )

    def _check_method(self) -> None:
        if self.interpolation_method not in self.allowed_methods:
            allowed = ", ".join(sorted(m.name for m in self.allowed_methods))
            raise ValueError(
                f"{type(self).__name__} does not support interpolation "
                f"'{getattr(self.interpolation_method, 'name', self.interpolation_method)}'; "
                f"allowed: {allowed}."
            )

    def _check_knots(self) -> None:
        knots, n_segments = self.knot_dates, len(self.knot_dates) - 1
        if len(knots) < MIN_KNOTS:
            raise ValueError(
                f"{type(self).__name__} needs at least 2 segments ({MIN_KNOTS} knots); "
                f"got {len(knots)} knots."
            )
        if len(self.forward_rates) != n_segments:
            raise ValueError(
                f"forward_rates must have one entry per segment: expected {n_segments}, "
                f"got {len(self.forward_rates)}."
            )
        if any(b <= a for a, b in zip(knots, knots[1:])):
            raise ValueError(f"knot_dates must be strictly increasing; got {knots}.")
        if knots[0] < self.curve_date:
            raise ValueError(
                f"The first knot {knots[0]} is before the curve date {self.curve_date}."
            )

    def _log_df_first_knot(self) -> float:
        """``log(df)`` at the first knot: from the spot rate, else the first forward
        extended back."""
        if self.spot_rate is None or self.knot_dates[0] == self.curve_date:
            return -float(self.forward_rates[0]) * float(self._knot_tau[0])
        return math.log(
            float(
                self.spot_rate.get_discount_factor(self.curve_date, self.knot_dates[0])
            )
        )

    def _build_forward_integral(self) -> Callable[[np.ndarray], np.ndarray]:
        """Integral of the forward from the first knot, as a function of any `tau`.

        Before the first knot the first forward is extended, and past the last knot
        the last forward.
        """
        method = self.interpolation_method
        self._fit: Callable[..., np.ndarray]
        if method is InterpolationMethod.PiecewiseConstant:
            return self._piecewise_constant_integral
        elif method is InterpolationMethod.CubicSpline:
            self._fit = CubicSpline(self._knot_tau, self._cum, bc_type="natural")
        elif method is InterpolationMethod.PiecewiseLinear:
            _warn_piecewise_linear()
            at_knots = self._linear_forwards_at_knots()
            self._fit = make_interp_spline(
                self._knot_tau, at_knots, k=1
            ).antiderivative()
        elif method is InterpolationMethod.HermiteCubicSpline:
            pchip = PchipInterpolator(self._knot_tau, np.exp(-self._cum))
            self._fit = lambda tau: -np.log(pchip(tau))
        elif method is InterpolationMethod.ConvexMonotone:
            self._fit = convex_monotone_forward(
                self._knot_tau, self.forward_rates
            ).antiderivative()
        else:
            raise ValueError(f"No forward integral defined for {method!r}.")
        self._last_forward: float = float(self.forward_rates[-1])
        if isinstance(self._fit, (BSpline, PPoly)):
            self._last_forward = float(self._fit(self._knot_tau[-1], 1))
        return self._fit_extended_flat

    def _piecewise_constant_integral(self, tau: np.ndarray) -> np.ndarray:
        # Segment of each date, searched among the inner knots only: before the first
        # one is the first forward, past the last one the last forward.
        idx = np.searchsorted(self._knot_tau[1:-1], tau, side="right")
        return self._cum[idx] + self.forward_rates[idx] * (tau - self._knot_tau[idx])

    def _fit_extended_flat(self, tau: np.ndarray) -> np.ndarray:
        first, last = self._knot_tau[0], self._knot_tau[-1]
        return (
            self._fit(np.clip(tau, first, last))
            + self.forward_rates[0] * np.minimum(tau - first, 0.0)
            + self._last_forward * np.maximum(tau - last, 0.0)
        )

    def _linear_forwards_at_knots(self) -> np.ndarray:
        """Forwards at the knots of the continuous line with the segment averages."""
        at_knots = np.empty(len(self.knot_dates))
        at_knots[:2] = self.forward_rates[0]
        for i, average in enumerate(self.forward_rates[1:], start=1):
            at_knots[i + 1] = 2 * average - at_knots[i]
        return at_knots

    def _dfs_from_days(self, days: np.ndarray) -> np.ndarray:
        dfs = np.exp(self._log_df_effective - self._forward_integral(days / DAY_BASE))
        if self.spot_rate is not None and self.knot_dates[0] > self.curve_date:
            spot = days / DAY_BASE < self._knot_tau[0]
            if spot.any():
                dfs[spot] = self.spot_rate.get_discount_factor(
                    self.curve_date,
                    [self.curve_date + timedelta(days=int(n)) for n in days[spot]],
                )
        return dfs

    def __str__(self) -> str:
        points = ", ".join(
            f"({d}|{f:.6f})" for d, f in zip(self.knot_dates[1:], self.forward_rates)
        )
        start = (
            f" effective {self.effective_date}"
            if self.effective_date > self.curve_date
            else ""
        )
        return (
            f"PillarCurve ({self.interpolation_method.name}) date {self.curve_date}"
            f"{start} [{points}]"
        )
