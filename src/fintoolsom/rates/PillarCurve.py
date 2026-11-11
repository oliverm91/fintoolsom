"""Curves parametrised by forward rates between dates, with an interpolation."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Self

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

    Before the first knot the first forward is extended, and past the last knot the
    last forward is extended flat.

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

    Subclasses give the role of the curve. They declare ``knot_dates``,
    ``forward_rates`` and ``interpolation_method`` as fields after their own identity
    fields, because dataclass fields of a base class come first in the constructor.

    Attributes
    ----------
    curve_date : date
        Curve date (df = 1 there).
    precalculate : bool
        Keyword only. Store the df of every day up to the last knot on creation.
    knot_dates : list of date
        At least 3, strictly increasing, the first one on or after `curve_date`.
    forward_rates : np.ndarray
        One forward per segment: ``len(knot_dates) - 1`` values.
    interpolation_method : InterpolationMethod
        Must be one of the class's ``allowed_methods``.
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
    precalculate: bool = field(default=True, kw_only=True, repr=False, compare=False)

    # For type checkers only: at runtime these would become dataclass fields here.
    if TYPE_CHECKING:
        knot_dates: list[date]
        forward_rates: np.ndarray
        interpolation_method: InterpolationMethod
        _knot_tau: np.ndarray
        _cum: np.ndarray
        _log_df_effective: float
        _forward_integral: Callable[[np.ndarray], np.ndarray]
        _fit: Callable[[np.ndarray], np.ndarray]
        _last_forward: float

    def __post_init__(self):
        self._check_method()
        self.knot_dates = list(self.knot_dates)
        self.forward_rates = np.asarray(self.forward_rates, dtype=float)
        self._check_knots()
        self._knot_tau = year_fractions(self.curve_date, self.knot_dates)
        self._cum = np.concatenate(
            ([0.0], np.cumsum(self.forward_rates * np.diff(self._knot_tau)))
        )
        self._log_df_effective = self._log_df_first_knot()
        self._forward_integral = self._build_forward_integral()
        if self.precalculate:
            self.precalculate_dfs()

    @property
    def last_date(self) -> date:
        return self.knot_dates[-1]

    def get_aged_curve(self, date: date) -> Self:
        """The curve from `date`: knots are `date` and the knots after it.

        The forwards are the average ones between those knots, so the dfs at them are
        ``df(knot) / df(date)`` whatever the interpolation. Between knots
        ``PiecewiseConstant`` is exact too; the other methods are refitted on the
        remaining knots, so they differ there. `date` must be before the second-to-last
        knot, so that the aged curve keeps two segments.
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
        """``log(df)`` at the first knot: the first forward, extended back."""
        return -float(self.forward_rates[0]) * float(self._knot_tau[0])

    def _build_forward_integral(self) -> Callable[[np.ndarray], np.ndarray]:
        """Integral of the forward from the first knot, as a function of any `tau`.

        Before the first knot the first forward is extended, and past the last knot
        the last forward.
        """
        method = self.interpolation_method
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
        self._last_forward = float(self.forward_rates[-1])
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
        return np.exp(self._log_df_effective - self._forward_integral(days / DAY_BASE))
