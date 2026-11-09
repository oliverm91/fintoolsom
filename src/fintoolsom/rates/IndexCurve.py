"""Curve of an interest index, stored as forward rates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, ClassVar, Sequence

import numpy as np

from .Curve import Curve, DateArg, InterpolationMethod, year_fractions
from .Rates import Rate

if TYPE_CHECKING:
    from ..market.index import Index


@dataclass
class IndexCurve(Curve):
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
        One forward per segment: ``len(knot_dates) - 1`` values.
    spot_rate : Rate, optional
        Rate from `curve_date` to the effective date (e.g. today's overnight fixing).
    interpolation_method : InterpolationMethod, optional
        Forward-space methods only; ``PiecewiseConstant`` by default.
    """

    allowed_methods: ClassVar[frozenset[InterpolationMethod]] = frozenset(
        {InterpolationMethod.PiecewiseConstant}
    )

    curve_date: date
    index: "Index"
    knot_dates: list[date]
    forward_rates: np.ndarray
    spot_rate: Rate | None = None
    interpolation_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant

    _knot_tau: np.ndarray = field(init=False, repr=False)
    _cum: np.ndarray = field(init=False, repr=False)
    _log_df_effective: float = field(init=False, repr=False)

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

    def get_dfs(self, dates: Sequence[date]) -> np.ndarray:
        dates = list(dates)
        tau = year_fractions(self.curve_date, dates)
        # Segment of each date: the first forward before the effective date, the last
        # one past the last knot.
        idx = np.clip(
            np.searchsorted(self._knot_tau, tau, side="right") - 1,
            0,
            len(self.forward_rates) - 1,
        )
        integral = self._cum[idx] + self.forward_rates[idx] * (
            tau - self._knot_tau[idx]
        )
        dfs = np.exp(self._log_df_effective - integral)
        if self.spot_rate is not None and self.knot_dates[0] > self.curve_date:
            spot = tau < self._knot_tau[0]
            if spot.any():
                dfs[spot] = self.spot_rate.get_discount_factor(
                    self.curve_date, [d for d, s in zip(dates, spot) if s]
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
        )

    def copy(self) -> IndexCurve:
        return IndexCurve(
            self.curve_date,
            self.index,
            list(self.knot_dates),
            self.forward_rates.copy(),
            None if self.spot_rate is None else self.spot_rate.copy(),
            self.interpolation_method,
        )

    # ── Projection ───────────────────────────────────────────────────────────

    def get_accrual(self, start: DateArg, end: DateArg) -> float | np.ndarray:
        """Interest per unit notional over ``[start, end]``: wealth factor - 1."""
        return self.get_wealth_factor(start, end) - 1

    def __str__(self) -> str:
        points = ", ".join(
            f"({d}|{f:.6f})" for d, f in zip(self.knot_dates[1:], self.forward_rates)
        )
        return (
            f"IndexCurve {self.index.name} ({self.interpolation_method.name}) date {self.curve_date} "
            f"effective {self.effective_date} [{points}]"
        )
