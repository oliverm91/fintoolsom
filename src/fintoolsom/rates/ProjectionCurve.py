from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Sequence, cast

import numpy as np

from .Rates import Rate, RateConvention

if TYPE_CHECKING:
    from .ZeroCouponCurve import ZeroCouponCurve


class ProjectionInterpolationMethod(Enum):
    """How a :class:`ProjectionCurve` shapes its forward-rate function ``f(u)``
    between knots (and hence how it solves the accrual integral ``∫ f``).

    Only ``PiecewiseConstant`` is implemented. Later, forward-space methods
    (piecewise-linear, cubic, monotone-convex) plug in behind the same
    ``get_accrual`` seam and only change the integrator (§4 of the design)."""

    PiecewiseConstant = 1


# A ``ProjectionCurve`` (or its aliased discount view) answers two queries between
# any two dates; ``start``/``end`` accept a single date or an aligned sequence.
DateArg = date | Sequence[date]


@dataclass(slots=True)
class ProjectionCurve:
    """Forward-rate curve used to *project* a floating index's forwards (as opposed
    to *discount* a cashflow — that stays on :class:`ZeroCouponCurve`).

    The curve's **state is forward rates**, one per segment ``[knotᵢ, knotᵢ₊₁]``
    (continuously compounded, act/``time_fraction_base``). Pseudo-DFs are a *derived*
    output (``pseudo_df(u) = exp(−∫ f)``) exposed only for DF-consumers. The public
    seam is:

    - ``get_accrual(s, e)``            — the interest earned per unit notional over
                                          ``[s, e]`` (``WF − 1``, so a leg multiplies
                                          it by its residual notional directly).
    - ``get_equivalent_forward_rate``  — the single rate equivalent to that accrual.
    - ``pseudo_df(u)``                 — derived cumulative discount factor.

    Knots are one-per-instrument-maturity (chosen by the bootstrap), so the solve is
    square. ``knot_dates`` are strictly increasing; ``knot_dates[0]`` is the curve's
    domain start (the short-end anchor, e.g. ``t + spot_lag``) and ``pseudo_df`` is
    normalised to 1 there. There are ``len(knot_dates) - 1`` segment forward rates."""

    curve_date: date
    knot_dates: list[date]
    forward_rates: np.ndarray
    interpolation_method: ProjectionInterpolationMethod = (
        ProjectionInterpolationMethod.PiecewiseConstant
    )
    time_fraction_base: int = 365

    # Cached integration scaffolding (days since curve_date, cumulative ∫ at knots).
    _knot_days: np.ndarray = field(init=False, default=None, repr=False)
    _cum_integral: np.ndarray = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.forward_rates = np.asarray(self.forward_rates, dtype=float)
        if len(self.knot_dates) < 2:
            raise ValueError(
                f"A ProjectionCurve needs at least 2 knots (one segment); got "
                f"{len(self.knot_dates)}."
            )
        if len(self.forward_rates) != len(self.knot_dates) - 1:
            raise ValueError(
                f"forward_rates must have one entry per segment: expected "
                f"{len(self.knot_dates) - 1} (= knots - 1), got {len(self.forward_rates)}."
            )
        if any(b <= a for a, b in zip(self.knot_dates, self.knot_dates[1:])):
            raise ValueError(f"knot_dates must be strictly increasing; got {self.knot_dates}.")
        if self.interpolation_method is not ProjectionInterpolationMethod.PiecewiseConstant:
            raise NotImplementedError(
                f"ProjectionCurve interpolation '{self.interpolation_method}' is not "
                "implemented yet; only PiecewiseConstant is available."
            )

        self._knot_days = np.array(
            [(d - self.curve_date).days for d in self.knot_dates], dtype=float
        )
        # cum[j] = ∫ from knot[0] to knot[j] = Σ_{i<j} fᵢ · Δtᵢ  (Δt in act/base years).
        seg_years = np.diff(self._knot_days) / self.time_fraction_base
        self._cum_integral = np.concatenate(
            ([0.0], np.cumsum(self.forward_rates * seg_years))
        )

    # ── integral of f from the domain start (knot[0]) to each date ──────────────

    def _integral_from_start(self, dates: Sequence[date]) -> np.ndarray:
        days = np.array([(d - self.curve_date).days for d in dates], dtype=float)
        # Segment index for each date; piecewise-constant flat-extrapolates the first
        # forward before knot[0] and the last forward beyond knot[-1].
        n_seg = len(self.forward_rates)
        idx = np.clip(np.searchsorted(self._knot_days, days, side="right") - 1, 0, n_seg - 1)
        return (
            self._cum_integral[idx]
            + self.forward_rates[idx] * (days - self._knot_days[idx]) / self.time_fraction_base
        )

    def _pseudo_dfs(self, dates: Sequence[date]) -> np.ndarray:
        return np.exp(-self._integral_from_start(dates))

    # ── public seam ─────────────────────────────────────────────────────────────

    def pseudo_df(self, t: date) -> float:
        """Derived cumulative discount factor ``exp(−∫_{start}^{t} f)`` (normalised
        to 1 at the domain start). Provided for DF-consumers; the curve's state is
        forward rates, not this."""
        return float(self._pseudo_dfs([t])[0])

    def pseudo_dfs(self, dates: Sequence[date]) -> np.ndarray:
        return self._pseudo_dfs(dates)

    def get_wealth_factor(self, start: DateArg, end: DateArg) -> float | np.ndarray:
        """Compounded factor the index earns over ``[start, end]`` = ``WF(s, e) =
        exp(∫ₛᵉ f) = pseudo_df(s) / pseudo_df(e)`` (≥ 1 for positive forwards)."""
        if isinstance(start, date) and isinstance(end, date):
            return float(self._pseudo_dfs([start])[0] / self._pseudo_dfs([end])[0])
        starts = cast("Sequence[date]", start)
        ends = cast("Sequence[date]", end)
        return self._pseudo_dfs(list(starts)) / self._pseudo_dfs(list(ends))

    def get_accrual(self, start: DateArg, end: DateArg) -> float | np.ndarray:
        """Interest earned per unit notional over ``[start, end]`` = ``WF − 1``.
        A leg multiplies this by its residual notional to get the coupon interest."""
        wf = self.get_wealth_factor(start, end)
        return wf - 1

    def get_equivalent_forward_rate(
        self, start: date, end: date, rate_convention: RateConvention = None
    ) -> Rate:
        """The single forward rate, under ``rate_convention``, equivalent to the
        curve's accrual over ``[start, end]``."""
        rc = rate_convention if rate_convention is not None else RateConvention()
        wf = self.get_wealth_factor(start, end)
        yf = rc.day_count_convention.get_time_fraction(start, end, rc.time_fraction_base)
        return Rate(rc, float(rc.interest_convention.get_rate_from_wf(wf, yf)))

    def __len__(self) -> int:
        return len(self.forward_rates)

    def __str__(self) -> str:
        pts = ", ".join(
            f"({self.knot_dates[i + 1]}|{self.forward_rates[i]:.6f})"
            for i in range(len(self.forward_rates))
        )
        return (
            f"ProjectionCurve({self.interpolation_method.name}) date {self.curve_date} "
            f"anchor {self.knot_dates[0]} [{pts}]"
        )


@dataclass(slots=True)
class DiscountProjectionView:
    """Thin forward *view* over a discount :class:`ZeroCouponCurve`, used when an
    overnight index is self-discounted (OIS) so its projection and discount curves
    are the **same** unknown (§7 of the design). It exposes the exact same seam as
    :class:`ProjectionCurve` (``get_accrual`` / ``get_wealth_factor`` / ``pseudo_df``)
    but reads the wealth/discount factors straight off the underlying curve — no
    separate pillars — so callers never branch on aliased vs independent."""

    discount_curve: "ZeroCouponCurve"

    def pseudo_df(self, t: date) -> float:
        return self.discount_curve.get_df(t)

    def pseudo_dfs(self, dates: Sequence[date]) -> np.ndarray:
        return self.discount_curve.get_dfs(list(dates))

    def get_wealth_factor(self, start: DateArg, end: DateArg) -> float | np.ndarray:
        if isinstance(start, date) and isinstance(end, date):
            return self.discount_curve.get_wf_fwd(start, end)
        starts = cast("Sequence[date]", start)
        ends = cast("Sequence[date]", end)
        return self.discount_curve.get_wfs_fwds(list(starts), list(ends))

    def get_accrual(self, start: DateArg, end: DateArg) -> float | np.ndarray:
        return self.get_wealth_factor(start, end) - 1

    def get_equivalent_forward_rate(
        self, start: date, end: date, rate_convention: RateConvention = None
    ) -> Rate:
        rc = rate_convention if rate_convention is not None else RateConvention()
        wf = self.get_wealth_factor(start, end)
        yf = rc.day_count_convention.get_time_fraction(start, end, rc.time_fraction_base)
        return Rate(rc, float(rc.interest_convention.get_rate_from_wf(wf, yf)))
