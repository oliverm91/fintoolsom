"""Discount curve of a currency under a risk-free index, stored as discount factors."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, ClassVar, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from .Curve import Curve, InterpolationMethod, year_fractions

if TYPE_CHECKING:
    from ..market.currencies import Currency
    from ..market.index import Index


@dataclass
class DiscountCurve(Curve):
    """Discount curve of `currency` seen from `risk_free_index` (e.g. USD under ICP).

    Only for a currency other than the risk-free index's: that one discounts on the
    index's own :class:`IndexCurve`. Built from discount factors at pillars; an
    implicit pillar ``df = 1`` sits at the curve date. Past the last pillar the last
    segment's forward is extended flat.

    Parameters
    ----------
    curve_date : date
        Curve date (df = 1 there).
    currency : Currency
        Currency discounted.
    risk_free_index : Index
        Numeraire the curve is built under; with `currency` it gives the curve id
        ``"<currency>_<index>"``.
    pillar_dates : list of date
        Pillars after `curve_date` (sorted on construction).
    dfs : array-like
        Discount factors at `pillar_dates`.
    interpolation_method : InterpolationMethod, optional
        Discount-factor methods only; ``LogLinear`` by default.
    """

    allowed_methods: ClassVar[frozenset[InterpolationMethod]] = frozenset(
        {InterpolationMethod.LogLinear, InterpolationMethod.HermiteCubicSpline}
    )

    curve_date: date
    currency: "Currency"
    risk_free_index: "Index"
    pillar_dates: list[date]
    dfs: np.ndarray
    interpolation_method: InterpolationMethod = InterpolationMethod.LogLinear

    _tau: np.ndarray = field(init=False, repr=False)
    _log_dfs: np.ndarray = field(init=False, repr=False)
    _last_forward: float = field(init=False, repr=False)
    _pchip: PchipInterpolator | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._check_method()
        if self.currency == getattr(self.risk_free_index, "currency", None):
            raise ValueError(
                f"{self.currency} is the currency of the risk-free index '{self.risk_free_index.name}': "
                "it discounts on that index's IndexCurve, not on a DiscountCurve."
            )
        pairs = sorted(zip(self.pillar_dates, np.asarray(self.dfs, dtype=float)), key=lambda p: p[0])
        if not pairs:
            raise ValueError("A DiscountCurve needs at least one pillar.")
        if any(b == a for (a, _), (b, _) in zip(pairs, pairs[1:])):
            raise ValueError(f"Duplicate pillar dates: {[d for d, _ in pairs]}.")
        if pairs[0][0] <= self.curve_date:
            raise ValueError(f"Pillars must be after the curve date {self.curve_date}; got {pairs[0][0]}.")
        if any(df <= 0 for _, df in pairs):
            raise ValueError("Discount factors must be positive.")
        self.pillar_dates = [d for d, _ in pairs]
        self.dfs = np.array([df for _, df in pairs])
        self._tau = np.concatenate(([0.0], year_fractions(self.curve_date, self.pillar_dates)))
        self._log_dfs = np.concatenate(([0.0], np.log(self.dfs)))
        self._last_forward = -(self._log_dfs[-1] - self._log_dfs[-2]) / (self._tau[-1] - self._tau[-2])
        if self.interpolation_method is InterpolationMethod.HermiteCubicSpline and len(self._tau) > 2:
            self._pchip = PchipInterpolator(self._tau, np.exp(self._log_dfs))

    @classmethod
    def from_zero_coupon_curve(cls, zcc, currency: "Currency", risk_free_index: "Index") -> DiscountCurve:
        """The same pillars and dfs as a legacy ZeroCouponCurve, with its method."""
        return cls(
            zcc.curve_date, currency, risk_free_index, list(zcc.dates), np.array(zcc.dfs, dtype=float),
            zcc.df_interpolation_method,
        )

    # ── Curve ────────────────────────────────────────────────────────────────

    def get_id(self) -> str:
        return f"{self.currency.value}_{self.risk_free_index.name}"

    @property
    def last_date(self) -> date:
        return self.pillar_dates[-1]

    @property
    def pillars(self) -> list[tuple[date, float]]:
        """``(date, df)`` pillars, sorted."""
        return list(zip(self.pillar_dates, (float(df) for df in self.dfs)))

    def get_dfs(self, dates: Sequence[date]) -> np.ndarray:
        tau = year_fractions(self.curve_date, list(dates))
        last = self._tau[-1]
        inside = tau <= last
        out = np.empty(len(tau))
        if self._pchip is not None:
            out[inside] = self._pchip(tau[inside])
        else:
            out[inside] = np.exp(np.interp(tau[inside], self._tau, self._log_dfs))
        out[~inside] = np.exp(self._log_dfs[-1] - self._last_forward * (tau[~inside] - last))
        return out

    def parallel_bump_bps(self, bps: float) -> DiscountCurve:
        shift = bps / 10_000
        return DiscountCurve(
            self.curve_date, self.currency, self.risk_free_index, list(self.pillar_dates),
            self.dfs * np.exp(-shift * self._tau[1:]), self.interpolation_method,
        )

    def copy(self) -> DiscountCurve:
        return DiscountCurve(
            self.curve_date, self.currency, self.risk_free_index, list(self.pillar_dates),
            self.dfs.copy(), self.interpolation_method,
        )

    def __str__(self) -> str:
        points = ", ".join(f"({d}|{df:.8f})" for d, df in self.pillars)
        return f"DiscountCurve {self.get_id()} ({self.interpolation_method.name}) date {self.curve_date} [{points}]"
