"""Discount curve of a currency under a risk-free index, stored as forward rates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from .Curve import InterpolationMethod
from .PillarCurve import PillarCurve

if TYPE_CHECKING:
    from ..market.currencies import Currency
    from ..market.index import Index


@dataclass
class DiscountCurve(PillarCurve):
    """Discount curve of `currency` seen from `risk_free_index` (e.g. USD under ICP).

    Only for a currency other than the risk-free index's: that one discounts on the
    index's own :class:`IndexCurve`. Its parameters are forward rates from the curve
    date, as in any :class:`PillarCurve`. Past the last knot the last forward is
    extended flat.

    Parameters
    ----------
    curve_date : date
        Curve date (df = 1 there).
    currency : Currency
        Currency discounted.
    risk_free_index : Index
        Numeraire the curve is built under; with `currency` it gives the curve id
        ``"<currency>_<index>"``.
    knot_dates : list of date
        Strictly increasing, the first one equal to `curve_date`.
    forward_rates : array-like
        One forward per segment: ``len(knot_dates) - 1`` values.
    interpolation_method : InterpolationMethod, optional
        ``PiecewiseConstant`` by default. ``CubicSpline`` is a natural spline on
        ``log(df)``, so the forward joins the flat extrapolation smoothly at the last
        knot.
    precalculate : bool, optional
        Keyword only. Store the df of every day up to the last knot on creation (the
        default). Creation is slower and lookups faster; False skips it for a curve
        that is created many times and read little. :meth:`precalculate_dfs` builds
        the grid later.
    """

    currency: "Currency"
    risk_free_index: "Index"
    knot_dates: list[date]
    forward_rates: np.ndarray
    interpolation_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant

    def __post_init__(self):
        super().__post_init__()
        if self.currency == getattr(self.risk_free_index, "currency", None):
            raise ValueError(
                f"{self.currency} is the currency of the risk-free index '{self.risk_free_index.name}': "
                "it discounts on that index's IndexCurve, not on a DiscountCurve."
            )
        if self.knot_dates[0] != self.curve_date:
            raise ValueError(
                f"The first knot {self.knot_dates[0]} must be the curve date {self.curve_date}."
            )

    # ── Curve ────────────────────────────────────────────────────────────────

    def get_id(self) -> str:
        return f"{self.currency.value}_{self.risk_free_index.name}"

    def parallel_bump_bps(self, bps: float) -> DiscountCurve:
        return DiscountCurve(
            self.curve_date,
            self.currency,
            self.risk_free_index,
            list(self.knot_dates),
            self.forward_rates + bps / 10_000,
            self.interpolation_method,
            precalculate=self.precalculate,
        )

    def copy(self) -> DiscountCurve:
        return DiscountCurve(
            self.curve_date,
            self.currency,
            self.risk_free_index,
            list(self.knot_dates),
            self.forward_rates.copy(),
            self.interpolation_method,
            precalculate=self.precalculate,
        )

    def __str__(self) -> str:
        points = ", ".join(
            f"({d}|{f:.6f})" for d, f in zip(self.knot_dates[1:], self.forward_rates)
        )
        return f"DiscountCurve {self.get_id()} ({self.interpolation_method.name}) date {self.curve_date} [{points}]"
