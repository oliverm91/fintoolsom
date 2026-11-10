"""Curves used for pricing, the market and the curve builder.

:class:`Curve` is the discounting interface every curve implements. Two concrete
roles derive from it:

* :class:`~fintoolsom.rates.IndexCurve` — the curve of an interest index, stored as
  forward rates; it discounts and projects that index.
* :class:`~fintoolsom.rates.DiscountCurve` — the discount curve of a currency that
  has no index of its own under the market's risk-free index (e.g. USD in Chile
  under ICP), stored as discount factors.

Time is measured in Act/365 year fractions from the curve date; forward and zero
rates are continuously compounded on that basis unless a ``RateConvention`` is given.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from enum import Enum
from typing import ClassVar, Self, Sequence

import numpy as np

from .Rates import Rate, RateConvention

# Day base of the curves' time axis (Act/365).
DAY_BASE = 365

DateArg = date | Sequence[date]


class InterpolationMethod(Enum):
    """How a curve fills the gaps between its pillars or knots.

    Each curve class accepts only the methods of its own space (see
    ``allowed_methods``):

    * ``LogLinear`` (discount factors): linear in ``log(df)``, i.e. piecewise-constant
      forwards between pillars;
    * ``HermiteCubicSpline`` (discount factors): monotone cubic (PCHIP) on the dfs;
    * ``PiecewiseConstant`` (forward rates): one constant forward per segment;
    * ``Linear`` (forward rates): forwards linear in time between knots;
    * ``CubicSpline`` (discount and index curves): natural C2 cubic spline on
      ``log(df)``. A pillar moves the whole curve.
    """

    LogLinear = 1
    HermiteCubicSpline = 2
    PiecewiseConstant = 3
    Linear = 4
    CubicSpline = 5


def year_fractions(curve_date: date, dates: Sequence[date]) -> np.ndarray:
    """Act/365 year fractions from `curve_date` to each of `dates`."""
    return np.array([(d - curve_date).days for d in dates], dtype=float) / DAY_BASE


def date_ordinals(dates: Sequence[date]) -> np.ndarray:
    """Proleptic Gregorian ordinals of `dates`, as integers."""
    return np.array([d.toordinal() for d in dates], dtype=np.int64)


class Curve(ABC):
    """A discount curve: maps a date to a discount factor from ``curve_date``.

    Subclasses implement the dfs, their id in the market, their last pillar and a
    bumped copy; everything else (forward dfs, wealth factors, zero and forward
    rates) derives from the dfs here.

    Attributes
    ----------
    curve_date : date
        Date the dfs are measured from (df = 1 there).
    interpolation_method : InterpolationMethod
        Must be one of the class's ``allowed_methods``.
    """

    allowed_methods: ClassVar[frozenset[InterpolationMethod]] = frozenset()
    curve_date: date
    interpolation_method: InterpolationMethod

    def _check_method(self) -> None:
        if self.interpolation_method not in self.allowed_methods:
            allowed = ", ".join(sorted(m.name for m in self.allowed_methods))
            raise ValueError(
                f"{type(self).__name__} does not support interpolation "
                f"'{getattr(self.interpolation_method, 'name', self.interpolation_method)}'; "
                f"allowed: {allowed}."
            )

    # ── Required ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_id(self) -> str:
        """Key of the curve in :attr:`Market.curves`."""

    @abstractmethod
    def get_dfs(self, dates: Sequence[date]) -> np.ndarray:
        """Discount factors from ``curve_date`` to each of `dates`."""

    @property
    @abstractmethod
    def last_date(self) -> date:
        """Last pillar or knot; past it the curve extrapolates (flat forward)."""

    @abstractmethod
    def parallel_bump_bps(self, bps: float) -> Self:
        """Copy with every continuously compounded zero rate shifted by `bps`."""

    @abstractmethod
    def copy(self) -> Self: ...

    def __copy__(self) -> Self:
        return self.copy()

    # ── Derived ──────────────────────────────────────────────────────────────

    def _dfs_at_ordinals(self, ordinals: np.ndarray) -> np.ndarray:
        """:meth:`get_dfs` for dates already held as integer ordinals."""
        return self.get_dfs([date.fromordinal(int(o)) for o in ordinals])

    def get_df(self, d: date) -> float:
        """Discount factor from ``curve_date`` to `d`."""
        return float(self.get_dfs([d])[0])

    def get_df_fwd(self, start: date, end: date) -> float:
        """Forward discount factor ``df(end) / df(start)``."""
        dfs = self.get_dfs([start, end])
        return float(dfs[1] / dfs[0])

    def get_dfs_fwds(self, starts: Sequence[date], ends: Sequence[date]) -> np.ndarray:
        """Forward discount factors ``df(end) / df(start)``, element-wise."""
        if len(starts) != len(ends):
            raise ValueError(
                f"starts and ends differ in length: {len(starts)} != {len(ends)}."
            )
        return self.get_dfs(list(ends)) / self.get_dfs(list(starts))

    def get_wealth_factor(self, start: DateArg, end: DateArg) -> float | np.ndarray:
        """Compounded factor ``df(start) / df(end)``: one date pair, or aligned sequences."""
        if isinstance(start, date) and isinstance(end, date):
            return 1.0 / self.get_df_fwd(start, end)
        return 1.0 / self.get_dfs_fwds(list(start), list(end))  # type: ignore[arg-type]

    def get_zero_rate(
        self, d: date, rate_convention: RateConvention | None = None
    ) -> Rate:
        """Zero rate from ``curve_date`` to `d` (annual compounding, Act/365 by default)."""
        rc = rate_convention if rate_convention is not None else RateConvention()
        yf = rc.day_count_convention.get_time_fraction(
            self.curve_date, d, rc.time_fraction_base
        )
        return Rate(
            rc, float(rc.interest_convention.get_rate_from_df(self.get_df(d), yf))
        )

    def get_forward_rate(
        self, start: date, end: date, rate_convention: RateConvention | None = None
    ) -> Rate:
        """Single rate over ``[start, end]`` equivalent to the curve's wealth factor."""
        rc = rate_convention if rate_convention is not None else RateConvention()
        yf = rc.day_count_convention.get_time_fraction(
            start, end, rc.time_fraction_base
        )
        return Rate(
            rc,
            float(
                rc.interest_convention.get_rate_from_wf(
                    self.get_wealth_factor(start, end), yf
                )
            ),
        )


class DailyGridCurve(Curve):
    """Curve that can store its df for every day up to its last pillar or knot.

    With the grid a df is an array lookup, whatever the interpolation; dates outside
    ``[curve_date, last_date]`` still use the formula. The grid is a snapshot of the
    curve when it was computed: call :meth:`precalculate_dfs` again after changing
    the curve's data in place.

    Subclasses hold a ``precalculate`` flag and implement :meth:`_dfs_from_days`.
    """

    precalculate: bool
    _grid: np.ndarray | None = None
    _grid_horizon: int = -1

    @abstractmethod
    def _dfs_from_days(self, days: np.ndarray) -> np.ndarray:
        """Discount factors from the curve's formula at day offsets from ``curve_date``."""

    def precalculate_dfs(self) -> None:
        """Compute the daily df grid, replacing any existing one.

        For a curve created with ``precalculate=False``. Copies and bumped curves of
        this curve then have a grid too.
        """
        horizon = (self.last_date - self.curve_date).days
        self._grid = self._dfs_from_days(np.arange(horizon + 1, dtype=np.int64))
        self._grid_horizon = horizon
        self.precalculate = True

    def _dfs_at_days(self, days: np.ndarray) -> np.ndarray:
        """Discount factors at int64 day offsets from ``curve_date``."""
        grid = self._grid
        if grid is None:
            return self._dfs_from_days(days)
        # Unsigned, a negative offset is huge: one comparison checks both ends.
        unsigned = days.view(np.uint64)
        if not days.size or unsigned.max() <= self._grid_horizon:
            return grid[days]
        inside = unsigned <= self._grid_horizon
        out = np.empty(len(days))
        out[inside] = grid[days[inside]]
        out[~inside] = self._dfs_from_days(days[~inside])
        return out

    def _dfs_at_ordinals(self, ordinals: np.ndarray) -> np.ndarray:
        return self._dfs_at_days(ordinals - self.curve_date.toordinal())

    def get_dfs(self, dates: Sequence[date]) -> np.ndarray:
        curve_date = self.curve_date
        return self._dfs_at_days(
            np.array([(d - curve_date).days for d in dates], dtype=np.int64)
        )

    def get_df(self, d: date) -> float:
        i = (d - self.curve_date).days
        if self._grid is not None and 0 <= i <= self._grid_horizon:
            return float(self._grid[i])
        return super().get_df(d)

    def get_df_fwd(self, start: date, end: date) -> float:
        grid = self._grid
        if grid is not None:
            i = (start - self.curve_date).days
            j = (end - self.curve_date).days
            if 0 <= i <= self._grid_horizon and 0 <= j <= self._grid_horizon:
                return float(grid[j] / grid[i])
        return super().get_df_fwd(start, end)
