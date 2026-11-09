from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Self

import numpy as np

from .Rates import Rate, RateConvention


class LegacyCurve(ABC):
    """Legacy term structure (:class:`ZeroCouponCurve`): maps a date ``t`` to a
    discount factor, and through it to wealth factors, zero rates and forward rates.

    Kept for the bond tools (``CLBonds``, Nelson-Siegel-Svensson fitting). Pricing,
    the market and the curve builder use :class:`fintoolsom.rates.Curve`.

    Backends (pillars + interpolation, Nelson-Siegel-Svensson, ...) differ only in
    how they produce discount factors. Every query below is abstract so that each
    backend must implement all of them: code written against ``LegacyCurve`` works with
    any backend. Backend-specific mechanics (pillars, calibration) are not part of
    this interface."""

    curve_date: date

    # ── Discount and wealth factors ──────────────────────────────────────────

    @abstractmethod
    def get_df(self, date: date) -> float: ...

    @abstractmethod
    def get_dfs(self, dates_t: list[date]) -> np.ndarray: ...

    @abstractmethod
    def get_df_fwd(self, start_date: date, end_date: date) -> float: ...

    @abstractmethod
    def get_dfs_fwds(
        self, start_dates: list[date], end_dates: list[date]
    ) -> np.ndarray: ...

    @abstractmethod
    def get_wf(self, date: date) -> float: ...

    @abstractmethod
    def get_wfs(self, dates: list[date]) -> np.ndarray: ...

    @abstractmethod
    def get_wf_fwd(self, start_date: date, end_date: date) -> float: ...

    @abstractmethod
    def get_wfs_fwds(
        self, start_dates: list[date], end_dates: list[date]
    ) -> np.ndarray: ...

    # ── Rates ────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_zero_rate(
        self, date: date, rate_convention: RateConvention | None = None
    ) -> Rate: ...

    @abstractmethod
    def get_forward_rates(
        self,
        start_dates: date | list[date],
        end_dates: date | list[date],
        rate_convention: RateConvention,
    ) -> Rate | list[Rate]: ...

    @abstractmethod
    def get_forward_rates_values(
        self,
        start_dates: date | list[date],
        end_dates: date | list[date],
        rate_convention: RateConvention | None = None,
    ) -> float | np.ndarray: ...

    # ── Interest ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_accrued_interest(
        self, notional: float, start_date: date, end_date: date
    ) -> float: ...

    @abstractmethod
    def get_accrued_interests(
        self, notional: np.ndarray, start_dates: list[date], end_dates: list[date]
    ) -> np.ndarray: ...

    # ── Transformations ──────────────────────────────────────────────────────

    @abstractmethod
    def parallel_bump_rates_bps(self, bps: float) -> None:
        """Shift every zero rate by `bps` basis points, in place, in the backend's
        native rate convention."""

    @abstractmethod
    def get_aged_curve(self, date: date) -> Self:
        """The curve as seen from `date` (> curve_date): df_aged(t) = df(t) / df(date)."""

    @abstractmethod
    def copy(self) -> Self: ...

    @abstractmethod
    def __copy__(self) -> Self: ...


def forward_rates_from_wfs(
    fwd_wfs: float | np.ndarray,
    start_dates: date | list[date],
    end_dates: date | list[date],
    rate_convention: RateConvention,
) -> Rate | list[Rate]:
    """Express forward wealth factors as `Rate` objects in `rate_convention`: a single
    Rate for scalar dates, a list for date lists. Shared by LegacyCurve backends."""
    tfs = rate_convention.day_count_convention.get_time_fraction(
        start_dates, end_dates, rate_convention.time_fraction_base
    )
    fwd_rates = rate_convention.interest_convention.get_rate_from_wf(fwd_wfs, tfs)
    if isinstance(fwd_rates, np.ndarray):
        return [Rate(rate_convention, rate) for rate in fwd_rates]
    return Rate(rate_convention, fwd_rates)
