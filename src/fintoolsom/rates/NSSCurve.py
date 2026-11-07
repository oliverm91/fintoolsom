from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Self

import numpy as np

from .Curve import Curve, forward_rates_from_wfs
from .Rates import ExponentialInterestConvention, Rate, RateConvention
from ..dates import ActualDayCountConvention

# NSS zero rates are continuously compounded on an act/365 time axis.
NSS_RATE_CONVENTION = RateConvention(ExponentialInterestConvention, ActualDayCountConvention, 365)
_DAY_BASE = 365


# ── NSS formulas: the single source for both NSSCurve and models.NelsonSiegelSvensson ──

def nss_factors(t: float | np.ndarray, lambda_: float, mu_: float) -> np.ndarray:
    """Factor loadings [level, slope, curvature, 2nd curvature] at year fraction `t`.

    Shape (4,) for scalar `t`, (n, 4) for an array: the zero rate is linear in the
    betas, r(t) = nss_factors(t) @ [b0, b1, b2, b3]. At t=0 the formulas are 0/0; the
    limits [1, 1, 0, 0] are used (r(0) = b0 + b1, the instantaneous short rate)."""
    t = np.asarray(t, dtype=float)
    at_zero = t == 0.0
    safe_t = np.where(at_zero, 1.0, t)
    lambda_t = safe_t / lambda_
    mu_t = mu_ * safe_t
    e_minus_lambda_t = np.exp(-lambda_t)
    e_minus_mu_t = np.exp(-mu_t)
    slope = (1 - e_minus_lambda_t) / lambda_t
    curvature = slope - e_minus_lambda_t
    curvature_2 = (1 - e_minus_mu_t) / mu_t - e_minus_mu_t
    return np.stack(
        (
            np.ones_like(t),
            np.where(at_zero, 1.0, slope),
            np.where(at_zero, 0.0, curvature),
            np.where(at_zero, 0.0, curvature_2),
        ),
        axis=-1,
    )


def nss_rate(
    t: float | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lambda_: float,
    mu_: float,
) -> float | np.ndarray:
    """Nelson-Siegel-Svensson continuously-compounded act/365 zero rate at year fraction `t`."""
    return nss_factors(t, lambda_, mu_) @ np.array([beta0, beta1, beta2, beta3])


def nss_df(
    t: float | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lambda_: float,
    mu_: float,
) -> float | np.ndarray:
    """Discount factor exp(-r(t)·t) at year fraction `t` (1 at t=0)."""
    return np.exp(-nss_rate(t, beta0, beta1, beta2, beta3, lambda_, mu_) * np.asarray(t, dtype=float))


@dataclass(slots=True)
class NSSCurve(Curve):
    """Curve backed by Nelson-Siegel-Svensson parameters, evaluated in closed form at
    any date (no pillars, no interpolation).

    `params_date` is the date the parameters' t=0 refers to (the calibration date).
    It equals `curve_date` unless the curve was aged; an aged curve keeps the original
    parameters and rebases: df(t) = df_params(t) / df_params(curve_date)."""

    curve_date: date
    b0: float
    b1: float
    b2: float
    b3: float
    lambda_: float
    mu_: float
    params_date: date | None = field(default=None)

    def __post_init__(self):
        if self.params_date is None:
            self.params_date = self.curve_date
        if self.params_date > self.curve_date:
            raise ValueError(
                f"params_date {self.params_date} must not be after curve_date {self.curve_date}."
            )

    @property
    def params(self) -> list[float]:
        return [self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_]

    def _params_dfs(self, dates: list[date]) -> np.ndarray:
        """DFs from params_date."""
        t = np.array([(d - self.params_date).days for d in dates], dtype=float) / _DAY_BASE
        return np.asarray(nss_df(t, *self.params))

    # ── Discount and wealth factors ──────────────────────────────────────────

    def get_df(self, date: date) -> float:
        return float(self.get_dfs([date])[0])

    def get_dfs(self, dates_t: list[date]) -> np.ndarray:
        dfs = self._params_dfs(list(dates_t))
        if self.params_date != self.curve_date:
            dfs = dfs / self._params_dfs([self.curve_date])[0]
        return dfs

    def get_df_fwd(self, start_date: date, end_date: date) -> float:
        return self.get_df(end_date) / self.get_df(start_date)

    def get_dfs_fwds(self, start_dates: list[date], end_dates: list[date]) -> np.ndarray:
        if len(start_dates) != len(end_dates):
            raise ValueError(
                f"Start and end dates must have the same length. Start dates: {start_dates}, end dates: {end_dates}"
            )
        return self.get_dfs(end_dates) / self.get_dfs(start_dates)

    def get_wf(self, date: date) -> float:
        return 1 / self.get_df(date)

    def get_wfs(self, dates: list[date]) -> np.ndarray:
        return 1 / self.get_dfs(dates)

    def get_wf_fwd(self, start_date: date, end_date: date) -> float:
        return 1 / self.get_df_fwd(start_date, end_date)

    def get_wfs_fwds(self, start_dates: list[date], end_dates: list[date]) -> np.ndarray:
        return 1 / self.get_dfs_fwds(start_dates, end_dates)

    # ── Rates ────────────────────────────────────────────────────────────────

    def get_zero_rate(self, date: date, rate_convention: RateConvention | None = None) -> Rate:
        if rate_convention is None:
            rate_convention = NSS_RATE_CONVENTION
        yf = rate_convention.day_count_convention.get_time_fraction(
            self.curve_date, date, rate_convention.time_fraction_base
        )
        r = rate_convention.interest_convention.get_rate_from_df(self.get_df(date), yf)
        return Rate(rate_convention, r)

    def get_forward_rates(
        self,
        start_dates: date | list[date],
        end_dates: date | list[date],
        rate_convention: RateConvention,
    ) -> Rate | list[Rate]:
        if isinstance(start_dates, date):
            fwd_wfs = self.get_wf_fwd(start_dates, end_dates)  # type: ignore[arg-type]
        else:
            fwd_wfs = self.get_wfs_fwds(start_dates, end_dates)  # type: ignore[arg-type]
        return forward_rates_from_wfs(fwd_wfs, start_dates, end_dates, rate_convention)

    def get_forward_rates_values(
        self,
        start_dates: date | list[date],
        end_dates: date | list[date],
        rate_convention: RateConvention | None = None,
    ) -> float | np.ndarray:
        rates_obj = self.get_forward_rates(start_dates, end_dates, rate_convention or NSS_RATE_CONVENTION)
        if isinstance(rates_obj, list):
            return np.array([r.rate for r in rates_obj])
        return rates_obj.rate

    # ── Interest ─────────────────────────────────────────────────────────────

    def get_accrued_interest(self, notional: float, start_date: date, end_date: date) -> float:
        return notional * (self.get_wf_fwd(start_date, end_date) - 1)

    def get_accrued_interests(
        self, notional: np.ndarray, start_dates: list[date], end_dates: list[date]
    ) -> np.ndarray:
        return notional * (self.get_wfs_fwds(start_dates, end_dates) - 1)

    # ── Transformations ──────────────────────────────────────────────────────

    def parallel_bump_rates_bps(self, bps: float) -> None:
        """b0 is the level factor: shifting it shifts every continuous zero rate by
        `bps`, exactly (also on an aged curve, whose rates are ratios of NSS dfs)."""
        if isinstance(bps, np.ndarray):
            bps = float(bps[0])
        self.b0 += bps / 10_000

    def get_aged_curve(self, date: date) -> Self:
        if date <= self.curve_date:
            raise ValueError(f"Date {date} must be posterior to curve date {self.curve_date}.")
        return replace(self, curve_date=date)

    def copy(self) -> Self:
        return replace(self)

    def __copy__(self) -> Self:
        return self.copy()

    def __str__(self) -> str:
        return (
            f"NSSCurve class. Curve date {self.curve_date}. Params "
            f"(b0={self.b0}, b1={self.b1}, b2={self.b2}, b3={self.b3}, "
            f"lambda={self.lambda_}, mu={self.mu_}) from {self.params_date}"
        )
