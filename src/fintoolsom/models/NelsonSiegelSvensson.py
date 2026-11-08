import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from scipy.optimize import minimize

import numpy as np

from ..fixedIncome.CLBonds import CLBond
from ..rates.Rates import Rate, RateConvention, ExponentialInterestConvention
from ..rates.ZeroCouponCurve import ZeroCouponCurve
from ..dates import ActualDayCountConvention


# ── NSS formulas ──

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


@dataclass
class NelsonSiegelSvensson:
    b0: float = field(init=False)
    b1: float = field(init=False)
    b2: float = field(init=False)
    b3: float = field(init=False)
    lambda_: float = field(init=False)
    mu_: float = field(init=False)

    def __post_init__(self):
        self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_ = (
            0.03,
            0.01,
            0,
            0.01,
            0.5,
            0.2,
        )

    def get_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        return nss_rate(t, *self.get_params())

    def get_df(self, t: float | np.ndarray) -> float | np.ndarray:
        return nss_df(t, *self.get_params())

    def calibrate(
        self,
        calibration_date: date,
        bonds_irr_list: list[tuple[CLBond, Rate]],
        initial_guess: Optional[list[float]] = None,
        method: Optional[str] = "powell",
    ):
        """Deprecated alias of :meth:`calibrate_from_bonds`."""
        warnings.warn(
            "NelsonSiegelSvensson.calibrate is deprecated; use calibrate_from_bonds instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.calibrate_from_bonds(calibration_date, bonds_irr_list, initial_guess, method)

    def calibrate_from_bonds(
        self,
        calibration_date: date,
        bonds_irr_list: list[tuple[CLBond, Rate]],
        initial_guess: Optional[list[float]] = None,
        method: Optional[str] = "powell",
    ):
        bonds_irr_list.sort(
            key=lambda b_irr_tuple: (
                (b_irr_tuple[0].get_maturity_date() - calibration_date).days
            )
        )
        mkt_pvs = np.array(
            [
                bond.get_present_value(calibration_date, irr)
                for bond, irr in bonds_irr_list
            ]
        )
        bond_t_flows_lst = [
            (
                np.array(
                    [
                        (ed - calibration_date).days / 365
                        for ed in bond.coupons.get_remaining_end_dates(calibration_date)
                    ]
                ),
                bond.coupons.get_remaining_flows(calibration_date),
            )
            for bond, _ in bonds_irr_list
        ]
        bonds_ts, bonds_flows = zip(*bond_t_flows_lst)
        flat_flows = np.concatenate(bonds_flows)
        flat_ts = np.concatenate(bonds_ts)
        starts_ts = np.cumsum([len(bond_ts) for bond_ts in bonds_ts])
        starts_ts = np.insert(starts_ts, 0, 0)
        lengths_ts = np.diff(starts_ts)
        starts_ts = starts_ts[:-1]
        starts_lengths = list(zip(starts_ts, lengths_ts))
        ns_pvs = np.zeros(len(bonds_ts))

        def get_valuation_error(params):
            ns_dfs = nss_df(flat_ts, *params)
            flows_vps = ns_dfs * flat_flows
            for ix, start_length in enumerate(starts_lengths):
                start, length = start_length
                ns_pvs[ix] = np.sum(flows_vps[start : start + length])
            return sum((ns_pvs - mkt_pvs) ** 2)

        if initial_guess is None:
            self.b0 = sum([irr.rate for _, irr in bonds_irr_list]) / len(
                bonds_irr_list
            )
            short_term_yields = [
                irr.rate
                for _, irr in bonds_irr_list[: min(3, len(bonds_irr_list))]
            ]
            self.b1 = sum(short_term_yields) / len(short_term_yields) - self.b0
            self.b2 = 0
            self.b3 = 0.01
            self.lambda_ = 2  # First hump around 2Y
            self.mu_ = 5  # Second hump arond 5Y
            initial_guess = [self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_]

        with np.errstate(over="ignore"):
            bounds = [
                (None, None),  # beta0
                (None, None),  # beta1
                (None, None),  # beta2
                (None, None),  # beta3
                (0.5, 3),  # lambda | First hump to be between 0.5 and 7Y
                (2, 20),  # mu | Second hump to be 3 and 20Y
            ]
            result = minimize(
                get_valuation_error,
                initial_guess,
                method=method,
                bounds=bounds,
                options={"maxiter": 300},
                tol=0.01,
            )
            if result.fun > 1:
                raise ValueError(
                    f"NSS could not converge to a reasonable value. Optimization error: {result.fun}. Number of iterations: {result.nit}"
                )
            if result.fun > 0.1:
                result = minimize(
                    get_valuation_error,
                    result.x,
                    method=method,
                    bounds=bounds,
                    options={"maxiter": 300},
                    tol=0.01,
                )

        if result.success is False:
            raise ValueError(result.message)

        self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_ = result.x

    def calibrate_from_curve(
        self,
        curve: ZeroCouponCurve,
        initial_guess: Optional[list[float]] = None,
        method: str = "powell",
    ):
        # For fixed lambda_/mu_, the NSS rate is linear in b0..b3 (see _get_design_matrix), so betas
        # can be solved exactly via least squares instead of optimizing over all 6 params. Only
        # lambda_/mu_ need the nonlinear optimizer, and target rates are computed once upfront.
        rate_convention = RateConvention(
            interest_convention=ExponentialInterestConvention,
            day_count_convention=ActualDayCountConvention,
            time_fraction_base=365,
        )
        t = curve.days / 365
        target_rates = curve.get_zero_rates_values(rate_convention)

        def get_valuation_error(tau_params):
            lambda_, mu_ = tau_params
            design_matrix = NelsonSiegelSvensson._get_design_matrix(t, lambda_, mu_)
            betas, *_ = np.linalg.lstsq(design_matrix, target_rates, rcond=None)
            fitted_rates = design_matrix @ betas
            return np.sum((fitted_rates - target_rates) ** 2)

        bounds = [
            (0.5, 3),  # lambda | First hump to be between 0.5 and 3Y
            (2, 20),  # mu | Second hump to be between 2 and 20Y
        ]
        if initial_guess is None:
            initial_guess = [2, 5]  # First hump around 2Y, second hump around 5Y
        with np.errstate(over="ignore"):
            result = minimize(
                get_valuation_error,
                initial_guess,
                method=method,
                bounds=bounds,
                options={"maxiter": 300},
                tol=0.01,
            )

        if result.success is False:
            raise ValueError(result.message)

        self.lambda_, self.mu_ = result.x
        design_matrix = NelsonSiegelSvensson._get_design_matrix(
            t, self.lambda_, self.mu_
        )
        betas, *_ = np.linalg.lstsq(design_matrix, target_rates, rcond=None)
        self.b0, self.b1, self.b2, self.b3 = betas

    # The module-level formulas, as static methods.
    _get_design_matrix = staticmethod(nss_factors)
    _get_rate = staticmethod(nss_rate)
    _get_df = staticmethod(nss_df)

    def get_params(self) -> list[float]:
        return [self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_]
