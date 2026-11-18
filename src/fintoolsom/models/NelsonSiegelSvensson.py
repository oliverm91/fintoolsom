import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from scipy.optimize import minimize

import numpy as np

from ..fixedIncome.CLBonds import CLBond
from ..rates.Rates import Rate
from ..rates.PillarCurve import PillarCurve


# ── NSS formulas ──

HUMP_PEAK = 1.7932821329007607
"""Where the curvature loading h(x) = (1 - e^-x)/x - e^-x is highest.

h'(x) = 0 gives e^-x (x² + x + 1) = 1, i.e. x² + x + 1 = e^x, whose root other
than 0 has no closed form: x ≈ 1.7933, where h ≈ 0.298.
"""


def _lambda_for_hump(years: float) -> float:
    """The `lambda_` whose first hump peaks `years` out (x = t / lambda_)."""
    return years / HUMP_PEAK


def _mu_for_hump(years: float) -> float:
    """The `mu_` whose second hump peaks `years` out (x = mu_ · t)."""
    return HUMP_PEAK / years


# Calibration bounds and starting points, written as the years to each hump: the
# first between 0.5 and 3Y (around 2Y), the second between 2 and 20Y (around 5Y).
# A later hump is a larger lambda_ but a smaller mu_, so mu_'s bounds swap ends.
_HUMP_BOUNDS = (
    (_lambda_for_hump(0.5), _lambda_for_hump(3.0)),
    (_mu_for_hump(20.0), _mu_for_hump(2.0)),
)
_DEFAULT_HUMPS = (_lambda_for_hump(2.0), _mu_for_hump(5.0))


def nss_factors(t: float | np.ndarray, lambda_: float, mu_: float) -> np.ndarray:
    """Factor loadings [level, slope, curvature, 2nd curvature] at year fraction `t`.

    Both curvature loadings are h(x) = (1 - e^-x)/x - e^-x, the slope loading is
    (1 - e^-x)/x, with x = t / lambda_ for the first hump and x = mu_ · t for the
    second: `lambda_` is a time scale (the hump peaks at HUMP_PEAK · lambda_ years)
    and `mu_` a decay rate (the hump peaks at HUMP_PEAK / mu_ years). See
    :func:`lambda_for_hump` and :func:`mu_for_hump`.

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
    """Nelson-Siegel-Svensson continuously-compounded act/365 zero rate at year fraction `t`.

    `lambda_` is the first hump's time scale and `mu_` the second's decay rate (see
    :func:`nss_factors`)."""
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
    return np.exp(
        -nss_rate(t, beta0, beta1, beta2, beta3, lambda_, mu_)
        * np.asarray(t, dtype=float)
    )


@dataclass
class NelsonSiegelSvensson:
    """Nelson-Siegel-Svensson zero-rate model, continuously compounded Act/365.

    r(t) = b0 + b1·slope(t) + b2·h(t / lambda_) + b3·h(mu_ · t), with
    h(x) = (1 - e^-x)/x - e^-x (see :func:`nss_factors`).

    Attributes
    ----------
    b0, b1, b2, b3 : float
        Level, slope, first and second curvature.
    lambda_ : float
        Time scale of the first hump, which peaks at HUMP_PEAK · lambda_ years.
        Calibrated with the hump between 0.5 and 3 years.
    mu_ : float
        Decay rate of the second hump, which peaks at HUMP_PEAK / mu_ years.
        Calibrated with the hump between 2 and 20 years.
    first_hump, second_hump : float
        The years to each hump (read-only).
    """

    b0: float = field(init=False)
    b1: float = field(init=False)
    b2: float = field(init=False)
    b3: float = field(init=False)
    lambda_: float = field(init=False)
    mu_: float = field(init=False)

    def __post_init__(self):
        self.b0, self.b1, self.b2, self.b3 = 0.03, 0.01, 0, 0.01
        self.lambda_, self.mu_ = _DEFAULT_HUMPS  # humps at 2 and 5 years

    @property
    def first_hump(self) -> float:
        """Years to the first hump: HUMP_PEAK · lambda_."""
        return HUMP_PEAK * self.lambda_

    @property
    def second_hump(self) -> float:
        """Years to the second hump: HUMP_PEAK / mu_."""
        return HUMP_PEAK / self.mu_

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
        return self.calibrate_from_bonds(
            calibration_date, bonds_irr_list, initial_guess, method
        )

    def calibrate_from_bonds(
        self,
        calibration_date: date,
        bonds_irr_list: list[tuple[CLBond, Rate]],
        initial_guess: Optional[list[float]] = None,
        method: Optional[str] = "powell",
    ):
        """Fit the six parameters to the bonds' prices at their IRRs.

        `initial_guess` is ``[b0, b1, b2, b3, lambda_, mu_]`` in the model's own
        parameters (use :func:`lambda_for_hump` / :func:`mu_for_hump` to start from
        hump years); by default the betas come from the IRRs and the humps start at
        2 and 5 years.
        """
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
        n_bonds = len(bonds_ts)
        bond_index = np.repeat(np.arange(n_bonds), [len(ts) for ts in bonds_ts])

        def get_valuation_error(params):
            flows_vps = nss_df(flat_ts, *params) * flat_flows
            ns_pvs = np.bincount(bond_index, weights=flows_vps, minlength=n_bonds)
            return np.sum((ns_pvs - mkt_pvs) ** 2)

        if initial_guess is None:
            self.b0 = sum([irr.rate for _, irr in bonds_irr_list]) / len(bonds_irr_list)
            short_term_yields = [
                irr.rate for _, irr in bonds_irr_list[: min(3, len(bonds_irr_list))]
            ]
            self.b1 = sum(short_term_yields) / len(short_term_yields) - self.b0
            self.b2 = 0
            self.b3 = 0.01
            self.lambda_, self.mu_ = _DEFAULT_HUMPS
            initial_guess = [self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_]

        with np.errstate(over="ignore"):
            # Free betas; the first hump between 0.5 and 3Y, the second between 2 and 20Y.
            bounds = [(None, None)] * 4 + list(_HUMP_BOUNDS)
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
        curve: PillarCurve,
        initial_guess: Optional[list[float]] = None,
        method: str = "powell",
    ):
        """Fit the model to the curve's continuously compounded Act/365 zero rates at
        its knots after the curve date.

        `initial_guess` is ``[lambda_, mu_]`` in the model's own parameters (by
        default the humps at 2 and 5 years); the betas are then solved exactly by
        least squares.
        """
        # For fixed lambda_/mu_, the NSS rate is linear in b0..b3 (see _get_design_matrix), so betas
        # can be solved exactly via least squares instead of optimizing over all 6 params. Only
        # lambda_/mu_ need the nonlinear optimizer, and target rates are computed once upfront.
        pillars = [d for d in curve.knot_dates if d > curve.curve_date]
        t = np.array([(d - curve.curve_date).days for d in pillars]) / 365
        target_rates = -np.log(curve.get_dfs(pillars)) / t

        def get_valuation_error(tau_params):
            lambda_, mu_ = tau_params
            design_matrix = NelsonSiegelSvensson._get_design_matrix(t, lambda_, mu_)
            betas, *_ = np.linalg.lstsq(design_matrix, target_rates, rcond=None)
            fitted_rates = design_matrix @ betas
            return np.sum((fitted_rates - target_rates) ** 2)

        bounds = list(_HUMP_BOUNDS)  # first hump in 0.5-3Y, second in 2-20Y
        if initial_guess is None:
            initial_guess = list(_DEFAULT_HUMPS)
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
