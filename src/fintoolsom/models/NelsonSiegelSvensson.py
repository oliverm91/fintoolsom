"""The Nelson-Siegel family of zero-rate models: :class:`NelsonSiegel` and
:class:`NelsonSiegelSvensson`.

Both give a continuously compounded Act/365 zero rate that is linear in its betas
once the humps are fixed: r(t) = factors(t, *humps) @ betas. Each hump is the years
to the peak of its curvature loading. The two models are siblings: a Svensson model
is not a Nelson-Siegel one with an extra term, nor the reverse (each has its own
parameter count), but they share the fitting in :class:`_NelsonSiegelModel`.
"""

from dataclasses import dataclass, fields
from datetime import date
from typing import Callable, ClassVar, Optional, TypedDict

import numpy as np
from scipy.optimize import least_squares

from ..fixedIncome.CLBonds import CLBond
from ..rates.PillarCurve import PillarCurve
from ..rates.Rates import Rate

# ── Formulas ──

HUMP_PEAK = 1.7932821329007607
"""Where the curvature loading h(x) = (1 - e^-x)/x - e^-x is highest.

h'(x) = 0 gives e^-x (x² + x + 1) = 1, i.e. x² + x + 1 = e^x, whose root other
than 0 has no closed form: x ≈ 1.7933, where h ≈ 0.298.
"""


def _slope_and_curvature(t: np.ndarray, hump: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope s(x) = (1 - e^-x)/x and curvature h(x) = s(x) - e^-x at
    x = HUMP_PEAK · t / hump, so h peaks at t = `hump` years.

    At t=0 the formulas are 0/0; their limits 1 and 0 are used."""
    at_zero = t == 0.0
    x = HUMP_PEAK * np.where(at_zero, 1.0, t) / hump
    e_minus_x = np.exp(-x)
    slope = (1 - e_minus_x) / x
    return np.where(at_zero, 1.0, slope), np.where(at_zero, 0.0, slope - e_minus_x)


def ns_factors(t: float | np.ndarray, lambda_: float) -> np.ndarray:
    """Nelson-Siegel factor loadings [level, slope, curvature] at year fraction `t`.

    The curvature peaks at `lambda_` years (see :func:`_slope_and_curvature`).
    Shape (3,) for scalar `t`, (n, 3) for an array: r(t) = ns_factors(t) @ [b0, b1, b2].
    At t=0 the loadings are [1, 1, 0] (r(0) = b0 + b1, the instantaneous short rate)."""
    t = np.asarray(t, dtype=float)
    slope, curvature = _slope_and_curvature(t, lambda_)
    return np.stack((np.ones_like(t), slope, curvature), axis=-1)


def nss_factors(t: float | np.ndarray, lambda_: float, mu_: float) -> np.ndarray:
    """Svensson factor loadings [level, slope, curvature, 2nd curvature] at year
    fraction `t`: :func:`ns_factors` plus a second curvature peaking at `mu_` years.

    Shape (4,) for scalar `t`, (n, 4) for an array: r(t) = nss_factors(t) @
    [b0, b1, b2, b3]. At t=0 the loadings are [1, 1, 0, 0]."""
    t = np.asarray(t, dtype=float)
    slope, curvature = _slope_and_curvature(t, lambda_)
    _, curvature_2 = _slope_and_curvature(t, mu_)
    return np.stack((np.ones_like(t), slope, curvature, curvature_2), axis=-1)


def ns_rate(
    t: float | np.ndarray, beta0: float, beta1: float, beta2: float, lambda_: float
) -> float | np.ndarray:
    """Nelson-Siegel continuously compounded Act/365 zero rate at year fraction `t`."""
    return ns_factors(t, lambda_) @ np.array([beta0, beta1, beta2])


def ns_df(
    t: float | np.ndarray, beta0: float, beta1: float, beta2: float, lambda_: float
) -> float | np.ndarray:
    """Nelson-Siegel discount factor exp(-r(t)·t) at year fraction `t` (1 at t=0)."""
    return np.exp(
        -ns_rate(t, beta0, beta1, beta2, lambda_) * np.asarray(t, dtype=float)
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
    """Nelson-Siegel-Svensson continuously compounded Act/365 zero rate at year
    fraction `t`; `lambda_` and `mu_` are the years to each hump."""
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
    """Nelson-Siegel-Svensson discount factor exp(-r(t)·t) at year fraction `t` (1 at t=0)."""
    return np.exp(
        -nss_rate(t, beta0, beta1, beta2, beta3, lambda_, mu_)
        * np.asarray(t, dtype=float)
    )


# ── Models ──

# least_squares tolerances (function, step and gradient), and the size of the
# hump grid calibrate_from_curve starts from.
_TOL = 1e-12
_START_GRID_POINTS = 150


class CalibrationError(TypedDict):
    """The fit at one calibration point: fitted minus observed, in basis points.

    ``term`` is in years: a bond's duration on the fitted curve, or a pillar's
    Act/365 year fraction."""

    term: float
    error_bps: float


class _NelsonSiegelModel:
    """Rate, discount factors and calibration shared by the Nelson-Siegel family.

    A subclass is a dataclass whose fields are its betas, then its humps (years to
    each curvature peak), with their defaults as the starting point of a fit. It sets
    ``_factors`` (the loadings at ``t`` for the humps) and ``_HUMP_BOUNDS`` (one
    ``(low, high)`` per hump, in years).
    """

    _HUMP_BOUNDS: ClassVar[tuple[tuple[float, float], ...]]
    # Factor loadings at year fraction t for the humps, one column per beta.
    _factors: ClassVar[Callable[..., np.ndarray]]

    # ── Parameters ──

    def get_params(self) -> list[float]:
        """The betas, then the humps, in field order."""
        return [getattr(self, f.name) for f in fields(self)]  # type: ignore[arg-type]

    def _set_params(self, params) -> None:
        for f, value in zip(fields(self), params, strict=True):  # type: ignore[arg-type]
            setattr(self, f.name, float(value))

    def _default_params(self) -> list[float]:
        return [f.default for f in fields(self)]  # type: ignore[arg-type, misc]

    @property
    def _n_betas(self) -> int:
        return len(fields(self)) - len(self._HUMP_BOUNDS)  # type: ignore[arg-type]

    # ── Rates ──

    def _rate(self, t: float | np.ndarray, params) -> float | np.ndarray:
        n = self._n_betas
        return self._factors(t, *params[n:]) @ np.asarray(params[:n], dtype=float)

    def _df(self, t: float | np.ndarray, params) -> float | np.ndarray:
        return np.exp(-self._rate(t, params) * np.asarray(t, dtype=float))

    def get_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Continuously compounded Act/365 zero rate at year fraction `t`."""
        return self._rate(t, self.get_params())

    def get_df(self, t: float | np.ndarray) -> float | np.ndarray:
        """Discount factor exp(-r(t)·t) at year fraction `t` (1 at t=0)."""
        return self._df(t, self.get_params())

    # ── Calibration ──
    # Every year fraction a fit sees is > 0 (flows after the calibration date, knots
    # after the curve date), so its loadings skip the t=0 guard of the factor
    # functions. Both fits are bounded least squares on a residual vector with an
    # analytic Jacobian (scipy's trust-region reflective least_squares). Measured
    # from random starts, it was as robust as Levenberg-Marquardt with the humps
    # mapped into their bounds, and more robust than a logistic map, which pins a
    # hump that reaches a bound.

    @staticmethod
    def _calibration_design(t: np.ndarray, humps) -> np.ndarray:
        """Loadings [1, s(x1), h(x1), h(x2), ...] at year fractions t > 0, with
        x_j = HUMP_PEAK · t / hump_j: ``_factors`` without its t=0 guard.

        A hump may be an array of shape (G, 1), for one design per row: (G, n, k)."""
        shape = np.broadcast_shapes(t.shape, *(np.shape(h) for h in humps))
        design = np.empty((*shape, 2 + len(humps)))
        design[..., 0] = 1.0
        for j, hump in enumerate(humps):
            x = (HUMP_PEAK / hump) * t
            e_minus_x = np.exp(-x)
            slope = (1 - e_minus_x) / x
            if j == 0:
                design[..., 1] = slope
            design[..., 2 + j] = slope - e_minus_x
        return design

    @staticmethod
    def _design_and_derivatives(
        t: np.ndarray, humps
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """:meth:`_calibration_design` at t > 0 and its derivative in each hump.

        With x = HUMP_PEAK · t / hump: dx/dhump = -x / hump,
        ds/dx = (e^-x (1 + x) - 1) / x² and dh/dx = ds/dx + e^-x."""
        design = np.empty((t.size, 2 + len(humps)))
        design[:, 0] = 1.0
        derivatives = []
        for j, hump in enumerate(humps):
            x = (HUMP_PEAK / hump) * t
            e_minus_x = np.exp(-x)
            slope = (1 - e_minus_x) / x
            dslope_dx = (e_minus_x * (1 + x) - 1) / (x * x)
            dx_dhump = -x / hump
            derivative = np.zeros_like(design)
            if j == 0:
                design[:, 1] = slope
                derivative[:, 1] = dslope_dx * dx_dhump
            design[:, 2 + j] = slope - e_minus_x
            derivative[:, 2 + j] = (dslope_dx + e_minus_x) * dx_dhump
            derivatives.append(derivative)
        return design, derivatives

    def _hump_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = np.array(self._HUMP_BOUNDS, dtype=float).T
        return low, high

    def _least_squares(
        self,
        residuals: Callable[[np.ndarray], np.ndarray],
        jacobian: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        n_free: int,
    ) -> np.ndarray:
        """Minimise ||residuals(p)||² over p = [n_free free parameters, humps], each
        hump within its bounds: trust-region reflective with the analytic Jacobian.

        `x0`'s humps start at least 1% inside their bounds (0.25 -> 0.2525,
        20 -> 19.8): started on a bound, the solver can stall there. The margin is
        relative to the bound, not to the range: 1% of 0.25-20 would start a low
        hump at 0.45, where the objective is flat enough for the fit to stop short
        of the bound. The fit can still end on a bound. Raises ``ValueError`` if the
        solver does not converge."""
        low, high = self._hump_bounds()
        humps0 = np.clip(x0[n_free:], low * 1.01, high * 0.99)
        lower = np.concatenate((np.full(n_free, -np.inf), low))
        upper = np.concatenate((np.full(n_free, np.inf), high))
        with np.errstate(over="ignore"):
            result = least_squares(
                residuals,
                np.concatenate((x0[:n_free], humps0)),
                jac=jacobian,  # type: ignore[arg-type]
                bounds=(lower, upper),
                x_scale="jac",  # betas are ~0.01, humps ~10
                ftol=_TOL,
                xtol=_TOL,
                gtol=_TOL,
            )
        return self._converged(result)

    def _converged(self, result) -> np.ndarray:
        if not result.success:
            raise ValueError(
                f"{type(self).__name__} did not converge: {result.message}"
            )
        return result.x

    def calibrate_from_bonds(
        self,
        calibration_date: date,
        bonds_irr_list: list[tuple[CLBond, Rate]],
        initial_guess: Optional[list[float]] = None,
    ) -> list[CalibrationError]:
        """Fit every parameter to the bonds' present values at their IRRs.

        Least squares on one residual per bond: the model's PV of its remaining
        flows minus its PV at its IRR. The betas sit inside the discount factors'
        exponential, so all parameters are fitted jointly, with the analytic
        Jacobian dPV/dp = -Σ flow · t · df · dr(t)/dp.

        `initial_guess` is in :meth:`get_params` order (betas, then humps in years),
        clipped to the hump bounds. By default b0 is the average IRR, b1 the average
        of the three shortest IRRs minus b0, and the other parameters start at the
        model's defaults. Raises ``ValueError`` if the solver does not converge.

        Returns one :class:`CalibrationError` per bond, in the order given: `term`
        is its duration on the fitted curve, Σ t · flow · df / PV (Macaulay with the
        curve's dfs), and `error_bps` the IRR that prices the bond at its fitted PV
        minus its quoted IRR, in the quote's convention.
        """
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
        # Sums each bond's flows: (bonds, flows), one matrix product per evaluation.
        to_bonds = (bond_index == np.arange(n_bonds)[:, None]).astype(float)
        n_betas = self._n_betas

        def residuals(params: np.ndarray) -> np.ndarray:
            design = self._calibration_design(flat_ts, params[n_betas:])
            flows_pvs = np.exp(-(design @ params[:n_betas]) * flat_ts) * flat_flows
            return to_bonds @ flows_pvs - mkt_pvs

        def jacobian(params: np.ndarray) -> np.ndarray:
            betas = params[:n_betas]
            design, derivatives = self._design_and_derivatives(
                flat_ts, params[n_betas:]
            )
            dpv_dr = -np.exp(-(design @ betas) * flat_ts) * flat_flows * flat_ts
            dr_dparams = np.column_stack([design, *(d @ betas for d in derivatives)])
            return to_bonds @ (dpv_dr[:, None] * dr_dparams)

        if initial_guess is None:
            by_maturity = sorted(
                bonds_irr_list,
                key=lambda bond_irr: bond_irr[0].get_maturity_date(),
            )
            irrs = [irr.rate for _, irr in by_maturity]
            b0 = sum(irrs) / len(irrs)
            short_term_irrs = irrs[:3]
            b1 = sum(short_term_irrs) / len(short_term_irrs) - b0
            initial_guess = [b0, b1, *self._default_params()[2:]]

        x0 = np.asarray(initial_guess, dtype=float)
        params = self._least_squares(residuals, jacobian, x0, n_betas)
        self._set_params(params)

        design = self._calibration_design(flat_ts, params[n_betas:])
        flows_pvs = np.exp(-(design @ params[:n_betas]) * flat_ts) * flat_flows
        model_pvs = to_bonds @ flows_pvs
        durations = (to_bonds @ (flows_pvs * flat_ts)) / model_pvs
        errors: list[CalibrationError] = []
        for (bond, irr), model_pv, duration in zip(
            bonds_irr_list, model_pvs, durations
        ):
            model_irr = bond.get_irr_from_present_value(
                calibration_date,
                float(model_pv),
                irr.rate_convention,
                initial_guess=irr.rate,
            )
            errors.append(
                CalibrationError(
                    term=float(duration),
                    error_bps=1e4 * (float(model_irr.rate) - irr.rate),
                )
            )
        return errors

    def calibrate_from_curve(
        self,
        curve: PillarCurve,
        initial_guess: Optional[list[float]] = None,
    ) -> list[CalibrationError]:
        """Fit the model to the curve's continuously compounded Act/365 zero rates at
        its knots after the curve date.

        Variable projection: for given humps the rate is linear in the betas, so
        every evaluation solves the betas exactly for the humps it is given, and the
        least-squares solver only moves the humps. The betas never go stale: they
        are always the best ones for the current humps, and the optimum is that of
        fitting all parameters jointly. The Jacobian of the projected residual is
        Golub and Pereyra's (1973).

        `initial_guess` is the humps in years, clipped to their bounds. By default
        the solver starts from the best point of a coarse hump grid (about 150
        points, solved in one batched linear solve), which keeps two-hump fits out
        of local minima. Raises ``ValueError`` if the solver does not converge.

        Returns one :class:`CalibrationError` per knot after the curve date: `term`
        is its Act/365 year fraction, and `error_bps` the fitted zero rate minus the
        curve's, both continuously compounded Act/365.
        """
        pillars = [d for d in curve.knot_dates if d > curve.curve_date]
        t = np.array([(d - curve.curve_date).days for d in pillars]) / 365
        target_rates = -np.log(curve.get_dfs(pillars)) / t

        def gram_inverse(design: np.ndarray) -> np.ndarray:
            # Equal humps make NSS singular: then the pseudo-inverse.
            gram = design.T @ design
            try:
                return np.linalg.inv(gram)
            except np.linalg.LinAlgError:
                return np.linalg.pinv(gram)

        def residuals(humps: np.ndarray) -> np.ndarray:
            design = self._calibration_design(t, humps)
            return (
                design @ (gram_inverse(design) @ (design.T @ target_rates))
                - target_rates
            )

        def jacobian(humps: np.ndarray) -> np.ndarray:
            # r = X b - y with b = G⁻¹ X'y, G = X'X, P = X G⁻¹ X'. For each hump:
            # dr/dhump = (I - P) dX b - X G⁻¹ dX' r.
            design, derivatives = self._design_and_derivatives(t, humps)
            inverse = gram_inverse(design)
            betas = inverse @ (design.T @ target_rates)
            residual = design @ betas - target_rates
            columns = []
            for derivative in derivatives:
                moved = derivative @ betas
                columns.append(
                    moved
                    - design @ (inverse @ (design.T @ moved))
                    - design @ (inverse @ (derivative.T @ residual))
                )
            return np.column_stack(columns)

        if initial_guess is None:
            x0 = self._grid_start(t, target_rates, *self._hump_bounds())
        else:
            x0 = np.asarray(initial_guess, dtype=float)
        humps = self._least_squares(residuals, jacobian, x0, 0)
        design = self._calibration_design(t, humps)
        betas = gram_inverse(design) @ (design.T @ target_rates)
        self._set_params([*betas, *humps])
        fitted_rates = design @ betas
        return [
            CalibrationError(term=float(term), error_bps=1e4 * float(fitted - target))
            for term, fitted, target in zip(t, fitted_rates, target_rates)
        ]

    def _grid_start(
        self,
        t: np.ndarray,
        target_rates: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
    ) -> np.ndarray:
        """Humps of the best fit on a geometric grid inside the bounds; the betas of
        every grid point come from one batched solve of the normal equations."""
        n_humps = low.size
        per_hump = max(2, round(_START_GRID_POINTS ** (1 / n_humps)))
        axes = [np.geomspace(lo, hi, per_hump) for lo, hi in zip(low, high)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, n_humps)
        design = self._calibration_design(t, [grid[:, [j]] for j in range(n_humps)])
        gram = np.einsum("gnk,gnl->gkl", design, design)
        # A tiny ridge: grid points with (nearly) equal humps make NSS singular.
        ridge = 1e-12 * np.trace(gram, axis1=1, axis2=2)
        gram += ridge[:, None, None] * np.eye(design.shape[-1])
        rhs = np.einsum("gnk,n->gk", design, target_rates)
        betas = np.linalg.solve(gram, rhs[..., None])[..., 0]
        fitted = np.einsum("gnk,gk->gn", design, betas)
        return grid[np.argmin(np.sum((fitted - target_rates) ** 2, axis=1))]


@dataclass
class NelsonSiegel(_NelsonSiegelModel):
    """Nelson-Siegel zero-rate model, continuously compounded Act/365.

    r(t) = b0 + b1·s(x) + b2·h(x), x = HUMP_PEAK · t / lambda_, with
    s(x) = (1 - e^-x)/x and h(x) = s(x) - e^-x (see :func:`ns_factors`).

    Attributes
    ----------
    b0, b1, b2 : float
        Level, slope and curvature.
    lambda_ : float
        Years to the hump. Calibrated between 0.25 and 20 years.
    """

    b0: float = 0.03
    b1: float = 0.01
    b2: float = 0.0
    lambda_: float = 2.0

    _HUMP_BOUNDS = ((0.25, 20.0),)
    _factors = staticmethod(ns_factors)


@dataclass
class NelsonSiegelSvensson(_NelsonSiegelModel):
    """Nelson-Siegel-Svensson zero-rate model, continuously compounded Act/365.

    r(t) = b0 + b1·s(x) + b2·h(x) + b3·h(y), x = HUMP_PEAK · t / lambda_,
    y = HUMP_PEAK · t / mu_ (see :func:`nss_factors`).

    Attributes
    ----------
    b0, b1, b2, b3 : float
        Level, slope, first and second curvature.
    lambda_ : float
        Years to the first hump. Calibrated between 0.25 and 3 years.
    mu_ : float
        Years to the second hump. Calibrated between 2 and 20 years.
    """

    b0: float = 0.03
    b1: float = 0.01
    b2: float = 0.0
    b3: float = 0.01
    lambda_: float = 2.0
    mu_: float = 5.0

    _HUMP_BOUNDS = ((0.25, 3.0), (2.0, 20.0))
    _factors = staticmethod(nss_factors)
