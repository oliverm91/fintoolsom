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
from typing import (
    Callable,
    ClassVar,
    NamedTuple,
    Optional,
    Self,
    Sequence,
    TypedDict,
)

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
# Batched calibration: iterations of its damped Newton before a curve falls
# back to the one-curve fit, and the largest (grid, curves, pillars, betas) block
# its grid start evaluates at once.
_MAX_BATCH_ITERATIONS = 100
# Second-to-first hump ratio under which a grid start counts as two equal (or
# swapped) humps, and is retried from humps apart (see _retry_apart).
_HUMPS_APART = 1.5
# Batched bond calibration: iterations of its Gauss-Newton before a set falls back
# to the one-set fit (also the IRR solves' cap), and the largest step of a log hump.
_MAX_BOND_ITERATIONS = 200
_MAX_LOG_HUMP_STEP = 0.5
_GRID_BLOCK = 2_000_000


def _curve_points(curve: PillarCurve) -> tuple[np.ndarray, np.ndarray]:
    """Act/365 year fractions of the curve's knots after its curve date and its
    continuously compounded Act/365 zero rates there."""
    pillars = [d for d in curve.knot_dates if d > curve.curve_date]
    t = np.array([(d - curve.curve_date).days for d in pillars]) / 365
    return t, -np.log(curve.get_dfs(pillars)) / t


def _start_grid(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Geometric grid of about ``_START_GRID_POINTS`` hump vectors inside the bounds:
    shape (G, number of humps)."""
    n_humps = low.size
    per_hump = max(2, round(_START_GRID_POINTS ** (1 / n_humps)))
    axes = [np.geomspace(lo, hi, per_hump) for lo, hi in zip(low, high)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, n_humps)


def _stack_curves(curves) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """The curves' pillars padded to one length: year fractions (C, N), zero rates
    (C, N), a 0/1 mask of the real pillars (C, N) and each curve's pillar count.

    Every curve keeps its own year fractions: curves need not share tenors. A
    padded slot has t = 1 (any t > 0 keeps the loadings finite) and weight 0."""
    points = [_curve_points(curve) for curve in curves]
    sizes = [t.size for t, _ in points]
    t = np.ones((len(points), max(sizes)))
    rates = np.zeros_like(t)
    mask = np.zeros_like(t)
    for i, (ti, ri) in enumerate(points):
        t[i, : ti.size], rates[i, : ti.size], mask[i, : ti.size] = ti, ri, 1.0
    return t, rates, mask, sizes


def _solve(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Stacked solves of (..., k, k) systems for (..., k) right-hand sides, with a
    ridge of 1e-12 of the trace: (nearly) equal NSS humps make the gram singular."""
    ridge = 1e-12 * np.trace(gram, axis1=-2, axis2=-1)
    gram = gram + ridge[..., None, None] * np.eye(gram.shape[-1])
    return np.linalg.solve(gram, rhs[..., None])[..., 0]


def _projection(
    design: np.ndarray, rates: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Best betas of every curve for its (unmasked) design (C, N, k) and the masked
    residual fitted minus target (C, N)."""
    weighted = design * mask[..., None]
    target = rates * mask
    betas = _solve(_gram(weighted), _transposed_times(weighted, target))
    return betas, _times(weighted, betas) - target


class _StackedBonds(NamedTuple):
    """Bond sets padded to B bonds of F flows each (see :func:`_stack_bond_sets`)."""

    t: np.ndarray  # (S, B·F) Act/365 year fractions of the flows, 1 when padded
    flows: np.ndarray  # (S, B·F) remaining flows, 0 when padded
    market_pvs: np.ndarray  # (S, B) PV at the quoted IRR, 0 for a padded bond
    quote_tf: np.ndarray  # (S, B, F) year fractions in each quote's day count
    quote_rates: np.ndarray  # (S, B) quoted IRRs
    conventions: np.ndarray  # (S, B) quote interest conventions, None when padded
    sizes: list[int]  # bonds per set
    n_flows: int  # F


def _stack_bond_sets(bond_sets) -> _StackedBonds:
    """The remaining flows of every bond of every set, padded: their Act/365 year
    fractions from the set's calibration date (as :meth:`calibrate_from_bonds`),
    their year fractions in the quote's day count, and the PV at the quote."""
    rows = []
    for calibration_date, bonds in bond_sets:
        set_rows = []
        for bond, irr in bonds:
            dates = bond.coupons.get_remaining_end_dates(calibration_date)
            convention = irr.rate_convention
            set_rows.append(
                (
                    np.array([(d - calibration_date).days for d in dates]) / 365,
                    np.asarray(
                        bond.coupons.get_remaining_flows(calibration_date), dtype=float
                    ),
                    np.asarray(
                        convention.day_count_convention.get_time_fraction(
                            calibration_date, dates, convention.time_fraction_base
                        ),
                        dtype=float,
                    ),
                    irr.rate,
                    convention.interest_convention,
                    bond.get_present_value(calibration_date, irr),
                )
            )
        rows.append(set_rows)
    sizes = [len(set_rows) for set_rows in rows]
    n_sets, n_bonds = len(rows), max(sizes)
    n_flows = max(row[0].size for set_rows in rows for row in set_rows)
    t = np.ones((n_sets, n_bonds, n_flows))
    flows = np.zeros_like(t)
    quote_tf = np.ones_like(t)
    market_pvs = np.zeros((n_sets, n_bonds))
    quote_rates = np.zeros_like(market_pvs)
    conventions = np.full((n_sets, n_bonds), None, dtype=object)
    for s, set_rows in enumerate(rows):
        for b, (ti, fi, qi, rate, convention, pv) in enumerate(set_rows):
            t[s, b, : ti.size], flows[s, b, : fi.size] = ti, fi
            quote_tf[s, b, : qi.size] = qi
            quote_rates[s, b], conventions[s, b], market_pvs[s, b] = (
                rate,
                convention,
                pv,
            )
    return _StackedBonds(
        t.reshape(n_sets, -1),
        flows.reshape(n_sets, -1),
        market_pvs,
        quote_tf,
        quote_rates,
        conventions,
        sizes,
        n_flows,
    )


def _quote_dfs(stacked: _StackedBonds, rates: np.ndarray) -> np.ndarray:
    """Discount factors of every flow at `rates` (S, B), each bond in its quote's
    convention: (S, B, F), 1 for a padded bond."""
    dfs = np.ones_like(stacked.quote_tf)
    for convention in {c for c in stacked.conventions.ravel() if c is not None}:
        at = stacked.conventions == convention
        dfs[at] = 1 / convention.get_wf_from_rate(
            rates[at][:, None], stacked.quote_tf[at]
        )
    return dfs


def _batched_irrs(stacked: _StackedBonds, target_pvs: np.ndarray) -> np.ndarray:
    """The IRR of every bond, in its quote's convention, at which its remaining
    flows are worth `target_pvs` (S, B): Newton from the quoted IRRs with a central
    difference slope, all bonds at once."""
    shape = stacked.market_pvs.shape + (stacked.n_flows,)
    flows = stacked.flows.reshape(shape)
    rates = stacked.quote_rates.copy()
    bump = 1e-6
    for _ in range(_MAX_BOND_ITERATIONS):
        pv = (flows * _quote_dfs(stacked, rates)).sum(-1)
        slope = (
            (flows * _quote_dfs(stacked, rates + bump)).sum(-1)
            - (flows * _quote_dfs(stacked, rates - bump)).sum(-1)
        ) / (2 * bump)
        step = np.divide(
            pv - target_pvs, slope, out=np.zeros_like(pv), where=slope != 0
        )
        rates -= step
        if np.max(np.abs(step)) < 1e-14:
            break
    return rates


# Stacked products of designs (..., n, k) by batched matmul (faster than einsum).
def _gram(design: np.ndarray) -> np.ndarray:
    """X'X: (..., k, k)."""
    return design.swapaxes(-1, -2) @ design


def _times(design: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """X b for b of shape (..., k): (..., n)."""
    return (design @ vector[..., None])[..., 0]


def _transposed_times(design: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """X'y for y of shape (..., n): (..., k)."""
    return (vector[..., None, :] @ design)[..., 0, :]


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
        ds/dx = (e^-x (1 + x) - 1) / x² and dh/dx = ds/dx + e^-x. Broadcasts like
        :meth:`_calibration_design` (t of shape (C, n) with humps of shape (C, 1)
        gives one design per curve: (C, n, k))."""
        shape = np.broadcast_shapes(t.shape, *(np.shape(h) for h in humps))
        design = np.empty((*shape, 2 + len(humps)))
        design[..., 0] = 1.0
        derivatives = []
        for j, hump in enumerate(humps):
            x = (HUMP_PEAK / hump) * t
            e_minus_x = np.exp(-x)
            slope = (1 - e_minus_x) / x
            dslope_dx = (e_minus_x * (1 + x) - 1) / (x * x)
            dx_dhump = -x / hump
            derivative = np.zeros_like(design)
            if j == 0:
                design[..., 1] = slope
                derivative[..., 1] = dslope_dx * dx_dhump
            design[..., 2 + j] = slope - e_minus_x
            derivative[..., 2 + j] = (dslope_dx + e_minus_x) * dx_dhump
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
        t, target_rates = _curve_points(curve)

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
        grid = _start_grid(low, high)
        design = self._calibration_design(t, [grid[:, [j]] for j in range(n_humps)])
        gram = np.einsum("gnk,gnl->gkl", design, design)
        # A tiny ridge: grid points with (nearly) equal humps make NSS singular.
        ridge = 1e-12 * np.trace(gram, axis1=1, axis2=2)
        gram += ridge[:, None, None] * np.eye(design.shape[-1])
        rhs = np.einsum("gnk,n->gk", design, target_rates)
        betas = np.linalg.solve(gram, rhs[..., None])[..., 0]
        fitted = np.einsum("gnk,gk->gn", design, betas)
        return grid[np.argmin(np.sum((fitted - target_rates) ** 2, axis=1))]

    # ── Batched calibration ──
    # C curves at once, their pillars padded to N slots with a 0/1 mask (see
    # _stack_curves): masked rows of the design and the rates are zero, so a padded
    # slot adds nothing to any gram, residual or Jacobian.

    @classmethod
    def calibrate_many_from_curves(
        cls,
        curves: Sequence[PillarCurve],
        global_humps: bool = False,
        initial_guess: Optional[list[float]] = None,
    ) -> tuple[list[Self], list[list[CalibrationError]]]:
        """Fit one model per curve, as :meth:`calibrate_from_curve` on each, in
        batched numpy.

        The curves need not share pillars: each keeps its own year fractions.

        With ``global_humps=False`` (default) every curve gets its own humps and
        betas: the same fits as a loop of :meth:`calibrate_from_curve`, to solver
        tolerance. The hump grid start is scored for every curve in one batched
        solve, then a batched, bounded damped Newton on the variable-projection
        error (in log humps) refines every curve's humps together, each stopping
        when it converges. A curve still not converged after
        ``_MAX_BATCH_ITERATIONS`` is refitted alone by :meth:`calibrate_from_curve`.

        With ``global_humps=True`` all curves share one set of humps (NS: lambda_,
        NSS: lambda_ and mu_), each curve keeping its own betas: the humps minimise
        the sum of every curve's squared rate errors, as Diebold and Li's fixed
        lambda. The betas then share one set of loadings over the curves, so their
        series read as level, slope and curvature on a fixed basis; each curve fits
        worse than with its own humps.

        Parameters
        ----------
        curves : sequence of PillarCurve
            Curves to fit; their zero rates at the knots after each curve date, as
            :meth:`calibrate_from_curve`.
        global_humps : bool, default False
            One set of humps for every curve instead of one per curve.
        initial_guess : list of float, optional
            Humps in years to start every curve from (or the shared humps),
            clipped to their bounds, instead of the grid start. Useful as a warm
            start, e.g. the previous day's humps.

        Returns
        -------
        tuple of (list of model, list of list of CalibrationError)
            One fitted model per curve, in order, and its errors as
            :meth:`calibrate_from_curve` returns them.

        Raises
        ------
        ValueError
            If a solver does not converge.
        """
        models: list[Self] = []
        errors: list[list[CalibrationError]] = []
        if not curves:
            return models, errors
        model = cls()
        t, rates, mask, sizes = _stack_curves(curves)
        n_curves, n_humps = t.shape[0], len(cls._HUMP_BOUNDS)
        low, high = model._hump_bounds()
        if initial_guess is not None:
            start = np.asarray(initial_guess, dtype=float)
            starts = np.broadcast_to(start, (n_curves, n_humps))
        else:
            grid = _start_grid(low, high)
            sse = model._batched_grid_sse(t, rates, mask, grid)
            start = grid[np.argmin(sse.sum(axis=1))]
            starts = grid[np.argmin(sse, axis=0)]
        if global_humps:
            humps = model._fit_global_humps(t, rates, mask, start)
            humps = np.broadcast_to(humps, (n_curves, n_humps))
        else:
            starts = np.clip(starts, low * 1.01, high * 0.99)
            humps, converged = model._batched_newton(t, rates, mask, starts)
            if initial_guess is None and n_humps > 1:
                model._retry_apart(t, rates, mask, grid, sse, starts, humps, converged)
            for i in np.flatnonzero(~converged):
                single = cls()
                single.calibrate_from_curve(
                    curves[int(i)], initial_guess=list(humps[i])
                )
                humps[i] = single.get_params()[single._n_betas :]
        design = model._calibration_design(t, [humps[:, [j]] for j in range(n_humps)])
        betas, residual = _projection(design, rates, mask)
        fitted = residual + rates * mask
        for i, size in enumerate(sizes):
            models.append(cls(*betas[i], *humps[i]))
            errors.append(
                [
                    CalibrationError(
                        term=float(t[i, j]),
                        error_bps=1e4 * float(fitted[i, j] - rates[i, j]),
                    )
                    for j in range(size)
                ]
            )
        return models, errors

    def _retry_apart(
        self,
        t: np.ndarray,
        rates: np.ndarray,
        mask: np.ndarray,
        grid: np.ndarray,
        sse: np.ndarray,
        starts: np.ndarray,
        humps: np.ndarray,
        converged: np.ndarray,
    ) -> None:
        """Refit from humps apart the curves whose grid start has two near-equal
        or swapped humps, keeping the better fit (in place in `humps` and
        `converged`).

        Equal humps make the two curvature loadings one: the model drops to three
        factors and the error is a flat, degenerate ridge, along which a start
        there can slide into a worse basin, or into the swapped region where the
        first hump is the longer. Such curves (second hump under ``_HUMPS_APART``
        times the first) also start from their best grid point whose second hump
        is at least that."""
        apart = grid[:, -1] >= _HUMPS_APART * grid[:, 0]
        near = np.flatnonzero(starts[:, -1] < _HUMPS_APART * starts[:, 0])
        if near.size == 0 or not apart.any():
            return
        low, high = self._hump_bounds()
        candidates = grid[apart]
        others = candidates[np.argmin(sse[apart][:, near], axis=0)]
        others = np.clip(others, low * 1.01, high * 0.99)
        sub = (t[near], rates[near], mask[near])
        refit, refit_converged = self._batched_newton(*sub, others)
        before = np.sum(self._projected_residual(*sub, humps[near])[0] ** 2, axis=1)
        after = np.sum(self._projected_residual(*sub, refit)[0] ** 2, axis=1)
        better = after < before
        humps[near[better]] = refit[better]
        converged[near[better]] = refit_converged[better]

    def _batched_grid_sse(
        self, t: np.ndarray, rates: np.ndarray, mask: np.ndarray, grid: np.ndarray
    ) -> np.ndarray:
        """Squared error of the best betas at every grid point for every curve:
        (G, C), in blocks of curves of at most ``_GRID_BLOCK`` design entries."""
        n_grid, n_humps = grid.shape
        n_curves, n_slots = t.shape
        sse = np.empty((n_grid, n_curves))
        block = max(1, _GRID_BLOCK // (n_grid * n_slots * (2 + n_humps)))
        # A grid column takes few distinct humps (12 per hump for NSS): loadings
        # are computed once per distinct hump and gathered into each grid point.
        axes = [np.unique(grid[:, j], return_inverse=True) for j in range(n_humps)]
        for s in range(0, n_curves, block):
            part = slice(s, s + block)
            design = np.empty((n_grid, *t[part].shape, 2 + n_humps))  # (G, c, N, k)
            design[..., 0] = 1.0
            for j, (values, at) in enumerate(axes):
                x = (HUMP_PEAK / values[:, None, None]) * t[None, part]
                e_minus_x = np.exp(-x)
                slope = (1 - e_minus_x) / x
                if j == 0:
                    design[..., 1] = slope[at]
                design[..., 2 + j] = (slope - e_minus_x)[at]
            design *= mask[None, part, :, None]
            target = rates[part] * mask[part]
            rhs = _transposed_times(design, target)
            betas = _solve(_gram(design), rhs)
            # Least-squares error without forming the fit: |y|² - b'X'y.
            sse[:, part] = np.sum(target**2, axis=-1) - np.sum(betas * rhs, axis=-1)
        return sse

    def _projected_residual(
        self,
        t: np.ndarray,
        rates: np.ndarray,
        mask: np.ndarray,
        humps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Variable-projection residual of every curve at its humps (C, N) and its
        Jacobian in the humps (C, N, number of humps), as
        :meth:`calibrate_from_curve`'s (Golub and Pereyra), batched over curves."""
        n_humps = humps.shape[1]
        design, derivatives = self._design_and_derivatives(
            t, [humps[:, [j]] for j in range(n_humps)]
        )
        design *= mask[..., None]
        target = rates * mask
        gram = _gram(design)
        betas = _solve(gram, _transposed_times(design, target))
        residual = _times(design, betas) - target
        columns = []
        for derivative in derivatives:
            derivative *= mask[..., None]
            moved = _times(derivative, betas)
            back = _solve(
                gram,
                _transposed_times(design, moved)
                + _transposed_times(derivative, residual),
            )
            columns.append(moved - _times(design, back))
        return residual, np.stack(columns, axis=-1)

    def _batched_newton(
        self,
        t: np.ndarray,
        rates: np.ndarray,
        mask: np.ndarray,
        humps0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each curve's humps minimising its projected squared error, in their bounds.

        Damped Newton on u = log(hump) (humps span 0.25-20 years), every curve with
        its own damping. The gradient is the analytic J'r; the Hessian is its
        forward difference (one extra evaluation per hump), not the Gauss-Newton
        J'J: on noisy rates the second-order term sum(r * d2r) dominates the hump
        curvature, J'J overshoots and a Levenberg-Marquardt crawls along the
        valley (scipy's least_squares too, the cost of a single fit). The Hessian's
        eigenvalues are taken in absolute value (saddle-free Newton): plain Newton
        is drawn to saddles, and two equal NSS humps are one by symmetry. Damping
        adds a multiple of diag(J'J), so a step that does not lower the error is
        refused and retried shorter, towards steepest descent.

        Bounds by projection: a hump on a bound whose gradient points out of it is
        held, and the step is clipped into the bounds. A curve stops when its step,
        its relative error reduction or its projected gradient falls under
        ``_TOL``. Only the curves still running are evaluated. Returns the humps (C,
        number of humps) and which curves converged."""
        low, high = (np.log(b) for b in self._hump_bounds())
        u = np.log(humps0)
        n_curves, n_humps = u.shape
        eye = np.eye(n_humps)
        damping = np.full(n_curves, 1e-6)
        converged = np.zeros(n_curves, dtype=bool)
        fd_step = 1e-6

        def evaluate(idx: np.ndarray, u_idx: np.ndarray):
            """Squared error, gradient J'r and diag(J'J) at log humps `u_idx`."""
            humps = np.exp(u_idx)
            residual, jac = self._projected_residual(
                t[idx], rates[idx], mask[idx], humps
            )
            jac = jac * humps[:, None, :]
            return (
                np.sum(residual**2, axis=1),
                np.einsum("cnh,cn->ch", jac, residual),
                np.einsum("cnh,cnh->ch", jac, jac),
            )

        def hessian(idx: np.ndarray, u_idx: np.ndarray, gradient: np.ndarray):
            """Forward difference of the gradient (stepping inwards at a bound),
            with its eigenvalues made positive: |eigenvalue|, at least 1e-10 of the
            largest."""
            columns = []
            for j in range(n_humps):
                h = np.where(u_idx[:, j] + fd_step > high[j], -fd_step, fd_step)
                moved = u_idx.copy()
                moved[:, j] += h
                columns.append((evaluate(idx, moved)[1] - gradient) / h[:, None])
            full = np.stack(columns, axis=-1)
            values, vectors = np.linalg.eigh(0.5 * (full + full.swapaxes(1, 2)))
            values = np.abs(values)
            values = np.maximum(values, 1e-10 * values.max(axis=1, keepdims=True))
            return (vectors * values[:, None, :]) @ vectors.swapaxes(1, 2)

        every = np.arange(n_curves)
        sse, gradient, scale = evaluate(every, u)
        curvature = hessian(every, u, gradient)
        for _ in range(_MAX_BATCH_ITERATIONS):
            idx = np.flatnonzero(~converged)
            if idx.size == 0:
                break
            u_idx, g = u[idx], gradient[idx]
            held = ((u_idx <= low) & (g > 0)) | ((u_idx >= high) & (g < 0))
            free = ~held
            system = curvature[idx] + damping[idx, None, None] * (
                scale[idx][:, :, None] * eye
            )
            system = (
                system * (free[:, :, None] & free[:, None, :]) + eye * held[:, :, None]
            )
            step = np.linalg.solve(system, (-g * free)[..., None])[..., 0]
            # Shorten the step to the bounds along its direction: clipping each hump
            # alone turns it, and can push a fit into another basin. A step blocked
            # at once (a free hump on its bound stepping out) is clipped instead.
            with np.errstate(divide="ignore", invalid="ignore"):
                room = np.where(
                    step > 0,
                    (high - u_idx) / step,
                    np.where(step < 0, (low - u_idx) / step, np.inf),
                )
            fraction = np.minimum(1.0, room.min(axis=1))
            fraction = np.where(fraction > 1e-3, fraction, 1.0)
            u_new = np.clip(u_idx + fraction[:, None] * step, low, high)
            sse_new, g_new, scale_new = evaluate(idx, u_new)
            better = sse_new < sse[idx]
            moved = np.max(np.abs(u_new - u_idx), axis=1)
            reduction = sse[idx] - sse_new
            projected_gradient = np.max(np.abs(g * free), axis=1)
            done = (
                (moved <= _TOL * (_TOL + np.max(np.abs(u_idx), axis=1)))
                | (better & (reduction <= _TOL * sse[idx]))
                | (projected_gradient <= _TOL * _TOL)
                | (damping[idx] > 1e16)
            )
            accepted = idx[better]
            u[accepted], sse[accepted] = u_new[better], sse_new[better]
            gradient[accepted], scale[accepted] = g_new[better], scale_new[better]
            damping[idx] = np.where(
                better, np.maximum(damping[idx] / 10, 1e-15), damping[idx] * 10
            )
            converged[idx[done]] = True
            moving = accepted[~converged[accepted]]
            if moving.size:
                curvature[moving] = hessian(moving, u[moving], gradient[moving])
        return np.exp(u), converged

    def _fit_global_humps(
        self,
        t: np.ndarray,
        rates: np.ndarray,
        mask: np.ndarray,
        start: np.ndarray,
    ) -> np.ndarray:
        """One set of humps minimising the summed projected squared error of every
        curve (each with its own betas): one bounded least squares on the stacked
        residuals, as :meth:`calibrate_from_curve` on one curve."""
        n_curves = t.shape[0]

        def shared(humps: np.ndarray) -> np.ndarray:
            return np.broadcast_to(humps, (n_curves, humps.size))

        def residuals(humps: np.ndarray) -> np.ndarray:
            return self._projected_residual(t, rates, mask, shared(humps))[0].ravel()

        def jacobian(humps: np.ndarray) -> np.ndarray:
            _, jac = self._projected_residual(t, rates, mask, shared(humps))
            return jac.reshape(-1, humps.size)

        return self._least_squares(residuals, jacobian, start, 0)

    # ── Batched bond calibration ──
    # S sets of bonds at once, padded to B bonds of F flows (see _stack_bond_sets):
    # a padded flow is 0 and a padded bond has market PV 0, so they add nothing.

    @classmethod
    def calibrate_many_from_bonds(
        cls,
        bond_sets: Sequence[tuple[date, list[tuple[CLBond, Rate]]]],
        initial_guess: Optional[list[float]] = None,
    ) -> tuple[list[Self], list[list[CalibrationError]]]:
        """Fit one model per set of bonds, as :meth:`calibrate_from_bonds` on each,
        in batched numpy.

        Each set is ``(calibration_date, [(bond, irr), ...])``; sets may differ in
        date, bond count and flows. The residuals, Jacobian and objective are those
        of :meth:`calibrate_from_bonds`. What differs is the start and the solver:

        * Start: by default the best point of the hump grid, with its betas, on a
          pseudo-curve of each bond's zero rate at its duration (the PV-weighted
          continuous zero rate of its flows at its own IRR), scored for every set
          in one batched solve. A single fit starts from fixed humps instead; with
          several local minima, as 10 noisy NSS bonds often have, the two can end
          in different ones.
        * Solver: a batched damped Gauss-Newton on all parameters (humps in log,
          each step at most ``_MAX_LOG_HUMP_STEP`` in log hump), every set with
          its own damping, stopping when it converges. A set still not converged
          after ``_MAX_BOND_ITERATIONS`` is refitted alone by
          :meth:`calibrate_from_bonds` from where it stopped.
        * Errors: the IRRs at the fitted PVs come from one batched Newton solve
          over every bond, in each quote's convention.

        Parameters
        ----------
        bond_sets : sequence of (date, list of (CLBond, Rate))
            Calibration date and quoted bonds of each fit.
        initial_guess : list of float, optional
            Parameters in :meth:`get_params` order to start every set from
            (humps clipped to their bounds), instead of the grid start.

        Returns
        -------
        tuple of (list of model, list of list of CalibrationError)
            One fitted model per set, in order, and its errors as
            :meth:`calibrate_from_bonds` returns them, in the order of its bonds.

        Raises
        ------
        ValueError
            If a set's fallback fit does not converge.
        """
        models: list[Self] = []
        errors: list[list[CalibrationError]] = []
        if not bond_sets:
            return models, errors
        model = cls()
        stacked = _stack_bond_sets(bond_sets)
        n_betas = model._n_betas
        low, high = model._hump_bounds()
        if initial_guess is not None:
            start = np.asarray(initial_guess, dtype=float)
            params = np.tile(start, (len(bond_sets), 1))
        else:
            params = model._bond_grid_start(stacked)
        params[:, n_betas:] = np.clip(params[:, n_betas:], low * 1.01, high * 0.99)
        params, converged = model._batched_bond_gauss_newton(stacked, params)
        for s in np.flatnonzero(~converged):
            calibration_date, bonds = bond_sets[int(s)]
            single = cls()
            single.calibrate_from_bonds(
                calibration_date, bonds, initial_guess=list(params[s])
            )
            params[s] = single.get_params()
        residual, flow_pvs = model._bond_pvs(stacked, np.arange(len(bond_sets)), params)
        model_pvs = residual + stacked.market_pvs
        shape = stacked.market_pvs.shape + (stacked.n_flows,)
        with np.errstate(invalid="ignore", divide="ignore"):
            durations = (flow_pvs * stacked.t).reshape(shape).sum(-1) / model_pvs
        model_irrs = _batched_irrs(stacked, model_pvs)
        for s, size in enumerate(stacked.sizes):
            models.append(cls(*params[s]))
            errors.append(
                [
                    CalibrationError(
                        term=float(durations[s, b]),
                        error_bps=1e4
                        * float(model_irrs[s, b] - stacked.quote_rates[s, b]),
                    )
                    for b in range(size)
                ]
            )
        return models, errors

    def _bond_pvs(
        self,
        stacked: "_StackedBonds",
        idx: np.ndarray,
        params: np.ndarray,
        jac: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Model PV minus market PV of every bond of the sets `idx` (s, B), and either
        each flow's model PV (s, B·F) or, with `jac`, the Jacobian of the residuals
        in the parameters (s, B, P): :meth:`calibrate_from_bonds`'s, batched."""
        n_betas = self._n_betas
        betas, humps = params[:, :n_betas], params[:, n_betas:]
        t, flows = stacked.t[idx], stacked.flows[idx]
        columns = [humps[:, [j]] for j in range(humps.shape[1])]
        if jac:
            design, derivatives = self._design_and_derivatives(t, columns)
        else:
            design, derivatives = self._calibration_design(t, columns), []
        flow_pvs = np.exp(-_times(design, betas) * t) * flows
        shape = (len(idx), stacked.market_pvs.shape[1], stacked.n_flows)
        residual = flow_pvs.reshape(shape).sum(-1) - stacked.market_pvs[idx]
        if not jac:
            return residual, flow_pvs
        rate_slopes = np.concatenate(
            [design, *(d @ betas[..., None] for d in derivatives)], axis=-1
        )
        jacobian = ((-flow_pvs * t)[..., None] * rate_slopes).reshape(*shape, -1)
        return residual, jacobian.sum(axis=2)

    def _bond_grid_start(self, stacked: "_StackedBonds") -> np.ndarray:
        """Each set's best grid humps and their betas on its bonds' pseudo-curve: a
        bond's PV-weighted continuous zero rate at its own IRR, sum(pv · -log df) /
        sum(pv · t), at its Macaulay duration at that IRR. Exact for a compounded or
        continuous Act/365 quote, whose zero rate is flat."""
        shape = stacked.market_pvs.shape + (stacked.n_flows,)
        dfs = _quote_dfs(stacked, stacked.quote_rates)
        flow_pvs = stacked.flows.reshape(shape) * dfs
        t = stacked.t.reshape(shape)
        weight = (flow_pvs * t).sum(-1)
        present = weight > 0
        safe = np.where(present, weight, 1.0)
        duration = np.where(
            present, weight / np.where(present, flow_pvs.sum(-1), 1.0), 1.0
        )
        rates = np.where(present, (flow_pvs * -np.log(dfs)).sum(-1) / safe, 0.0)
        mask = present.astype(float)
        low, high = self._hump_bounds()
        grid = _start_grid(low, high)
        sse = self._batched_grid_sse(duration, rates, mask, grid)
        humps = grid[np.argmin(sse, axis=0)]
        design = self._calibration_design(
            duration, [humps[:, [j]] for j in range(humps.shape[1])]
        )
        betas, _ = _projection(design, rates, mask)
        return np.concatenate([betas, humps], axis=1)

    def _batched_bond_gauss_newton(
        self, stacked: "_StackedBonds", params0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each set's parameters minimising its squared PV errors, humps in their
        bounds: damped Gauss-Newton on v = [betas, log humps], every set with its
        own damping (a multiple of diag(J'J)). A step moves a log hump by at most
        ``_MAX_LOG_HUMP_STEP``: an undamped first step can throw the first hump onto
        its lower bound, where huge opposite slope and curvature betas fit worse.
        Bounds by projection, as :meth:`_batched_newton`; the same stopping tests.
        Returns the parameters (S, P) and which sets converged."""
        n_betas = self._n_betas
        low, high = (np.log(b) for b in self._hump_bounds())
        lower = np.concatenate((np.full(n_betas, -np.inf), low))
        upper = np.concatenate((np.full(n_betas, np.inf), high))
        v = params0.copy()
        v[:, n_betas:] = np.log(v[:, n_betas:])
        n_sets, n_params = v.shape
        eye = np.eye(n_params)
        damping = np.full(n_sets, 1e-3)
        converged = np.zeros(n_sets, dtype=bool)

        def to_params(v_idx: np.ndarray) -> np.ndarray:
            params = v_idx.copy()
            params[:, n_betas:] = np.exp(v_idx[:, n_betas:])
            return params

        def evaluate(idx: np.ndarray, v_idx: np.ndarray):
            params = to_params(v_idx)
            residual, jac = self._bond_pvs(stacked, idx, params, jac=True)
            jac[..., n_betas:] *= params[:, None, n_betas:]
            return (
                np.sum(residual**2, axis=1),
                np.einsum("sbp,sb->sp", jac, residual),
                jac.swapaxes(1, 2) @ jac,
            )

        sse, gradient, normal = evaluate(np.arange(n_sets), v)
        for _ in range(_MAX_BOND_ITERATIONS):
            idx = np.flatnonzero(~converged)
            if idx.size == 0:
                break
            v_idx, g = v[idx], gradient[idx]
            held = ((v_idx <= lower) & (g > 0)) | ((v_idx >= upper) & (g < 0))
            free = ~held
            scale = np.diagonal(normal[idx], axis1=1, axis2=2)
            system = normal[idx] + damping[idx, None, None] * (scale[:, :, None] * eye)
            system = (
                system * (free[:, :, None] & free[:, None, :]) + eye * held[:, :, None]
            )
            step = np.linalg.solve(system, (-g * free)[..., None])[..., 0]
            largest = np.max(np.abs(step[:, n_betas:]), axis=1)
            step *= np.minimum(1.0, _MAX_LOG_HUMP_STEP / np.maximum(largest, 1e-300))[
                :, None
            ]
            v_new = np.clip(v_idx + step, lower, upper)
            sse_new, g_new, normal_new = evaluate(idx, v_new)
            better = sse_new < sse[idx]
            moved = np.max(np.abs(v_new - v_idx), axis=1)
            reduction = sse[idx] - sse_new
            projected_gradient = np.max(np.abs(g * free), axis=1)
            done = (
                (moved <= _TOL * (_TOL + np.max(np.abs(v_idx), axis=1)))
                | (better & (reduction <= _TOL * sse[idx]))
                | (projected_gradient <= _TOL * _TOL)
                | (damping[idx] > 1e16)
            )
            accepted = idx[better]
            v[accepted], sse[accepted] = v_new[better], sse_new[better]
            gradient[accepted], normal[accepted] = g_new[better], normal_new[better]
            damping[idx] = np.where(
                better, np.maximum(damping[idx] / 3, 1e-15), damping[idx] * 4
            )
            converged[idx[done]] = True
        return to_params(v), converged


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
