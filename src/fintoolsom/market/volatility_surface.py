"""FX volatility surface on a days-to-maturity by put-delta grid.

Conventions, shared with the option pricer (``derivatives.calculator._options``):

* time is ACT/365 from the surface date: ``T = days / 365``;
* ``log_moneyness`` is ``k = ln(K/F)``, with ``F`` the outright forward;
* deltas are spot, non-premium-adjusted put deltas (``-df_q * N(-d1)``);
* ``"atm"`` is the ATM forward, ``k = 0``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
import itertools
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
from scipy.optimize import LinearConstraint, least_squares, minimize
from scipy.special import ndtri

from ..rates import Curve, LegacyCurve

DAYS_IN_YEAR = 365
"""Day base of the surface time ``T = days / 365`` (ACT/365, as in option pricing)."""


class VolatilitySurfaceInterpolationMethod(Enum):
    """How a :class:`VolatilitySurface` interpolates between its quotes.

    The ``Double*`` methods and ``SABR`` interpolate in two steps (see
    :class:`DoubleInterpolationModel`): total variance in time at constant delta,
    then a smile in log-moneyness. They differ only in the smile. ``eSSVI`` fits
    one global surface.
    """

    DoubleLinear = "linear"
    DoubleMonotoneCubic = "monotone-cubic"
    SABR = "sabr"
    eSSVI = "essvi"


@dataclass
class VolatilitySurfaceInterpolationModel(ABC):
    """Surface model over a grid of `days` (rows) by put `deltas` (columns).

    `days` is sorted ascending; `vols` and `log_moneyness` (the pillar
    ``ln(K/F)``, see :func:`_log_moneyness_from_delta`) have shape
    ``(len(days), len(deltas))``; `foreign_log_dfs` is ``ln df_q`` at each pillar
    tenor. See :class:`VolatilitySurface` for the meaning of each input. No curve
    is held: the pillars are all a model needs.
    """

    days: np.ndarray
    deltas: tuple[Any, ...]
    vols: np.ndarray = field(repr=False)
    log_moneyness: np.ndarray = field(repr=False)
    foreign_log_dfs: np.ndarray = field(repr=False)

    def __post_init__(self):
        pass

    @abstractmethod
    def interpolate_surface(
        self, log_moneyness: float | np.ndarray, days: float
    ) -> float | np.ndarray:
        pass


@dataclass
class DoubleInterpolationModel(VolatilitySurfaceInterpolationModel, ABC):
    """Two-step interpolation: time first, then the smile.

    For a query at ``T = days / 365``:

    1. **Time**, at each delta column: total variance ``w = vol^2 * T`` linear in
       ``T`` between the bracketing tenors, so the forward variance is constant
       between pillars and ``w`` never decreases where the quotes do not. Before
       the first tenor and after the last, the vol is flat.
    2. **Strikes**: each column's log-moneyness is recomputed at ``T`` from its
       delta and the interpolated vol (:func:`_log_moneyness_from_delta`), with
       ``ln df_q`` linear in ``T`` (exact for a flat foreign rate), anchored at
       ``ln df_q(0) = 0`` and extended past the last tenor at its last slope.
    3. **Smile**: :meth:`_smile` through the ``(k, vol)`` nodes of step 2, read at
       the queried log-moneyness.

    Smiles are cached by ``days``.
    """

    _min_deltas = 1
    _smiles: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if len(self.deltas) < self._min_deltas:
            raise ValueError(
                f"{type(self).__name__} needs at least {self._min_deltas} delta "
                f"columns; got {len(self.deltas)}."
            )
        self._year_fractions = self.days / DAYS_IN_YEAR
        self._total_variances = self.vols**2 * self._year_fractions[:, None]

    def interpolate_surface(
        self, log_moneyness: float | np.ndarray, days: float
    ) -> float | np.ndarray:
        smile = self._smiles.get(days)
        if smile is None:
            k_nodes, vol_nodes = self.get_smile_nodes(days)
            smile = self._smile(k_nodes, vol_nodes, days / DAYS_IN_YEAR)
            self._smiles[days] = smile
        vol = smile(np.asarray(log_moneyness, dtype=float))
        return float(vol) if np.ndim(vol) == 0 else vol

    def get_smile_t(self, days: float) -> np.ndarray:
        """Vol of each delta column at `days` (step 1), in column order."""
        t = self._year_fraction(days)
        ts = self._year_fractions
        if t <= ts[0]:
            return self.vols[0].copy()
        if t >= ts[-1]:
            return self.vols[-1].copy()
        i = int(np.searchsorted(ts, t, side="right")) - 1
        weight = (t - ts[i]) / (ts[i + 1] - ts[i])
        w = self._total_variances[i] + weight * (
            self._total_variances[i + 1] - self._total_variances[i]
        )
        return np.sqrt(w / t)

    def get_smile_log_moneynesses(self, days: float) -> np.ndarray:
        """Log-moneyness of each delta column at `days` (step 2), in column order."""
        t = self._year_fraction(days)
        vols = self.get_smile_t(days)
        df_q = np.exp(self._foreign_log_df(t))
        return _log_moneyness_from_delta(
            self.deltas, vols[None, :], np.array([t]), np.array([df_q]), days
        )[0]

    def get_smile_nodes(self, days: float) -> tuple[np.ndarray, np.ndarray]:
        """``(k, vol)`` smile nodes at `days`, sorted by ``k``.

        Raises
        ------
        ValueError
            If two columns land on the same log-moneyness.
        """
        k = self.get_smile_log_moneynesses(days)
        vols = self.get_smile_t(days)
        order = np.argsort(k, kind="stable")
        k, vols = k[order], vols[order]
        if (np.diff(k) <= 0).any():
            raise ValueError(
                f"Smile nodes at {days} days are not distinct in log-moneyness: "
                f"{k.tolist()}."
            )
        return k, vols

    @abstractmethod
    def _smile(
        self, k_nodes: np.ndarray, vol_nodes: np.ndarray, t: float
    ) -> Callable[[np.ndarray], float | np.ndarray]:
        """Vol as a function of log-moneyness through the nodes, at time `t`."""

    def _foreign_log_df(self, t: float) -> float:
        ts = np.concatenate(([0.0], self._year_fractions))
        log_dfs = np.concatenate(([0.0], self.foreign_log_dfs))
        if t <= ts[-1]:
            return float(np.interp(t, ts, log_dfs))
        slope = (log_dfs[-1] - log_dfs[-2]) / (ts[-1] - ts[-2])
        return float(log_dfs[-1] + slope * (t - ts[-1]))

    @staticmethod
    def _year_fraction(days: float) -> float:
        if days <= 0:
            raise ValueError(f"days must be positive; got {days}.")
        return days / DAYS_IN_YEAR


class DoubleInterpolationLinear(DoubleInterpolationModel):
    """Two-step model with a piecewise-linear smile in vol, flat beyond the nodes."""

    def _smile(self, k_nodes, vol_nodes, t):
        return lambda k: np.interp(k, k_nodes, vol_nodes)


class DoubleInterpolationMonotoneCubic(DoubleInterpolationModel):
    """Two-step model with a monotone cubic (PCHIP) smile, flat beyond the nodes.

    PCHIP is C1 and does not overshoot: between two nodes the vol stays within
    their values, unlike a natural cubic spline.
    """

    _min_deltas = 2

    def _smile(self, k_nodes, vol_nodes, t):
        pchip = PchipInterpolator(k_nodes, vol_nodes, extrapolate=False)
        return lambda k: pchip(np.clip(k, k_nodes[0], k_nodes[-1]))


@dataclass
class SABRParameters:
    """SABR parameters with ``beta = 1`` (lognormal backbone).

    ``alpha`` is the vol level, ``rho`` the spot-vol correlation (skew) and
    ``nu`` the vol of vol (curvature).
    """

    alpha: float
    rho: float
    nu: float


def sabr_volatility(
    log_moneyness: float | np.ndarray, t: float, params: SABRParameters
) -> float | np.ndarray:
    """Black implied vol of SABR with ``beta = 1`` (Hagan et al., 2002).

    With ``k = ln(K/F)`` and ``z = (nu / alpha) * ln(F/K)``::

        vol = alpha * z / x(z) * (1 + (rho*nu*alpha/4 + (2 - 3*rho^2)*nu^2/24) * T)
        x(z) = ln((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))

    With ``beta = 1`` the formula depends on ``K/F`` only, not on the forward
    level, so no spot is needed. ``z / x(z) -> 1 - rho*z/2`` as ``z -> 0``.
    """
    alpha, rho, nu = params.alpha, params.rho, params.nu
    z = nu / alpha * -np.asarray(log_moneyness, dtype=float)
    small = np.abs(z) < 1e-7
    z_safe = np.where(small, 1.0, z)
    x = np.log((np.sqrt(1 - 2 * rho * z_safe + z_safe**2) + z_safe - rho) / (1 - rho))
    ratio = np.where(small, 1 - rho * z / 2, z_safe / x)
    correction = 1 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * t
    return alpha * ratio * correction


def calibrate_sabr(
    k_nodes: np.ndarray, vol_nodes: np.ndarray, t: float
) -> SABRParameters:
    """Least-squares SABR (``beta = 1``) fit of one smile at time `t`.

    Three nodes are fitted exactly when the smile is reachable; more nodes are
    fitted in the least-squares sense. Bounds: ``alpha`` in ``(0, 5]``, ``rho``
    in ``[-0.999, 0.999]``, ``nu`` in ``(0, 10]``.

    Raises
    ------
    ValueError
        If the fit does not converge.
    """
    atm = int(np.argmin(np.abs(k_nodes)))
    x0 = [float(vol_nodes[atm]), 0.0, 0.5]

    def residuals(x):
        return sabr_volatility(k_nodes, t, SABRParameters(*x)) - vol_nodes

    result = least_squares(
        residuals,
        x0,
        bounds=([1e-6, -0.999, 1e-6], [5.0, 0.999, 10.0]),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )
    if not result.success:
        raise ValueError(f"SABR calibration failed at T={t:.6f}: {result.message}")
    return SABRParameters(*result.x)


class DoubleInterpolationSABR(DoubleInterpolationModel):
    """Two-step model with a SABR (``beta = 1``) smile fitted at each tenor queried.

    Steps 1 and 2 are those of :class:`DoubleInterpolationModel`; the smile is
    :func:`calibrate_sabr` on the nodes at the queried tenor, so the quotes are
    matched at every tenor and the wings follow SABR instead of staying flat.
    Needs at least three delta columns (three parameters).
    """

    _min_deltas = 3

    def _smile(self, k_nodes, vol_nodes, t):
        params = calibrate_sabr(k_nodes, vol_nodes, t)
        return lambda k: sabr_volatility(k, t, params)

    def get_sabr_parameters(self, days: float) -> SABRParameters:
        """SABR parameters fitted at `days`."""
        k_nodes, vol_nodes = self.get_smile_nodes(days)
        return calibrate_sabr(k_nodes, vol_nodes, days / DAYS_IN_YEAR)


class eSSVI(VolatilitySurfaceInterpolationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.theta_params = []
        self.rho_params = []
        self.phi_params = []
        self._all_params = [self.theta_params, self.rho_params, self.phi_params]
        self.do_kwargs(**kwargs)

        self.calibrate()

    def do_kwargs(self, **kwargs):
        self.parameters_type = kwargs.get("parameter_type", "parametric")
        if self.parameters_type == "functional_form":
            self._function_type = kwargs.get("function_type", "exponential")
            if self._function_type == "exponential":
                self.parameter_formula = self.exponential_formula
                for i in range(3):
                    for _ in range(2):
                        self._all_params[i].append(None)
            elif self._function_type == "polynomial":
                self.parameter_formula = self.polynomial_formula
                self.polynomial_order = kwargs.get("polynomial_order", 2)
                for i in range(3):
                    for _ in range(1 + self.polynomial_order):
                        self._all_params[i].append(None)
            else:
                raise ValueError(
                    f"function_type value {self._function_type} not valid."
                )
        elif self.parameters_type == "parametric":
            self.parameter_formula = self.parametric_formula
            parameter_interpolation_method = kwargs.get(
                "parameters_interpolation_method", "linear"
            )
            if parameter_interpolation_method == "linear":
                self._parameter_interpolator = interp1d
            elif parameter_interpolation_method == "cubic-spline":
                self._parameter_interpolator = CubicSpline
            else:
                raise ValueError(
                    f'parameters_interpolation_method value {parameter_interpolation_method} not valid. Try "linear" or "cubic-spline".'
                )
            for i in range(3):
                for _ in range(len(self.days)):
                    self._all_params[i].append(None)
        else:
            raise ValueError(f"parameter_type value {self.parameters_type} not valid.")

    def exponential_formula(self, t: float | np.ndarray, *params) -> float | np.ndarray:
        return params[0] * np.exp(params[1] * t)

    def polynomial_formula(self, t: float | np.ndarray, *params) -> float | np.ndarray:
        ixs = np.arange(len(params))
        return np.sum(np.array(params)[ixs] * t.reshape(-1, 1) ** ixs, axis=1)

    def parametric_formula(self, t: float | np.ndarray, *params) -> float | np.ndarray:
        return self._parameter_interpolator(self.days / DAYS_IN_YEAR, params)(t)

    def interpolate_surface(
        self, log_moneyness: float | np.ndarray, days: int | np.ndarray
    ) -> float | np.ndarray:
        flatten_params = list(itertools.chain(*self._all_params))
        total_variance = self._get_total_variance(log_moneyness, days, *flatten_params)
        vol = np.sqrt(total_variance / (days / DAYS_IN_YEAR))
        return vol

    def _get_total_variance(
        self,
        log_moneyness: float | np.ndarray,
        days: int | np.ndarray,
        *flatten_params: list[float],
        validate_shapes: bool = True,
    ) -> float | np.ndarray:
        t = days / DAYS_IN_YEAR
        if validate_shapes:
            # Ensure log_moneyness and days are numpy arrays for easier manipulation
            if isinstance(log_moneyness, float):
                log_moneyness = np.array([log_moneyness])
            if isinstance(t, float):
                t = np.array([t])

            # Check if both log_moneyness and days are arrays
            if isinstance(log_moneyness, np.ndarray) and isinstance(t, np.ndarray):
                if log_moneyness.ndim == 1 and t.ndim == 1:
                    # Both are simple arrays, no need to reshape
                    pass
                elif log_moneyness.ndim == 1 and t.ndim == 2:
                    if t.shape[0] == len(log_moneyness) and t.shape[1] == 1:
                        # log_moneyness is an array and days is a matrix of size len(log_moneyness)x1
                        pass
                    else:
                        raise ValueError(
                            "When both log_moneyness and days are arrays, days must be of shape (len(log_moneyness), 1)."
                        )
                else:
                    raise ValueError("Invalid dimensions for log_moneyness and days.")

        params_amount = int(len(flatten_params) / 3)

        theta = self.parameter_formula(t, *flatten_params[0:params_amount])
        rho = self.parameter_formula(
            t, *flatten_params[params_amount : 2 * params_amount]
        )
        phi = self.parameter_formula(
            t, *flatten_params[2 * params_amount : 3 * params_amount]
        )

        phi_k = phi * log_moneyness
        total_variance = (theta / 2) * (
            1 + rho * phi_k + np.sqrt((phi_k + rho) ** 2 + 1 - rho**2)
        )

        return total_variance

    def calibrate(self):
        days = self.days[:, None]
        mkt_vol_surface_matrix = self.vols
        mkt_total_variances_matrix = (days / DAYS_IN_YEAR) * mkt_vol_surface_matrix**2

        # Initial guesses for theta, rho and phi. Independently of parameter form. theta represents level, rho correlation (positive for FX) and phi skewness (slightly positive for FX)
        # Bounds:.
        #  Thetas must grow on time. Exponential => theta_0 & theta_1 > 0. Polynomial => first 2 orders are greater than 0. Parametric => theta and rho are greater than 0.
        #  Set limit to negative side of rho and phi.

        # Constraints:
        #  Theta function must grow on time. That is achieved with bounds in exponential and polynomial form (first and second parameter are positive).
        #  For parametric a constraint is needed. That is, theta_0 < theta_1 < theta_n.

        initial_params = self._all_params.copy()  # Until now params are all None. Lets set average level for theta, 0.3 for rho and 0.3 to skew as starting point. Rest params no related to level set to 0.
        constraints = None
        if self.parameters_type == "parametric":
            average_vols_per_tenor = np.mean(mkt_total_variances_matrix, axis=1)
            bounds = []
            for i in range(mkt_total_variances_matrix.shape[0]):
                bounds.append((0, 1))  # thetas are positive
                initial_params[0][i] = average_vols_per_tenor[i]  # Thetas
                initial_params[1][i] = 0.3  # rhos
                initial_params[2][i] = 0.3  # phis
            for i in range(mkt_total_variances_matrix.shape[0]):
                bounds.append((-0.3, 1))  # rhos are positively biased
            for i in range(mkt_total_variances_matrix.shape[0]):
                bounds.append((0, 1))  # phis are positive

            m = mkt_total_variances_matrix.shape[0] - 1
            A = np.zeros((m - 1, 3 * mkt_total_variances_matrix.shape[0]))
            for i in range(m - 1):
                A[i, i] = -1
                A[i, i + 1] = 1

            # Create a LinearConstraint object
            constraints = LinearConstraint(A, 0, np.inf)
        else:
            if self._function_type == "exponential":
                initial_params = [
                    [mkt_total_variances_matrix.mean(), 0],
                    [0.3, 0],
                    [0.3, 0],
                ]
                # Theta level and slope is positive. Rho level is positively biased. phi is positive
                bounds = [(0, 1), (0, 1), (-0.3, 1), (-1, 1), (0, 1), (-1, 1)]
            if self._function_type == "polynomial":
                initial_params[0][0] = mkt_total_variances_matrix.mean()  # Thetas
                initial_params[1][0] = 0.3  # rhos
                initial_params[2][0] = 0.3  # phis
                bounds = [
                    (0, None),
                    (0, None),  # theta level and slope is positive
                ]
                theta_bounds_to_add = 1 + self.polynomial_order - 2
                for _ in range(theta_bounds_to_add):
                    bounds.append((-1, 1))
                bounds.append((-0.3, 1))  # rho level is positively biased
                rho_bounds_to_add = 1 + self.polynomial_order - 1
                for _ in range(rho_bounds_to_add):
                    bounds.append((-1, 1))

                bounds.append((0, 1))  # phi is positive
                phi_bounds_to_add = 1 + self.polynomial_order - 1
                for _ in range(phi_bounds_to_add):
                    bounds.append((-1, 1))
                for i in range(1, 1 + self.polynomial_order):
                    initial_params[0][i] = 0
                    initial_params[1][i] = 0
                    initial_params[2][i] = 0

        flatten_initial_params = list(itertools.chain(*initial_params))

        if len(bounds) != len(flatten_initial_params):
            raise ValueError(
                f"Length of flatten params is {len(flatten_initial_params)}. But set bounds for {len(bounds)} params."
            )
        # bounds = [(-1,1)] * len(flatten_initial_params)
        log_moneyness_np = self.log_moneyness

        def objective_func(flatten_params):
            model_total_variances = self._get_total_variance(
                log_moneyness_np, days, *flatten_params, validate_shapes=False
            )
            return np.sum((model_total_variances - mkt_total_variances_matrix) ** 2)

        initial_model_total_variances = self._get_total_variance(
            log_moneyness_np, days, *flatten_initial_params, validate_shapes=False
        )
        print(
            f"\tStarting calibration... Initial error: {np.sum((initial_model_total_variances - mkt_total_variances_matrix) ** 2)}"
        )
        result = minimize(
            objective_func,
            flatten_initial_params,
            bounds=bounds,
            constraints=constraints,
            method="trust-constr",
        )

        if result.success:
            print(f"\tCalibration worked. Error achieved: {result.fun}")
            fitted_params = result.x
            params_amount = int(len(fitted_params) / 3)
            # print("Fitted parameters:", fitted_params)
            self._all_params = [
                fitted_params[0:params_amount],
                fitted_params[params_amount : 2 * params_amount],
                fitted_params[2 * params_amount : 3 * params_amount],
            ]
        else:
            raise ValueError("Fitting failed")


def _validated_grid(
    days: ArrayLike, deltas: Sequence, vols: ArrayLike
) -> tuple[np.ndarray, tuple, np.ndarray]:
    """`days`, `deltas` and `vols` checked, with the rows sorted by `days`.

    Raises
    ------
    TypeError
        If `days` are not integers.
    ValueError
        If `days` are not positive or repeat, a delta is neither ``"atm"`` nor a
        number in ``[1, 100)``, or `vols` is not a finite, positive
        ``(len(days), len(deltas))`` matrix.
    """
    days_arr = np.asarray(days)
    if days_arr.ndim != 1 or len(days_arr) == 0:
        raise ValueError("days must be a non-empty 1-D sequence of days to maturity.")
    if not np.issubdtype(days_arr.dtype, np.integer):
        raise TypeError("days must be integers, representing days to maturity.")
    days_arr = days_arr.astype(np.int64)
    if (days_arr <= 0).any():
        raise ValueError("days must be positive.")
    if len(np.unique(days_arr)) != len(days_arr):
        raise ValueError("days must not repeat.")

    deltas_t = tuple(deltas)
    for delta in deltas_t:
        if isinstance(delta, str) and delta == "atm":
            continue
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float, np.number))
            or not 1 <= delta < 100
        ):
            raise ValueError(
                "Each delta must be a put delta in percent, in [1, 100) "
                f'(e.g. 25, not 0.25), or "atm"; got {delta!r}.'
            )

    vols_arr = np.array(vols, dtype=float)
    if vols_arr.shape != (len(days_arr), len(deltas_t)):
        raise ValueError(
            f"vols must have shape (len(days), len(deltas)) = "
            f"{(len(days_arr), len(deltas_t))}; got {vols_arr.shape}."
        )
    if not np.isfinite(vols_arr).all() or (vols_arr <= 0).any():
        raise ValueError("vols must be finite and positive. Ex: 14% is 0.14")

    order = np.argsort(days_arr, kind="stable")
    return days_arr[order], deltas_t, vols_arr[order]


def _foreign_dfs(days: np.ndarray, foreign_curve: Curve | LegacyCurve) -> np.ndarray:
    """Foreign discount factor at ``foreign_curve.curve_date + days``."""
    t0 = foreign_curve.curve_date
    return np.array([foreign_curve.get_df(t0 + timedelta(days=int(d))) for d in days])


def _log_moneyness_from_delta(
    deltas: tuple,
    vols: np.ndarray,
    year_fractions: np.ndarray,
    foreign_dfs: np.ndarray,
    days: ArrayLike,
) -> np.ndarray:
    """Log-moneyness ``k = ln(K/F)`` of each (tenor, put delta) node.

    Inverting the spot, non-premium-adjusted put delta
    ``-df_q * N(-d1) = -delta`` and using ``ln(F/S) = (r - q) T``::

        k = N^-1(delta / df_q) * vol * sqrt(T) + vol^2 * T / 2

    Spot and the domestic rate cancel. An ``"atm"`` column is ``k = 0`` (ATM
    forward). `vols` has one row per entry of `year_fractions` and `foreign_dfs`;
    `days` is only used in error messages.

    Raises
    ------
    ValueError
        If a put delta reaches the foreign discount factor (``delta / df_q >= 1``):
        no strike has that spot delta at that tenor.
    """
    days_arr = np.atleast_1d(np.asarray(days))
    log_moneyness = np.zeros(vols.shape)
    for column, delta in enumerate(deltas):
        if isinstance(delta, str):  # "atm"
            continue
        ratio = float(delta) / 100 / foreign_dfs
        if (ratio >= 1).any():
            row = int(np.argmax(ratio >= 1))
            raise ValueError(
                f"A {delta}-delta put is unreachable at {days_arr[row]} days: spot "
                f"put delta is bounded by the foreign discount factor "
                f"{foreign_dfs[row]:.4f}."
            )
        vol = vols[:, column]
        log_moneyness[:, column] = (
            ndtri(ratio) * vol * np.sqrt(year_fractions) + vol**2 * year_fractions / 2
        )
    return log_moneyness


_MODELS = {
    VolatilitySurfaceInterpolationMethod.DoubleLinear: DoubleInterpolationLinear,
    VolatilitySurfaceInterpolationMethod.DoubleMonotoneCubic: DoubleInterpolationMonotoneCubic,
    VolatilitySurfaceInterpolationMethod.SABR: DoubleInterpolationSABR,
}


class VolatilitySurface:
    """FX volatility surface quoted on a days-to-maturity by put-delta grid.

    Parameters
    ----------
    days : array-like of int
        Days to maturity of each row, from ``foreign_curve.curve_date``; sorted
        ascending internally.
    deltas : sequence of int, float or "atm"
        Column labels: put deltas in percent (``25`` is the 25-delta put, not
        ``0.25``) or ``"atm"`` (ATM forward, log-moneyness 0). Put deltas are
        spot, non-premium-adjusted: ``75`` is the 75-delta put, which is the
        25-delta call only when the foreign discount factor is 1.
    vols : array-like of float
        Volatilities, shape ``(len(days), len(deltas))``: one row per entry of
        `days`, one column per entry of `deltas`; 14% is ``0.14``.
    foreign_curve : Curve
        Discount curve of the base (foreign) currency, used once, at
        construction, to turn each spot put delta into a log-moneyness (see
        :func:`_log_moneyness_from_delta`). Neither spot nor the domestic curve
        is needed, and no curve is kept.
    interpolation_method : VolatilitySurfaceInterpolationMethod, default SABR
        Surface interpolation model; see :class:`DoubleInterpolationModel` for the
        two-step methods.
    **kwargs
        Passed to :class:`eSSVI` when it is the interpolation method.

    Raises
    ------
    TypeError, ValueError
        If the grid is invalid (see :func:`_validated_grid`), a put delta is
        unreachable at its tenor (see :func:`_log_moneyness_from_delta`), or the
        method needs more delta columns.
    NotImplementedError
        For an unsupported `interpolation_method`.

    Examples
    --------
    Two tenors (1M and 3M) by three quotes (25-delta put, ATM, 75-delta put):

    >>> days = [30, 90]
    >>> deltas = [25, "atm", 75]
    >>> vols = [
    ...     [0.125, 0.120, 0.118],  # 30 days:  25P, ATM, 75P
    ...     [0.130, 0.124, 0.121],  # 90 days:  25P, ATM, 75P
    ... ]
    >>> surface = VolatilitySurface(days, deltas, vols, usd_curve)  # doctest: +SKIP
    >>> surface.get_volatility(log_moneyness=0.01, days=60)  # doctest: +SKIP
    """

    def __init__(
        self,
        days: ArrayLike,
        deltas: Sequence,
        vols: ArrayLike,
        foreign_curve: Curve | LegacyCurve,
        interpolation_method: VolatilitySurfaceInterpolationMethod = VolatilitySurfaceInterpolationMethod.SABR,
        **kwargs,
    ):
        self.days, self.deltas, self.vols = _validated_grid(days, deltas, vols)
        foreign_dfs = _foreign_dfs(self.days, foreign_curve)
        self.log_moneyness = _log_moneyness_from_delta(
            self.deltas,
            self.vols,
            self.days / DAYS_IN_YEAR,
            foreign_dfs,
            self.days,
        )
        self.interpolation_method = interpolation_method

        grid = (
            self.days,
            self.deltas,
            self.vols,
            self.log_moneyness,
            np.log(foreign_dfs),
        )
        if interpolation_method == VolatilitySurfaceInterpolationMethod.eSSVI:
            self._interpolation_model = eSSVI(*grid, **kwargs)
        elif interpolation_method in _MODELS:
            self._interpolation_model = _MODELS[interpolation_method](*grid)
        else:
            raise NotImplementedError(
                f"Interpolation method: {interpolation_method} not implemented."
            )

    def get_volatility(
        self, log_moneyness: float | np.ndarray, days: float
    ) -> float | np.ndarray:
        """Vol at log-moneyness ``ln(K/F)`` and `days` from the surface date."""
        return self._interpolation_model.interpolate_surface(log_moneyness, days)
