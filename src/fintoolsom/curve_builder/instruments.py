"""Curve bootstrap from fixed-income instruments: :func:`build_curve_from_instruments`.

Bootstraps a :class:`~fintoolsom.rates.PillarCurve` from Chilean bonds
(:class:`~fintoolsom.fixedIncome.CLBond`) and deposits
(:class:`~fintoolsom.fixedIncome.Deposit`) quoted by their :class:`~fintoolsom.rates.Rate`.
Each instrument pins one pillar at its maturity date.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import root

from ..fixedIncome.CLBonds import CLBond
from ..fixedIncome.Deposits import Deposit
from ..rates import InterpolationMethod, PillarCurve, Rate
from ..rates.Curve import DAY_BASE
from ..rates.PillarCurve import MIN_KNOTS
from ._errors import InsufficientQuotesError, RepricingError

# Default repricing tolerance on base-100 instrument PVs.
_INSTRUMENT_REPRICE_TOL = 1e-6
_NEWTON_MAX_ITER = 50
_NEWTON_TOL = 1e-13
_MAX_ABS_FORWARD = 5.0


@dataclass(frozen=True, slots=True)
class _ParsedInstrument:
    """One bond or deposit normalised to its remaining flows and base-100 target PV."""

    maturity: date
    end_dates: tuple[date, ...]
    days: np.ndarray
    flows: np.ndarray
    target_pv: float
    seed_rate: float


def _parse_instrument(
    curve_date: date,
    instrument: CLBond | Deposit,
    irr: Rate,
) -> _ParsedInstrument:
    if isinstance(instrument, CLBond):
        end_dates = instrument.coupons.get_remaining_end_dates(curve_date)
        if not end_dates:
            raise ValueError(
                f"Bond has no flows after the curve date {curve_date}: "
                f"it matured on {instrument.get_maturity_date()}."
            )
        flows = np.asarray(
            instrument.coupons.get_remaining_flows(curve_date), dtype=float
        )
        days = np.array([(d - curve_date).days for d in end_dates], dtype=np.int64)
        target_pv = float(instrument.get_present_value(curve_date, irr))
        return _ParsedInstrument(
            maturity=end_dates[-1],
            end_dates=tuple(end_dates),
            days=days,
            flows=flows,
            target_pv=target_pv,
            seed_rate=float(irr.rate),
        )
    if isinstance(instrument, Deposit):
        payment_date = instrument.payment_date
        if payment_date <= curve_date:
            raise ValueError(
                f"Deposit {instrument.nemo!r} pays on {payment_date}, on or before the "
                f"curve date {curve_date}."
            )
        days = np.array([(payment_date - curve_date).days], dtype=np.int64)
        target_pv = 100.0 * float(irr.get_discount_factor(curve_date, payment_date))
        return _ParsedInstrument(
            maturity=payment_date,
            end_dates=(payment_date,),
            days=days,
            flows=np.array([100.0]),
            target_pv=target_pv,
            seed_rate=float(irr.rate),
        )
    raise TypeError(f"Expected CLBond or Deposit, got {type(instrument).__name__}.")


def _solve_segment_forward(
    dt: np.ndarray,
    flows: np.ndarray,
    df_prev: float,
    remaining_pv: float,
    x0: float,
) -> float:
    """Solve ``sum(flows * df_prev * exp(-f * dt)) == remaining_pv`` for ``f``.

    With positive cash flows and ``dt > 0``, the left-hand side is strictly decreasing
    and strictly convex in ``f``, so Newton-Raphson converges quadratically.
    """
    if remaining_pv <= 0.0:
        raise ValueError(
            "Inconsistent instrument quotes: remaining PV on the final segment is "
            f"non-positive ({remaining_pv:.6g})."
        )
    # Closed form when there is a single cash flow in the segment (e.g. Deposit or zero-coupon).
    if dt.size == 1:
        ratio = remaining_pv / (flows[0] * df_prev)
        if ratio <= 0.0:
            raise ValueError(
                "Inconsistent instrument quotes: implied discount factor is non-positive."
            )
        return float(-math.log(ratio) / dt[0])

    f = float(np.clip(x0, -_MAX_ABS_FORWARD, _MAX_ABS_FORWARD))
    for _ in range(_NEWTON_MAX_ITER):
        scaled = flows * df_prev * np.exp(-f * dt)
        val = float(scaled.sum()) - remaining_pv
        if abs(val) <= _NEWTON_TOL:
            return f
        deriv = float(-(scaled * dt).sum())
        if deriv == 0.0 or not math.isfinite(deriv):
            break
        step = val / deriv
        f_next = f - step
        # Guard against overflow in exp(-f * dt) on wild initial guesses.
        if not math.isfinite(f_next) or abs(f_next) > _MAX_ABS_FORWARD:
            f_next = float(np.clip(f_next, -_MAX_ABS_FORWARD, _MAX_ABS_FORWARD))
        f = f_next
        if abs(step) <= _NEWTON_TOL:
            return f

    res = root(
        lambda x: float((flows * df_prev * np.exp(-x[0] * dt)).sum()) - remaining_pv,
        [x0],
        method="hybr",
    )
    if not res.success:
        raise ValueError(
            f"Piecewise-constant bootstrap failed to converge: {res.message}"
        )
    return float(res.x[0])


def _bootstrap_piecewise_constant(
    curve_date: date,
    parsed: list[_ParsedInstrument],
) -> np.ndarray:
    """Exact sequential bootstrap of continuously compounded Act/365 segment forwards."""
    n = len(parsed)
    knot_tau = np.array(
        [0.0, *[(inst.maturity - curve_date).days / DAY_BASE for inst in parsed]],
        dtype=float,
    )
    forwards = np.empty(n, dtype=float)
    cum = np.zeros(n + 1, dtype=float)

    for i, inst in enumerate(parsed):
        tau_prev = knot_tau[i]
        df_prev = math.exp(-cum[i])
        taus = inst.days / DAY_BASE

        if i == 0:
            known_pv = 0.0
            mask_new = np.ones(taus.size, dtype=bool)
        else:
            mask_new = taus > tau_prev
            mask_old = ~mask_new
            if mask_old.any():
                t_old = taus[mask_old]
                # Segment index in [0, i - 1] for each earlier flow.
                seg = np.searchsorted(knot_tau[1:i], t_old, side="right")
                dfs_old = np.exp(-(cum[seg] + forwards[seg] * (t_old - knot_tau[seg])))
                known_pv = float((inst.flows[mask_old] * dfs_old).sum())
            else:
                known_pv = 0.0

        dt_new = taus[mask_new] - tau_prev
        flows_new = inst.flows[mask_new]
        remaining_pv = inst.target_pv - known_pv
        x0 = forwards[i - 1] if i > 0 else inst.seed_rate
        f_i = _solve_segment_forward(dt_new, flows_new, df_prev, remaining_pv, x0)
        forwards[i] = f_i
        cum[i + 1] = cum[i] + f_i * (knot_tau[i + 1] - tau_prev)

    return forwards


def build_curve_from_instruments(
    curve_date: date,
    instruments: Sequence[tuple[CLBond | Deposit, Rate]],
    *,
    interp_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant,
    reprice_tol: float = _INSTRUMENT_REPRICE_TOL,
) -> PillarCurve:
    """Bootstrap a :class:`~fintoolsom.rates.PillarCurve` from bonds and deposits.

    Each instrument pins one pillar at its maturity date (payment date for a
    :class:`~fintoolsom.fixedIncome.Deposit`, last coupon end date for a
    :class:`~fintoolsom.fixedIncome.CLBond`). Instruments may be passed in any order;
    their maturities must be strictly after `curve_date` and mutually distinct.

    A :class:`~fintoolsom.fixedIncome.Deposit` is valued as a single cash flow of 100
    at its payment date discounted in its quote's :class:`~fintoolsom.rates.Rate`
    convention; a :class:`~fintoolsom.fixedIncome.CLBond` is valued at its base-100
    present value (:meth:`~fintoolsom.fixedIncome.CLBond.get_present_value`).

    Parameters
    ----------
    curve_date : date
        Valuation / curve date (``df = 1``).
    instruments : sequence of (CLBond | Deposit, Rate)
        Instruments and their quoted rates (at least 2, with distinct maturities).
    interp_method : InterpolationMethod, optional
        Interpolation method of the resulting :class:`PillarCurve`
        (``PiecewiseConstant`` by default).
    reprice_tol : float, optional
        Maximum allowed ``|PV(curve) - PV(quote)|`` on base 100 across all
        instruments (``1e-6`` by default); above it, :class:`RepricingError`.

    Returns
    -------
    PillarCurve
        Curve with knots ``[curve_date, *sorted_maturities]`` and precalculated daily
        discount factors.

    Raises
    ------
    InsufficientQuotesError
        If fewer than 2 instruments are provided (a :class:`PillarCurve` needs at
        least 2 segments / 3 knots).
    TypeError
        If an element is neither a :class:`CLBond` nor a :class:`Deposit`.
    ValueError
        If an instrument has no flows after `curve_date`, two instruments share the
        same maturity date, `reprice_tol` is not a positive finite number, or the
        solver does not converge.
    RepricingError
        If any instrument's base-100 PV residual exceeds `reprice_tol`.
    """
    if not (
        isinstance(reprice_tol, (int, float))
        and math.isfinite(reprice_tol)
        and reprice_tol > 0
    ):
        raise ValueError(
            f"reprice_tol must be a positive finite number; got {reprice_tol!r}."
        )
    if len(instruments) < MIN_KNOTS - 1:
        raise InsufficientQuotesError(
            f"A PillarCurve needs at least {MIN_KNOTS - 1} instruments "
            f"({MIN_KNOTS} knots); got {len(instruments)}."
        )

    parsed = [_parse_instrument(curve_date, inst, irr) for inst, irr in instruments]
    parsed.sort(key=lambda item: item.maturity)

    # Check for duplicate maturity dates.
    duplicates = sorted(
        {a.maturity for a, b in zip(parsed, parsed[1:]) if a.maturity == b.maturity}
    )
    if duplicates:
        raise ValueError(
            f"Instruments must have distinct maturity dates; duplicates: {duplicates}."
        )

    knots = [curve_date, *(inst.maturity for inst in parsed)]
    forwards = _bootstrap_piecewise_constant(curve_date, parsed)
    curve = PillarCurve(
        curve_date,
        knots,
        forwards,
        interp_method,
        precalculate=False,
    )

    # Pre-index all unique flow day offsets for fast vectorized valuation.
    unique_days = np.unique(np.concatenate([inst.days for inst in parsed]))
    flow_slices = [
        (np.searchsorted(unique_days, inst.days), inst.flows, inst.target_pv)
        for inst in parsed
    ]

    def residuals(x: np.ndarray) -> np.ndarray:
        curve._set_forward_rates(x)
        dfs = curve._dfs_from_days(unique_days)
        return np.array(
            [
                float((dfs[idx] * flows).sum()) - target
                for idx, flows, target in flow_slices
            ],
            dtype=float,
        )

    if interp_method is not InterpolationMethod.PiecewiseConstant:
        try:
            res = root(residuals, forwards, method="hybr")
        except ValueError as exc:
            raise ValueError(
                f"Instrument curve build did not converge ({exc})."
            ) from exc
        if not res.success:
            raise ValueError(
                f"Instrument curve build failed to converge: {res.message}"
            )
        curve._set_forward_rates(res.x)

    final_residuals = residuals(curve.forward_rates)
    max_err = float(np.max(np.abs(final_residuals)))
    if not math.isfinite(max_err) or max_err > reprice_tol:
        raise RepricingError(
            f"Built curve did not reprice instruments within {reprice_tol}: "
            f"max |PV error| = {max_err:.6g}."
        )

    curve.precalculate_dfs()
    return curve
