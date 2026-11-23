# fintoolsom.models

Parametric term-structure models: Nelson-Siegel and Nelson-Siegel-Svensson.

## Nelson-Siegel family (`NelsonSiegelSvensson.py`)

The zero rate (continuous, Act/365) at year fraction `t` is:

```
Nelson-Siegel:   r(t) = b0 + b1·S(x) + b2·C(x)
Svensson:        r(t) = b0 + b1·S(x) + b2·C(x) + b3·C(y)
S(x) = (1 − e^−x) / x,   C(x) = S(x) − e^−x
x = HUMP_PEAK·t / λ,     y = HUMP_PEAK·t / μ
```

`b0` is the level, `b1` the slope, and `b2` and `b3` the curvatures. `λ` and `μ`
are the years to each hump: `C` peaks at `HUMP_PEAK` ≈ 1.7933, so `C(x)` peaks at
`t = λ`.

| Name | What it does |
|---|---|
| `ns_factors(t, λ)`, `nss_factors(t, λ, μ)` | Factor loadings `[level, slope, curvature(, curvature 2)]` |
| `ns_rate(t, b0..b2, λ)`, `nss_rate(t, b0..b3, λ, μ)` | Zero rate at `t` |
| `ns_df(...)`, `nss_df(...)` | Discount factor `exp(−r(t)·t)`, 1 at `t = 0` |
| `NelsonSiegel(b0, b1, b2, lambda_)` | One hump, calibrated in 0.25-20 years |
| `NelsonSiegelSvensson(b0, b1, b2, b3, lambda_, mu_)` | Two humps, calibrated in 0.25-3 and 2-20 years |
| `.get_rate(t)`, `.get_df(t)`, `.get_params()` | Both models; `get_params()` is the betas, then the humps |
| `.calibrate_from_bonds(date, [(CLBond, Rate), ...], duration_weighted=True)` | Fits every parameter to the bonds' PVs at their IRRs; returns one `CalibrationError` per bond |
| `.calibrate_from_instruments(date, bonds_irr_list=[(CLBond, Rate), ...], deposits_irr_list=[(Deposit, Rate), ...], duration_weighted=True)` | The same with bonds and deposits in one stack of flows (a deposit is one flow of 100 at its payment date); returns one `CalibrationError` per instrument, bonds first |
| `.calibrate_from_curve(curve)` | Fits a `PillarCurve`'s zero rates at its knots (humps by the solver, betas solved exactly for each); returns one `CalibrationError` per knot |
| `Model.calibrate_many_from_curves(curves, global_humps=False)` | Fits many curves at once in batched numpy; returns `(models, errors)`, one model and one error list per curve |
| `Model.calibrate_many_from_bonds([(date, [(CLBond, Rate), ...]), ...])` | Fits many bond sets at once in batched numpy; returns `(models, errors)`, one model and one error list per set |
| `CalibrationError` | `TypedDict` `{"term": float, "error_bps": float}`: the fit at one point |

### Calibration errors

Both calibrations set the model's parameters and return the fit at each point,
fitted minus observed: `[{"term": 1.0, "error_bps": 2.1}, ...]`.

- Bonds, in the order given: `term` is the bond's duration on the fitted curve,
  `CLBond.get_duration_from_curve` (Σ t·flow·df / PV), and `error_bps` the IRR that
  prices the bond at its fitted PV minus its quoted IRR, in the quote's convention.
- Curve, per knot after the curve date: `term` is its Act/365 year fraction, and
  `error_bps` the fitted zero rate minus the curve's, continuous Act/365.

An exact fit reports errors of about 1e-7bp, the floor of the normal equations.

Every parameter has a default, the starting point of a fit, so `NelsonSiegel()`
works. The two models are siblings, not a subclass of each other: they share their
calibration, but not their parameters.

The functions take year fractions (floats or numpy arrays), not dates, so they can
be vectorised and used inside optimisers.

## Example

```python
import numpy as np

from fintoolsom.models import NelsonSiegel, NelsonSiegelSvensson
from fintoolsom.models.NelsonSiegelSvensson import nss_rate

t = np.array([0.5, 1, 5, 10, 30])
nss_rate(t, 0.05, -0.01, 0.02, -0.01, 1.5, 12.0)  # b0..b3, humps at 1.5y and 12y
NelsonSiegel(0.05, -0.01, 0.02, lambda_=3.0).get_df(t)
```

Fitting Nelson-Siegel to a curve with two humps leaves a few bp at each pillar:

```python
from datetime import date, timedelta

from fintoolsom.rates import InterpolationMethod, PillarCurve

curve_date = date(2024, 7, 8)
pillars = [curve_date + timedelta(days=round(365 * y)) for y in (0.5, 1, 2, 5, 10, 30)]
dfs = [0.9775, 0.9522, 0.8998, 0.7524, 0.5552, 0.1848]
curve = PillarCurve.from_dfs(curve_date, pillars, dfs, InterpolationMethod.PiecewiseConstant)

ns = NelsonSiegel()
errors = ns.calibrate_from_curve(curve)
# [{'term': 0.4986, 'error_bps': 3.78}, {'term': 1.0, 'error_bps': -3.55}, ...,
#  {'term': 30.0, 'error_bps': 0.99}]; ns.lambda_ is about 5.09 years
```

## Many curves

`NelsonSiegel.calibrate_many_from_curves(curves)` (and the same on
`NelsonSiegelSvensson`) is a classmethod that fits every curve at once and returns
new models with their errors, as `calibrate_from_curve` would:

```python
models, errors = NelsonSiegelSvensson.calibrate_many_from_curves(curves)
models, errors = NelsonSiegelSvensson.calibrate_many_from_curves(curves, global_humps=True)
models, errors = NelsonSiegel.calibrate_many_from_curves(curves, initial_guess=[1.5])  # warm start
```

- **Pillars need not match.** Each curve keeps its own year fractions; curves are
  padded to the longest with a mask, so they can differ in pillar dates and count.
- **`global_humps=False` (default): humps per curve.** The same fits as a loop of
  `calibrate_from_curve`: the hump grid is scored for every curve in one batched
  solve, then a batched damped Newton refines every curve's humps, each stopping
  when it converges. Its Hessian is a finite difference of the analytic gradient
  with absolute eigenvalues: on noisy rates the Gauss-Newton J'J that
  `least_squares` uses mispredicts the hump curvature, and plain Newton is drawn to
  the saddle of two equal NSS humps. NSS curves whose best grid start has
  near-equal or swapped humps are also started from humps apart; the better fit is
  kept. A curve not converged after 100 iterations is refitted alone.
- **`global_humps=True`: one set of humps for all curves**, each curve keeping its
  own betas, minimising the summed squared error (Diebold-Li's fixed lambda). The
  betas then sit on one set of loadings, so their series read as level, slope and
  curvature on a fixed basis; each curve fits worse.
- **Speed.** 100 daily curves of 15 pillars, each pillar moved by a random
  ±3 days, 1bp of noise: NS 15.6 ms against 166 ms for the loop (10.6x), NSS
  21.9 ms against 148 ms (6.8x); global humps 16.0 and 17.0 ms. At 1,000 curves
  13.5x and 10.0x. No curve fits worse than the loop; where the single fit stops
  early (one NSS curve in 1,000 raised, a few stopped on the equal-humps saddle),
  the batched fit is better.

## Many bond sets

`calibrate_many_from_bonds(bond_sets)` fits one model per
`(calibration_date, [(bond, irr), ...])` set, with the residuals and objective of
`calibrate_from_bonds`:

```python
models, errors = NelsonSiegelSvensson.calibrate_many_from_bonds(
    [(date_1, bonds_1), (date_2, bonds_2), ...]
)
```

- **Sets may differ** in date, bond count and flows (padded with zero flows), and
  quotes may be in any rate convention.
- **Start.** The best point of the hump grid, with its betas, on a pseudo-curve of
  each bond's zero rate at its duration (the PV-weighted continuous zero rate of
  its flows at its own IRR), scored for every set at once. The single fit starts
  from fixed humps, so on multi-modal fits (10 noisy NSS bonds often are) the two
  can end in different local minima. `initial_guess=` starts every set from given
  parameters instead.
- **Solver.** A batched damped Gauss-Newton on all parameters, humps in log with
  steps of at most 0.5 in log hump (an undamped first step can throw the first
  hump onto its lower bound, where huge opposite betas fit worse). A set not
  converged after 200 iterations is refitted alone by `calibrate_from_bonds`.
- **Errors.** The IRRs at the fitted PVs come from one batched Newton solve over
  every bond, in each quote's convention.
- **Speed.** Sets of 10 semiannual bullets (1-30 years) on drifting NSS curves,
  quoted with 1bp of IRR noise, against a loop of `calibrate_from_bonds`:

  | | 50 sets | 100 sets |
  | --- | --- | --- |
  | NS | 51.6 ms vs 187 ms (3.6x) | 113 ms vs 383 ms (3.4x) |
  | NSS | 117 ms vs 343 ms (2.9x) | 313 ms vs 859 ms (2.7x) |

  NS fits match the loop (one set better). NSS fits better on average (mean rms
  0.718 vs 0.737bp on 50 sets, 0.740 vs 0.754bp on 100): 6 and 11 sets better, 2
  and 2 worse (up to 0.08 and 1.3bp), the loop also failing on 2 of the 100.

## Notes

- **As a curve.** `rates.ModelCurve(t, model, horizon)` wraps a fitted model into a
  `Curve` for a `Market` or `CLBond`; both models are `rates.TermStructureModel`s.
- **Fewer inputs, fewer humps.** With a handful of bonds Svensson's two humps are
  close to collinear; `NelsonSiegel` fits them more stably.
- **Solver.** Both calibrations are bounded least squares on a residual vector
  (`scipy.optimize.least_squares`, trust-region reflective) with an analytic
  Jacobian, and raise `ValueError` if it does not converge:
  - `calibrate_from_bonds` and `calibrate_from_instruments`: one residual per
    instrument (model PV minus PV at the IRR), all parameters fitted jointly,
    since the betas sit inside the dfs' exponential. With
    `duration_weighted=True` (the default) each residual is divided by the
    instrument's dPV/dIRR at its quote, so the fit minimises the IRR errors
    (to first order) and a 30-day deposit weighs as much per bp as a 10-year
    bond; `False` fits PV differences, which weighs an instrument by its
    duration (long bonds dominate, the short end takes the error in bp).
    `calibrate_many_from_bonds` fits PV differences, as `duration_weighted=False`.
  - `calibrate_from_curve`: variable projection. Each evaluation solves the betas
    exactly for the humps it is given, so the solver only moves the humps and the
    betas always belong to them (Golub-Pereyra Jacobian). It starts from the best
    point of a coarse hump grid (~150 points, one batched linear solve), which
    keeps two-hump fits out of local minima.
  - On 10 bonds / 20 pillars: bonds 2.3 ms (NS) and 5.1 ms (NSS), about 0.5 ms of
    which is the IRR solves behind the errors; curve 0.8 ms and 1.3 ms. Powell took
    42, 30, 3.3 and 7.1 ms.
  - Pass `initial_guess` to start from a previous fit. Its humps start at least 1%
    inside their bounds (0.25 to 0.2525, 20 to 19.8), and the fit can still end on
    one.
- **Short humps.** A hump before the first pillar or bond flow is weakly
  identified: the fit's rates still match, but the hump itself moves more than the
  rates do.
