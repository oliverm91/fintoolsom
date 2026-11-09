# fintoolsom.models

Parametric term-structure models. Today: Nelson-Siegel-Svensson.

## Nelson-Siegel-Svensson (`NelsonSiegelSvensson.py`)

The zero rate (continuous, Act/365) at year fraction `t` is:

```
r(t) = b0 + b1·L1(t/λ) + b2·C(t/λ) + b3·C(μ·t)
L1(x) = (1 − e^−x) / x,   C(x) = L1(x) − e^−x
```

`b0` is the level, `b1` the slope, and `b2` and `b3` the two curvatures. Note the
two decays enter differently: `λ` is a time scale (`t/λ`), while `μ` is a rate
(`μ·t`).

| Name | What it does |
|---|---|
| `nss_factors(t, λ, μ)` | Factor loadings `[level, slope, curvature, curvature 2]` |
| `nss_rate(t, b0..b3, λ, μ)` | Zero rate at `t` |
| `nss_df(t, b0..b3, λ, μ)` | Discount factor `exp(−r(t)·t)`, 1 at `t = 0` |
| `NelsonSiegelSvensson` | Holds the parameters: `get_rate(t)`, `get_df(t)`, `get_params()` |
| `.calibrate_from_bonds(date, [(CLBond, Rate), ...])` | Fits the parameters to bond IRRs (`calibrate` is a deprecated alias) |
| `.calibrate_from_curve(zcc)` | Fits the parameters to a `ZeroCouponCurve` |

The functions take year fractions (floats or numpy arrays), not dates, so they can
be vectorised and used inside optimisers.

## Example

```python
import numpy as np

from fintoolsom.models.NelsonSiegelSvensson import nss_df, nss_rate

params = (0.05, -0.01, 0.02, -0.01, 1.5, 0.1)  # b0, b1, b2, b3, lambda, mu
t = np.array([0.5, 1, 5, 10, 30])
nss_rate(t, *params)
nss_df(t, *params)
```

## Notes

- **Legacy curves.** Calibration works on `fintoolsom.fixedIncome.CLBond` and on
  the legacy `ZeroCouponCurve`, which is kept for this module.
- **No market curve.** To use a fitted curve in a `Market`, sample `nss_df` on
  pillar dates and build a `rates.DiscountCurve` or `rates.IndexCurve` from them.
  `NelsonSiegelSvensson.get_curve` was removed together with `NSSCurve`.
- **Optimiser.** `calibrate_from_bonds` uses `scipy.optimize.minimize` (`powell` by default);
  pass `initial_guess` to start from a previous fit.
