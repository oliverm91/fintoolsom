# fintoolsom.curve_builder

Bootstraps the curves of a `Market` from instrument quotes, so that every quote
that pins a pillar reprices to zero MTM.

```python
from fintoolsom.curve_builder import build_curves
from fintoolsom.rates import InterpolationMethod

build_curves(quotes, market)  # fills market.curves in place
build_curves(quotes, market, discount_interp_method=InterpolationMethod.HermiteCubicSpline)
```

## What it builds

- **Index curves.** Every index a quote reads (collateral, floating legs) gets an
  `IndexCurve` under its name, such as `"ICP"` or `"SOFR"`.
- **Discount curves.** Every currency other than the risk-free index's gets a
  `DiscountCurve` under `"<ccy>_<risk-free index>"`, such as `"USD_ICP"`.
- **Interpolation.** New curves use `discount_interp_method` (`LogLinear` by
  default, `HermiteCubicSpline` or `CubicSpline`) and `index_interp_method`
  (`PiecewiseConstant` forwards by default, `Linear` forwards or `CubicSpline`). A
  curve already in the market keeps its own method when the build extends it past
  its last date.

## What the market needs

- `market.t` equal to the quotes' `quote_date`. A mismatch raises `ValueError`.
- `market.risk_free_index`: the numeraire every discount curve is built under.
- FX spot for every pair a quote converts, and index histories for:
  - fixings that started coupons read;
  - the short end of overnight curves: today's fixing, or the last level ratio for
    ICP.

## How it works

The build is split in private modules, driven by `builder._CurveBuilder`.

1. **Keys (`_keys.py`).**
   - `_curves_needed(quote)` lists every curve a quote reads: the curves of its
     payment currencies, its collateral index and its floating indexes.
   - Quotes that read the same set of curves form a group.
2. **Pillar ownership (`_plan.py`).**
   - Each quote owns at most one pillar: its maturity, on the first of its
     candidate curves where that date is still open (curve not built, or the date
     past its end) and not already taken by another quote of the group.
   - Candidates, in preference order:
     - FX forward or cross-currency swap: the non-risk-free funding curve, i.e.
       the basis.
     - Single-currency swap: its floating index curve.
     - Basis swap: whichever index no IRS already pins.
   - A quote with no open pillar is *ditched*: the curves ignore it, and the build
     reports it in a `RedundantQuoteWarning`.
3. **Scheduling (`_plan.py`).**
   - A group is ready once no other pending group still owns a pillar on a curve
     it reads, so a curve is never read before it is complete.
   - Ready groups solve smallest first, then in quote order.
   - Groups waiting on each other form a cycle and are merged into one joint
     solve. A typical cycle is SOFR-collateralised cross-currency swaps that read
     SOFR while the OIS builds it.
   - Ownership is re-planned after every solve.
4. **Seeds and anchors (`_seeds.py`).** A new index curve gets a fixed short end:
   - Overnight index with fixings: knots start at the next business day, and
     today's rate covers `[t, t + 1bd]` as the curve's `spot_rate`.
   - Term index: the first segment is today's fixing over `[t + spot lag, + term]`.
   - Otherwise: knots start at `t`.

   Free pillars are seeded from the quotes themselves: fixed-leg zero rates,
   covered interest parity for FX forwards, spreads for basis swaps. Good seeds
   keep the root solves short; the solve itself is exact.
5. **Solving (`_solver.py`).**
   - Unknowns are log discount factors on discount curves and segment forwards on
     index curves.
   - With local interpolation on every curve (log-linear dfs, piecewise-constant
     or linear forwards), a quote depends only on pillars up to its own maturity. The group
     is then bootstrapped date by date, in small square blocks of at most one
     unknown per curve.
   - A spline curve moves everywhere, so its group is solved jointly with one
     `scipy.optimize.root` (hybr).
   - If any curve is a `CubicSpline`, the groups still solve in order to settle
     pillar ownership, and every pillar is then re-solved in one joint `root` from
     that result: a later spline pillar moves quotes that were fitted earlier.
   - Every solve is atomic: on failure the curves are restored.
6. **Checks (`builder.py`).**
   - After the last group, every quote that pinned a pillar must still reprice
     within `1e-4` (report currency CLP). Otherwise the build raises
     `RepricingError`: a later group moved a curve it was fitted on.
   - Ditched quotes are then warned once, with their MTM:
     - swaps also in bp of their DV01;
     - FX forwards also as a percentage of notional.

## Errors and warnings

| Name | When |
|---|---|
| `InsufficientQuotesError` | A group has more free pillars than quotes, reads a curve nobody builds, or a term index has no rate history |
| `RepricingError` | A pinning quote no longer reprices after the whole build |
| `RedundantQuoteWarning` | Quotes owned no pillar. `.ditched` holds `(quote, mtm, mtm_pct, mtm_bp)` |

## Notes

- **Quote order.** The order of the quotes changes neither ownership nor the
  resulting curves. Tests check this to `1e-9`.
- **Pillar dates.** Pillar dates are quote maturities, not a fixed grid. Two
  quotes of one group with the same maturity and no other open candidate both keep
  that pillar: the system is over-determined and the build fails. Drop one of them.
- **Bumping.** Bump a built curve with `market.set_curve(curve.parallel_bump_bps(1))`.
  Rebuilding from bumped quotes is the alternative for a quote-level risk view.
- **Not built.** No synthetic numeraire curves (such as `CLP_SOFR`) and no
  discounting on term indexes.
