# Branch `cpp-curve-builder`

Branched from `cpp-backend` at `36dc02f` (2026-11-16). Goal: port the whole
`curve_builder` to the C++23 backend (`fintoolsom._native`), so an all-C++ build runs
from the quotes to the final curves in C++, with the Python builder kept as the
reference and as the default.

Status: **phases 0-3 done** (harness, determinism, backend switch; legs as arrays,
built in one C++ call; the group solves in C++; the hot parts of building the
instruments, planning and seeding in C++). This file is the plan and the acceptance criteria; it is updated as
the port lands.

## Verifying

    uv run python cpp/benchmarks/bench_curve_builder.py [--quotes notebook|factory] [--no-timing]

builds with the Python and the C++ curve builder (both on the C++ valuation backend),
for every method, and prints the timings, the max forward difference (exit status 1
above 1e-8, or if the builders ditch different quotes) and the valuation error: max /
RMS / sum |MTM| of the fitted quotes, and the count and max of the ditched ones
(quotes owning no pillar, not fitted by design: with them the "total" is about 12
CLP on the notebook quotes whatever the builder, so they are reported apart).
`tests/unit/curve_builder/test_builder_backends.py` asserts the same on the factory
quotes, and that builds do not depend on `PYTHONHASHSEED`.

## Scope

The Python `curve_builder` today (2,069 lines):

| Module | Lines | What it does |
|---|---:|---|
| `builder.py` | 466 | `build_curves`: groups the quotes, orders the groups, solves each, checks the repricing |
| `_solver.py` | 613 | `_GroupSolve` (unknowns to curves, atomic apply / rollback), sweeps with block Newton, joint hybr with the analytic Jacobian |
| `_seeds.py` | 364 | initial guesses (fixed legs, covered interest parity, basis swaps, interpolation) and short-end anchors |
| `_plan.py` | 335 | curves each quote reads, pillar ownership, fixed and free pillars |
| `_keys.py` | 246 | curve keys, maturities, the instrument of each quote |
| `_errors.py` | 34 | `InsufficientQuotesError`, `RepricingError`, `RedundantQuoteWarning` |

What a C++ build still runs in Python (all-C++ backend, notebook quotes, cProfile
shares of a PiecewiseConstant build): the instruments built from the quotes (~30%),
the solves' glue (~40%: Newton or hybr, curve updates, batches), planning and seeding
(~20%), the rest grouping and checks. The valuation itself is already C++.

## Acceptance criteria

Every change is verified on the notebook quotes (`notebooks/quotes.json`, as in
`cpp/benchmarks/bench_curves.py`) and on the test factories' quotes, for every
interpolation method (PiecewiseConstant, PiecewiseLinear, CubicSpline,
HermiteCubicSpline, ConvexMonotone):

1. **Timings**: 3 warm-ups + 10 builds, the 2 slowest dropped, mean ± std, against
   the baseline below and against the Python builder on the same backend.
2. **Forward rates**: per curve, max |forward C++ builder − forward Python builder|,
   in absolute terms and in bp. Target: bit-identical; if a step cannot be (e.g. a
   different solver path), the difference is stated and bounded (≤ 1e-12).
3. **Total valuation error**: every quote instrument revalued by
   `Calculator.valuate` on the final curves, in the report currency: max |MTM|, RMS
   and sum |MTM|, for both builders. Target: not worse than the Python builder's.
4. The whole suite passes in the seven backend configurations (python;
   `FINTOOLSOM_BACKEND=cpp`;
   `FINTOOLSOM_BACKEND_{RATES,MARKET,DERIVATIVES,DATES,CURVE_BUILDER}=cpp`), run with
   `pytest -n 4` (fastest: 22 s against 60 s serial and 27 s with `-n auto`); the errors
   and warnings raised are the Python builder's.

Criterion 2's target is superseded by the decisions below: within 1e-8, not
bit-identical.

## Baseline (cpp-backend at 36dc02f)

Notebook build, `PYTHONHASHSEED=0` (builds depend on it at ~4e-15), 3 warm-ups +
10 builds, 2 slowest dropped:

| Interpolation | Python backend | C++ backend |
|---|---:|---:|
| PiecewiseConstant | 40.6 ± 0.3 ms | 12.1 ± 0.1 ms |
| PiecewiseLinear | 72.3 ± 2.0 ms | 14.1 ± 1.4 ms |
| CubicSpline | 65.9 ± 0.5 ms | 18.3 ± 0.8 ms |
| HermiteCubicSpline | 89.3 ± 0.4 ms | 20.0 ± 0.3 ms |
| ConvexMonotone | 113.3 ± 0.8 ms | 20.0 ± 0.3 ms |

Largest repricing error of the quote instruments on the final curves: 2.9e-11
(PiecewiseConstant), 1.5e-11 (PiecewiseLinear), 7.3e-11 (CubicSpline, ConvexMonotone),
2.9e-11 (HermiteCubicSpline). Python and C++ backends give bit-identical curves.

Phase-0 baseline of the harness (both builders still the Python code, so no
difference; notebook quotes, C++ valuation), fitted quotes' |MTM| max / RMS / sum,
5 ditched quotes (max |MTM| 12 CLP):

| Interpolation | Build | Fitted max / RMS / sum |MTM| |
|---|---:|---|
| PiecewiseConstant | 12.1 ms | 2.9e-11 / 6.8e-12 / 1.9e-10 |
| PiecewiseLinear | 12.8 ms | 1.5e-11 / 6.1e-12 / 1.7e-10 |
| CubicSpline | 17.4 ms | 7.3e-11 / 1.6e-11 / 5.0e-10 |
| HermiteCubicSpline | 19.5 ms | 2.9e-11 / 8.4e-12 / 3.1e-10 |
| ConvexMonotone | 20.2 ms | 7.3e-11 / 1.6e-11 / 5.1e-10 |

## Phase 0 (done)

- Builds no longer depend on `PYTHONHASHSEED`: `_Planner.free_pillars` walked a
  group's frozenset of curve keys, so the order of a joint solve's unknowns (and its
  path) followed the hash seed (CubicSpline differed at ~4e-15 for seed 3). It now
  walks them in curve-id order. Checked over seeds 0, 1-7 and 11 for every method.
- `curve_builder` is a backend subpackage (`settings.SUBPACKAGES`, `PORTED`): the
  global `"cpp"` reaches it, `FINTOOLSOM_BACKEND_CURVE_BUILDER` overrides it. Its C++
  path is still the Python code.
- The harness above, and its test.

## Phase 1 (done): instruments from quotes

Profiling showed the cost was not the coupon objects but each leg's fixed overhead:
some fifteen small numpy calls (~1.3 µs each: date conversions, date checks,
amortizations), about 20 µs per leg, on legs of ~8 coupons.

- **Legs hold per-coupon arrays** (`derivatives/swaps/legs.py`): start, end, optional
  payment and FX fixing dates and residuals, stored as read-only ordinal and float
  arrays. The amortizations come from the residuals (the notional is fully repaid by
  the last coupon; negative amortizations are drawdowns). One rate per fixed leg, one
  spread per floating leg. Term-rate fixing dates default to the start dates and may
  not be after them. `get_coupons()` / `get_coupon(i)` build coupon objects on
  request; `from_coupons()` builds a leg from them, for legacy code. No `__eq__`
  (identity). The swaps README documents it. Removed: hand-built legs whose
  amortizations do not repay the notional (the README's "explicit amounts" trick).
- **One C++ call per leg** on the `derivatives` C++ backend
  (`fintoolsom._native.leg_arrays`): the ordinals, the date checks, the
  amortizations and the par-schedule flag. It returns None for inputs it does not
  take (dates that are not exact `date`s, mismatched lengths, a failed check), and the
  Python path then raises its usual errors.
- `dates.date_counts.date_ordinals` also takes an `(n, 3)` integer array of
  `(year, month, day)` rows (vectorised numpy, both backends).

Forward rates are bit-identical to phase 0's on both backends, every method, both
quote sets. Results (notebook quotes, C++ valuation, 3 warm-ups + 10 builds, 2
slowest dropped):

| | Phase 0 | Phase 1 |
|---|---:|---:|
| Fixed leg / overnight leg construction | 20.1 / 24.9 µs | 9.1 / 10.6 µs |
| One IRS instrument | 55.3 µs | 29.9 µs |
| The 73 notebook instruments | 4.38 ms | 2.24 ms |

| Interpolation | Build, phase 0 | Build, phase 1 | Fitted max / RMS / sum \|MTM\| | Max \|dfwd\| vs Python builder |
|---|---:|---:|---|---:|
| PiecewiseConstant | 12.1 ms | 10.4 ± 0.1 ms | 2.9e-11 / 6.8e-12 / 1.9e-10 | 0 |
| PiecewiseLinear | 12.8 ms | 11.0 ± 0.1 ms | 1.5e-11 / 6.1e-12 / 1.7e-10 | 0 |
| CubicSpline | 17.4 ms | 15.8 ± 0.2 ms | 7.3e-11 / 1.6e-11 / 5.0e-10 | 0 |
| HermiteCubicSpline | 19.5 ms | 17.9 ± 0.2 ms | 2.9e-11 / 8.4e-12 / 3.1e-10 | 0 |
| ConvexMonotone | 20.2 ms | 18.2 ± 0.2 ms | 7.3e-11 / 1.6e-11 / 5.1e-10 | 0 |

The gain is the same for both builders: they share the instruments (the derivatives
backend), so the "C++ builder" column of the harness still equals the Python one.

Where an all-C++ PiecewiseConstant build spends its time now (cProfile shares): the
solves ~50% (block Newton and its Python glue: residual values, curve updates), the
instruments ~20%, planning and seeding ~23%. Phase 2 targets the first.

## Phase 2 (done): the group solves in C++

`fintoolsom._native.GroupSolver` (`cpp/src/bind_solver.cpp`) takes a group solve
when the C++ curve builder is selected, every residual instrument is valued in one C++
market call and every curve solved has a C++ core (`_solver._native_solver`). The
unknowns are set on the curves' cores in place; the Python curves catch up on the
builder's `apply` of the solution. No Python runs per iteration.

- **Sweeps**: the block Newton of `_solver._newton` / `_solve_small` line for line
  (`math.fsum` included), residuals from `market_valuate_many`: the same steps, so
  **bit-identical** forwards. A block Newton gives up on (one on the factory quotes,
  as in Python) goes back to Python for hybr, then the C++ sweep resumes at the next
  block.
- **Joint solve**: Newton steps with backtracking on the analytic Jacobian of
  `_Jacobian` (batch gradient times d log df / d forwards, bumped on private copies of
  the cores), instead of scipy's hybr: the same root by another path, **forwards
  within 1.1e-14** of the Python builder's, fitted repricing errors of the same order
  (better on CubicSpline). The batch layout (`ValuationBatch(compile_all=True)` and
  `_Jacobian._source_terms`) is compiled in C++ (`compile_batch` in
  `bind_market.cpp`): in Python it cost 3 ms per spline build.
- Fallbacks: hybr in Python when the joint Newton stops short or the batch does not
  compile (a coupon already running, another instrument type); the whole Python solve
  when a valuation raises (so its error is the calculator's).

Results (notebook quotes, C++ valuation, 3 warm-ups + 10 builds, 2 slowest dropped;
this machine ran ~10% slower than during phase 1: the Python builder column is the
same code as phase 1's 10.4-18.2 ms):

| Interpolation | Python builder | C++ builder | Gain | Max \|dfwd\| | Fitted max / RMS / sum \|MTM\| (C++) |
|---|---:|---:|---:|---:|---|
| PiecewiseConstant | 11.5 ± 0.7 ms | 9.1 ± 0.6 ms | 1.27x | 0 | 2.9e-11 / 6.8e-12 / 1.9e-10 |
| PiecewiseLinear | 12.6 ± 0.6 ms | 9.9 ± 0.6 ms | 1.27x | 0 | 1.5e-11 / 6.1e-12 / 1.7e-10 |
| CubicSpline | 17.6 ± 0.5 ms | 11.1 ± 0.5 ms | 1.58x | 8.9e-15 | 2.9e-11 / 7.8e-12 / 2.6e-10 |
| HermiteCubicSpline | 19.6 ± 0.5 ms | 11.4 ± 0.5 ms | 1.73x | 1.1e-14 | 2.9e-11 / 1.2e-11 / 4.8e-10 |
| ConvexMonotone | 19.4 ± 0.1 ms | 12.1 ± 0.5 ms | 1.60x | 3.5e-15 | 5.8e-11 / 1.3e-11 / 3.3e-10 |

Time in the solves per build (C++ valuation): PiecewiseConstant 4.0 → 1.6 ms,
PiecewiseLinear 4.6 → 2.3 ms, CubicSpline 8.8 → 3.7 ms, HermiteCubicSpline
10.8 → 4.0 ms, ConvexMonotone 11.5 → 4.9 ms. The factory quotes give the same
picture (bit-identical local methods, splines within 1.1e-14). What a C++ build still
runs in Python: the instruments from the quotes (~2.2 ms), planning and seeding
(~2.5 ms), the grouping, the repricing check and the final curves' grids; each C++
residual evaluation still reads the instruments' Python attributes through the market
view (~3 µs per instrument).

## Phase 3 (done): instruments, planning and seeding

Profiled first (PiecewiseConstant, C++ builder, ~9.9 ms a build): the instruments
~2.9 ms (~39 µs an IRS, ~45 µs an XCCY), `_get_maturity` 1.2 ms (650 calls a build,
each walking the legs' date lists), the seeds 0.9 ms (most of it ~10 scalar wealth
factors per fixed leg), the rest of the planning ~1.5 ms spread thinly over dict
and set work. What moved to C++:

- **Bullet legs in one call** (`_native.bullet_leg`, `SwapLeg._bullet_parts`): the
  schedule, the payment and FX fixing lags, the ordinals, checks, amortizations and
  the fixed interest and flows of `FixedLeg` / `OvernightLeg` / `XCCYFloatingLeg` /
  `TermRateLeg.from_term`, handed to `__post_init__` through an `InitVar` (not a
  field: `dataclasses.replace` rebuilds the arrays). On the `dates` and `derivatives`
  C++ backends, the interest too on `rates`; anything it does not take (a convention
  subclass, a negative lag, a non-finite factor) runs the Python path, which raises
  its errors. Bit for bit the Python path's legs (same dates, arrays, flags and
  shared lists), tested over leg types, lags, started legs and both rates backends.
- **Every pillar date of an instrument, read once** (`_keys._maturities`, one
  `_native.maturity_tables` call per build on the C++ builder): the planner and the
  seeder look up `{curve: pillar}` and the last date instead of recomputing them; the
  Python builder caches the Python tables the same way.
- **The fixed-leg seed** (`_native.fixed_leg_wealth`): the compounded wealth factor of
  `_Seeder.fixed_leg_zero`, the same left-to-right product (the log stays in Python).

Left in Python: grouping, ownership, readiness and cycle merging (~1 ms of small
dict / set / hash work; a C++ planner would save part of that for a large port),
the quote objects and the leg and swap dataclasses themselves (~26 µs an IRS now),
the redundant-quote warning (it reprices bumped quotes), the repricing check.

Results (notebook quotes, C++ valuation, 3 warm-ups + 10 builds, 2 slowest dropped;
the Python builder gains too, from the C++ legs and the cached tables):

| Interpolation | Python builder | C++ builder | Gain | Max \|dfwd\| | C++ builder in phase 2 |
|---|---:|---:|---:|---:|---:|
| PiecewiseConstant | 9.5 ± 0.6 ms | 6.5 ± 0.4 ms | 1.46x | 0 | 9.1 ms |
| PiecewiseLinear | 10.3 ± 0.9 ms | 7.0 ± 0.1 ms | 1.47x | 0 | 9.9 ms |
| CubicSpline | 15.1 ± 0.6 ms | 8.2 ± 0.1 ms | 1.84x | 8.9e-15 | 11.1 ms |
| HermiteCubicSpline | 17.5 ± 0.6 ms | 9.0 ± 0.4 ms | 1.96x | 1.1e-14 | 11.4 ms |
| ConvexMonotone | 17.6 ± 0.6 ms | 10.2 ± 1.0 ms | 1.73x | 3.5e-15 | 12.1 ms |

Phase 3 changes no number a build computes: the forwards and fitted MTMs are phase 2's.
Against the branch's baseline (`cpp-backend` at 36dc02f, C++ valuation: 12.1, 14.1,
18.3, 20.0, 20.0 ms) a C++ build is now 1.9-2.2x faster.

## Decisions (2026-11-16)

- **Not bit-identical.** The C++ builder may differ from the Python one; what matters
  is that its forward rates stay within **1e-8** of the Python builder's, on every
  curve and method. The total valuation error is reported, not bounded.
- **Phases 0 → 1 → 2** first (below); phase 3 (planning and seeding) later, if worth it.
- **Backend.** `curve_builder` is a backend subpackage like the others:
  `settings.set_backend("cpp")` switches it too, and
  `settings.set_backend("python", "curve_builder")` afterwards keeps the Python builder.

## Plan

0. **Verification harness and determinism (Python)** (done). Builds deterministic whatever
   `PYTHONHASHSEED`; a `curve_builder` backend subpackage; a benchmark script and
   tests comparing the two builders on the three criteria.
1. **Instruments from quotes in C++** (`_instrument_of`): done, see Phase 1.
2. **The solver in C++**: curve cores and a compiled residual plan held in C++, block
   Newton and the joint solve with the analytic Jacobian, no Python per iteration:
   done, see Phase 2.
3. **Planning and seeding**: the instruments' legs, pillar tables and fixed-leg
   seeds in C++: done, see Phase 3. A C++ planner (grouping, ownership, cycles) is
   left: ~1 ms a build.
