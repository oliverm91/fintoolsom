# Branch `cpp-curve-builder`

Branched from `cpp-backend` at `36dc02f` (2026-11-16). Goal: port the whole
`curve_builder` to the C++23 backend (`fintoolsom._native`), so an all-C++ build runs
from the quotes to the final curves in C++, with the Python builder kept as the
reference and as the default.

Status: **not started**. This file is the plan and the acceptance criteria; it is
updated as the port lands.

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
4. The whole suite passes in the six backend configurations (python;
   `FINTOOLSOM_BACKEND=cpp`; `FINTOOLSOM_BACKEND_{RATES,MARKET,DERIVATIVES,DATES}=cpp`),
   run with `pytest -n auto`; the errors and warnings raised are the Python builder's.

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

## Plan

To be agreed before starting.
