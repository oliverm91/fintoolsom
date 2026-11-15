# Branch `cpp-backend`: C++ kernels for fintoolsom

This branch adds a C++23 implementation of fintoolsom's numerical cores, bound to
Python with [nanobind](https://github.com/wjakob/nanobind). The plan covers `rates`
(curves), `market`, and `derivatives`. **Only the `rates` curves are ported so far**
(the `PillarCurve` family: `IndexCurve` and `DiscountCurve`). The Python code stays the
reference, and each port must give the same results.

## Layout

| Path | What it holds |
|------|---------------|
| `CMakeLists.txt` | Builds the extension `fintoolsom._native` (C++23, nanobind, static libnanobind). |
| `cpp/src/bindings.cpp` | The nanobind module: `PillarCurveCore`, numpy in/out. |
| `cpp/src/curves/pillar_curve.*` | `PillarCurveCore`: day offsets to discount factors, and the daily grid. |
| `cpp/src/curves/fits.*` | Ports of the scipy fits the Python curves use (CubicSpline natural, PCHIP, the `k=1` B-spline antiderivative, convex monotone). |
| `cpp/src/curves/piecewise_poly.hpp` | Piecewise polynomial evaluated as scipy `PPoly` does. |
| `cpp/benchmarks/bench_curves.py` | Python vs C++ benchmarks (below). |
| `src/fintoolsom/settings.py` | Chooses the backend per subpackage. |
| `tests/unit/rates/test_cpp_parity.py` | Parity of both backends. |
| `tests/unit/test_settings.py` | The settings module. |

## Building

The package now builds with scikit-build-core instead of hatchling. CMake 4.2 or newer
is fetched automatically, and a C++23 compiler is required (MSVC 19.4x/VS 2026,
GCC ≥ 13, Clang ≥ 17, Apple Clang ≥ 15).

```sh
uv sync --group dev          # builds and installs the extension (rebuilt when cpp/ changes)
```

For local speed tests only, `-C cmake.define.FTS_NATIVE_ARCH=ON` adds `-march=native`
or `/arch:AVX2`. Distributed wheels never use it.

### Releases

`.github/workflows/publish.yml` builds wheels with cibuildwheel for CPython 3.11–3.14 on
these platforms:

- manylinux x86_64 and aarch64
- Windows AMD64
- macOS x86_64 and arm64 (macOS ≥ 11)

It also builds the sdist and publishes everything through trusted publishing. Each
wheel runs the rates and settings tests (including parity) before upload. Other
platforms, such as musllinux and 32-bit, install from the sdist and need a C++23
compiler. `ci.yml` now also runs on macOS and runs the suite a second time with
`FINTOOLSOM_BACKEND=cpp`.

## Choosing the backend

```python
from fintoolsom import settings

settings.set_backend("cpp")              # every ported subpackage (rates)
settings.set_backend("python", "rates")  # one subpackage
with settings.use_backend("python"):     # temporarily, e.g. to debug a build
    ...
```

The same settings are available as environment variables, read at import:

- `FINTOOLSOM_BACKEND=cpp` applies to every ported subpackage.
- `FINTOOLSOM_BACKEND_RATES=python` sets one subpackage and overrides the global value.

The default is `python`. Asking for `cpp` on a subpackage that isn't ported yet raises
`ValueError`. Asking for it without the compiled extension raises `ImportError`.

An object reads the backend when it is created. A curve built under `cpp` keeps its C++
core, and a curve derived from it (copy, bump, aged) reads the setting that is current
when the derived curve is made.

## How a curve uses the C++ core

`PillarCurve.__post_init__` validates the curve, then builds either the Python forward
integral or a `PillarCurveCore`, depending on the setting. `DailyGridCurve` routes the
following methods to the core when there is one:

- `_dfs_at_days`
- `_dfs_at_ordinals`
- `_dfs_from_days`
- `precalculate_dfs`

Dates, validation, ids, and the curve API stay in Python.

`IndexCurve`'s spot segment, days `[0, first knot)`, is handed to the core as a
precomputed table. Negative days on a curve with a spot rate need the spot rate's own
day count, so the core sends them back to a small Python closure. That closure
deliberately captures the rate and date, not the curve, to avoid reference cycles.

## Data layout

The layout is designed for cache use and SIMD:

- **Structure of arrays** for the per-knot state: `knot_tau`, `forwards` and `cum` are
  contiguous `double` arrays, and lookups binary-search `knot_tau`.
- **Array of 32-byte aligned records** (`Cubic { double c[4]; }`) for the polynomial
  pieces. One piece is one aligned load, and every coefficient an evaluation needs sits
  on the same cache line.
- **Grid filling by runs.** `precalculate` walks the grid one segment or knot interval
  at a time. Inside a run the coefficients are loop invariant, the polynomial degree is
  a template parameter, and the body has no branches, so the compiler vectorizes the
  arithmetic and `exp`. The PiecewiseLinear B-spline uses the same idea: de Boor's
  recursion unrolled for `k = 2` with the knots hoisted out of the loop. This made its
  grid 3× faster than evaluating it per day.
- **Zero-copy grid.** The grid lives in a `shared_ptr<const vector<double>>`. Python
  gets a read-only numpy view whose capsule owns a reference, so copying a curve
  doesn't copy the grid. Writing to it raises an error.
- Lookups on the grid use one unsigned compare per day for the bounds check.

## Parity

The C++ code does each calculation the Python/scipy code does, with the same operations
in the same order:

- scipy `PPoly`'s power-sum evaluation, not Horner
- LAPACK `dgtsv`'s pivoting for the natural spline
- PCHIP's `_find_derivatives` and `_edge_case`
- `CubicHermiteSpline`'s coefficients
- `PPoly.antiderivative`'s continuity fix
- fitpack's `_deBoor_D` and `find_interval` for the linear-forward B-spline
- the convex monotone construction line by line

The build sets strict IEEE floating point: `/fp:precise` on MSVC, `-ffp-contract=off`
elsewhere, so there are no fused multiply-adds.

On Windows the two backends agree **to the bit** (0 ulps) for:

- every method (PiecewiseConstant, PiecewiseLinear, HermiteCubicSpline, CubicSpline,
  ConvexMonotone)
- every curve kind (DiscountCurve, IndexCurve, and IndexCurve with a spot rate)
- days −40 to 11 357
- formula and grid
- copies, bumps and aged curves
- the full curve build from the notebook quotes

The parity tests allow 4 ulps, because other platforms' libm `exp`/`log` may differ from
the one numpy uses by an ulp.

## Benchmarks

The numbers below come from `uv run python cpp/benchmarks/bench_curves.py` on Windows
11 with Python 3.11 and MSVC 14.50, using the generic wheel flags (no `-march`).

### Atomized operations

The test curve is an 18-knot IndexCurve with a spot rate, 30 years long. Each figure is
the median time per call.

| Method | Operation | Python | C++ | Speed-up |
|---|---|---:|---:|---:|
| PiecewiseConstant | build | 10.6 µs | 11.9 µs | 0.9× |
| | build + grid | 169 µs | 65 µs | 2.6× |
| | 10 dfs (formula) | 2.2 µs | 0.6 µs | 3.5× |
| | 10k dfs (formula) | 15.3 µs | 5.3 µs | 2.9× |
| | 10k dfs (grid) | 15.1 µs | 5.4 µs | 2.8× |
| PiecewiseLinear | build | 72.9 µs | 12.3 µs | 5.9× |
| | build + grid | 458 µs | 106 µs | 4.3× |
| HermiteCubicSpline | build | 121 µs | 12.8 µs | 9.5× |
| | build + grid | 394 µs | 145 µs | 2.7× |
| CubicSpline | build | 108 µs | 13.0 µs | 8.3× |
| | build + grid | 347 µs | 81 µs | 4.3× |
| ConvexMonotone | build | 184 µs | 14.9 µs | 12.3× |
| | build + grid | 418 µs | 81 µs | 5.1× |
| all methods | `get_dfs`, 10 dates | 2.8 µs | 1.9 µs | 1.4–1.5× |

The script prints the full table.

What the numbers show:

- Building the fitted curves is 6–12× faster.
- Lookups are about 3× faster, with grid or formula.
- PiecewiseConstant construction is dominated by the shared Python validation, so it
  breaks even.
- `get_dfs` is dominated by converting dates to ordinals in Python.

### Curve building

The test builds every curve from `notebooks/quotes.json` with `build_curves`: 3
warm-ups, then 10 timed builds, dropping the 2 slowest, reported as mean ± std. The two
backends build identical curves (the largest relative df difference is 0).

| Interpolation | Python | C++ | Speed-up |
|---|---:|---:|---:|
| PiecewiseConstant | 62.9 ± 1.1 ms | 42.7 ± 1.1 ms | 1.47× |
| CubicSpline | 359.7 ± 3.2 ms | 207.0 ± 1.3 ms | 1.74× |
| ConvexMonotone | 391.2 ± 4.0 ms | 210.5 ± 3.3 ms | 1.86× |

The rest of a build is swap valuation and the solver, which are still Python
(`derivatives`). That code is the next target.

## Not ported yet / caveats

- `market` and `derivatives` (swap legs and PV) are not ported. `ZeroCouponCurve` and
  the legacy curves are not ported either; they always use Python.
- The core copies a curve's knots and forwards when the curve is built. Mutating
  `curve.forward_rates` in place afterwards is not seen by the core. That was never
  supported anyway, because the Python fit is also built once.
- `forward_integral` is exposed on the core for tests and debugging only.
