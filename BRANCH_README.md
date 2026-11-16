# Branch `cpp-backend`: C++ kernels for fintoolsom

This branch adds a C++23 implementation of fintoolsom's numerical cores, bound to
Python with [nanobind](https://github.com/wjakob/nanobind). **Ported:**

- `rates`: the curves (the `PillarCurve` family: `IndexCurve` and `DiscountCurve`),
  `Rate` factors (`get_wealth_factor`, `get_discount_factor`, `get_accrued_interest`),
  curve year fractions and the knot checks.
- `derivatives`: swap valuation (`Swap`, `SimplifiedSwap`, `NonStartedSimplifiedSwap`:
  leg PVs, leg flows, the par shortcut and its eligibility check, non-deliverable FX),
  leg duration and DV01, FX forwards and NDFs (MtM, projected settlement FX) and
  options (Garman-Kohlhagen price and greeks, strike from delta).
- `market`: the lookups of a valuation (curves, FX rates, FX known-until), done in C++
  on the Python `Market` containers, so a swap or forward is valued in one C++ call,
  realised fixings of started coupons included; `Market.get_dfs`,
  `_future_fx_rates_at_ordinals`, `get_fx_known_until` and the rate-accrual loops; the
  gap filling and level compounding of the overnight index histories; the UF
  extension (`extend_with_cpi`). The volatility surface stays Python.
- `dates`: calendars (`is_holiday`, `add_business_days`), the business-day conventions,
  `Tenor`, `Term`, schedule generation, day counts, time fractions, `date_ordinals` and,
  through the calendars, the CPI/UF dates. The holiday rules (`HolidayRule`) stay
  Python: they only feed holidays into the `holidays` objects.

The Python code stays the reference, and each port must give the same results.

## Layout

| Path | What it holds |
|------|---------------|
| `CMakeLists.txt` | Builds the extension `fintoolsom._native` (C++23, nanobind, static libnanobind). |
| `cpp/src/bindings.cpp` | The nanobind module: `PillarCurveCore`, numpy in/out. |
| `cpp/src/curves/pillar_curve.*` | `PillarCurveCore`: day offsets to discount factors, and the daily grid. |
| `cpp/src/curves/fits.*` | Ports of the scipy fits the Python curves use (CubicSpline natural, PCHIP, the `k=1` B-spline antiderivative, convex monotone). |
| `cpp/src/curves/piecewise_poly.hpp` | Piecewise polynomial evaluated as scipy `PPoly` does. |
| `cpp/src/swaps/swap_leg.*` | `SwapLegCore`: a leg's coupons as flat arrays and its valuation arithmetic; `fsum`, CPython's `math.fsum`. |
| `cpp/src/bind_swaps.cpp`, `cpp/src/py_curve.hpp` | Bindings of `SwapLegCore`; discount factors read from a curve's C++ core, or from Python. |
| `src/fintoolsom/derivatives/calculator/_native_swap_pv.py` | The Python side of the C++ swap valuation: market lookups and realised fixings. |
| `cpp/src/fx/fx_derivatives.hpp`, `cpp/src/bind_fx.cpp` | Forward, NDF and option kernels and their bindings. |
| `cpp/src/fx/special.hpp` | `ndtr` (libm `erfc`) and `ndtri` (Cephes), as `_normal.py` computes them. |
| `cpp/src/discount.hpp` | `DfSource` and `Discount` (`a`, or `a * b / c`), shared by swaps and FX. |
| `src/fintoolsom/derivatives/calculator/_native_fx.py` | The Python side of the C++ FX valuation: spot, fixings, curves, surface, conventions. |
| `src/fintoolsom/derivatives/calculator/_normal.py` | Pure-Python normal cdf and inverse, the reference for both backends. |
| `cpp/src/bind_market.cpp` | `MarketView`: the Market lookups through the C API (realised fixings included), whole swap / simplified swap / forward / NDF valuations, leg duration, the Market methods and accrual loops. |
| `cpp/src/bind_histories.cpp` | The day walks of the index histories (gap filling, level compounding, rescaling) and the UF extension, on the Python dicts in place. |
| `cpp/src/rates/interest.hpp`, `cpp/src/bind_rates.cpp`, `cpp/src/py_dates.hpp` | Interest conventions and `Rate` factors, year fractions, knot checks; dates and `Rate`s read from Python. |
| `cpp/src/dates/schedule.hpp`, `cpp/src/bind_dates.cpp` | `CalendarCore` (a calendar's days, loaded from Python in one call), the adjustment conventions, `Tenor` and `generate_schedule`. |
| `cpp/src/dates/day_counts.hpp` | Day counts and time fractions. |
| `cpp/benchmarks/bench_curves.py` | Python vs C++ benchmarks (below). |
| `src/fintoolsom/settings.py` | Chooses the backend per subpackage. |
| `tests/unit/rates/test_cpp_parity.py` | Parity of both curve backends. |
| `tests/unit/derivatives/test_cpp_swap_parity.py` | Parity of both swap valuation backends. |
| `tests/unit/derivatives/test_cpp_fx_parity.py` | Parity of both forward, NDF and option backends. |
| `tests/unit/market/test_cpp_market_parity.py` | Parity of whole valuations on Python and C++ markets, errors included. |
| `tests/unit/market/test_cpp_market_methods_parity.py` | Parity of realised fixings, duration and DV01, the par check, the Market methods and accrual loops, the index histories and the UF extension. |
| `tests/unit/rates/test_cpp_rates_parity.py` | Parity of `Rate` factors (result types included), year fractions and knot checks. |
| `tests/unit/rates/test_set_forward_rates.py` | Forwards updated in place give the curve built on them; the solver builds each curve once per solve. |
| `tests/unit/curve_builder/test_block_newton.py` | The sweep blocks' Newton solver: small linear solves, convergence and reuse of a block's Jacobian, giving up (non-decreasing or non-finite residuals); sweeps solve by Newton, match hybr's curves, fall back to hybr, and keep the backends bit-identical. |
| `tests/unit/curve_builder/test_jacobian.py` | The solver's analytic Jacobian matches central differences of the residuals on both backends; it solves the same curves as hybr's differencing, keeps the backends bit-identical, and is left to hybr when an instrument is not compiled. |
| `tests/unit/derivatives/test_valuation_batch.py` | A `ValuationBatch` gives `Calculator.valuate`'s values and errors for every swap case, simplified swap, forward and NDF, on both backends, as the curves move; the solver values through batches. |
| `tests/unit/dates/test_cpp_schedule_parity.py` | Parity of calendars, conventions and schedules. |
| `tests/unit/dates/test_cpp_dates_parity.py` | Parity of day counts, time fractions, ordinals, tenors, terms and CPI/UF dates; the one-call holiday load. |
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

settings.set_backend("cpp")              # every subpackage (dates, rates, market, derivatives)
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
when the derived curve is made. A swap leg reads the `derivatives` setting the same
way. Curves and legs with a C++ core can be copied and pickled: the core is rebuilt
on loading.

The settings are independent, and any combination gives the same results. The fast
one is `cpp` for all. With C++ swaps on Python curves (`rates=python`,
`derivatives=cpp`), the C++ code reads every discount factor through the curve's Python
`_dfs_at_ordinals`: no faster than Python, so use it only to debug one side. A `Market`
reads the `market` setting when created; its lookups run in C++ only for an
instrument with a C++ core (`derivatives=cpp`), since Python arithmetic does its own
lookups. Everything in `dates` reads its setting when called.

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

## How a swap uses the C++ core

A leg built under the `cpp` backend of `derivatives` gets a `SwapLegCore` on its first
valuation. The core holds the leg's per-coupon data as contiguous arrays (start, end,
payment and fixing ordinals, residuals, flows, spreads) plus the par flags.

The calculator's functions stay the entry points:

- `_leg_pv`, `_leg_pv_non_deliverable` and `_leg_flows`
- `_simplified_leg_pv` and `_non_started_leg_pv`

Each one routes a C++ leg to `_native_swap_pv` (on a Python market; on a C++ market the
whole valuation is one C++ call, below). That module does, in Python, what reads the
market:

- the projection and discount curves, looked up in the same order as the Python path,
  so a missing curve raises the same error;
- the realised fixings of started coupons (running overnight coupons, set term-rate
  coupons), from the index history;
- the known FX fixings of non-deliverable flows.

C++ then does the rest in one call:

- the projected accruals and flows;
- the discount factors: `a`, or `a * b / c` for collateral in another currency;
- the FX projection of later fixings on the funding curves;
- the par shortcut;
- the sums.

When a curve has a C++ core, C++ reads its discount factors from the core, without
going through Python.

Curve building rarely touches the history. Its swaps haven't started, so their whole
valuation runs in C++.

## How forwards and options use the C++ core

A `Forward`, `NDF`, `Call` or `Put` reads the `derivatives` backend when it is created
(`_native`); `get_strike_from_delta` reads it when it is called. The calculator
functions in `_forwards` and `_options` stay the entry points and route to
`_native_fx`, which reads the market in Python in the same order as the Python path:

- spot, the realised or published FX fixing of an NDF, and the reporting-currency rate;
- the discount curves (`a`, or `a`, `b`, `c` for collateral in another currency), or
  for an option the two funding curves, whose dfs feed the rate convention;
- an option's volatility, from the pair's surface.

C++ then computes the discount factors (from a curve's core when it has one), an NDF's
settlement FX projected on the funding curves, and the price, greeks or strike.

## How the market and the schedules use C++

**Market.** The Python `Market` keeps its data: `curves`, `fx_history` and each
`FX_RateData.historic_data` are plain dicts that callers (the curve builder, tests,
users) mutate directly, so a C++ copy would go stale. `MarketView` reads them where
they stand, through the C API: a dict lookup per curve or FX rate, an `isinstance`
check for index curves, `CurrencyPair` keys cached per pair of currency names. Each
lookup is the Python method's, in the same order and with the same errors (`KeyError`,
`ValueError` and their messages). On a C++ market, `get_swap_mtm`, `get_leg_pv`, the
simplified and non-started swap MtMs and the forward and NDF MtMs hand the whole
valuation to C++: settlement currency, leg PVs, conversion at spot. Realised fixings
of started coupons are read in C++ too: the level dict of an `OvernightRateHistory` or
`OvernightInterestPriceHistory`, the fixings of a `TermRateHistory` and their `Rate`
conventions; the projection over `[t, end]` of a running coupon is the curve's own
`get_accrual`. Another history class, a level that isn't a float or a rate convention
not ported falls back to the Python callback. `get_leg_duration` runs the same way, and
so do `Market.get_dfs` / `_dfs_at_ordinals`, `_future_fx_rates_at_ordinals`,
`get_fx_known_until` and the loops of `accrue_rates_reset_business_days` and
`accrue_rates_custom_reset_days` (their argument checks stay Python).

**Index histories and the UF.** With the `market` and `dates` backends on C++, the day
walks of `OvernightRateHistory` (`_fill_gaps`, `_compound_levels`, the rescaling of a
replaced fixing) and `OvernightInterestPriceHistory._check_for_gaps` run in C++ on the
index calendar's core, writing into the Python dicts in Python's order. Where a step
needs Python (an exponential or numpy-valued rate, a level that isn't a float, a
result where Python raises), C++ stops before it and the Python loop carries on, so
the result is the same. `extend_with_cpi` fills its period in C++, with `round(x, 2)`
reproduced as the correctly rounded 2-decimal string read back.

**Dates.** A `Calendar` gets a `CalendarCore` on first use. The core asks Python once
for the flags of every day from 5 years before the first date it sees to 65 after:
holiday of the `holidays` object, its weekend days (which can vary by date in
`holidays`) and weekend working days, as one `bytes` built with numpy. A date outside
that window extends it by at least 20 years, again in one call, so a typical session
crosses into Python once per calendar. The neighbouring years are populated too, so an
observed 31 December added with the next year is there. Populating the `holidays`
object is the cost (about 1 ms a year for CL, 0.4 ms for GB, 2 ms for JP), paid once
per calendar; the calendars are shared per country. `Calendar.is_holiday` (a holiday, or Saturday/Sunday) and
`add_business_days` (`HolidayBase.get_nth_working_day`) are reproduced exactly, as are
the four adjustment conventions and month arithmetic (`relativedelta`'s clamping to
the month's end). `add_custom_holiday` drops the core. Day counts, time fractions and
`date_ordinals` are integer arithmetic on ordinals with the Python divisions (the same
doubles). Inputs off the plain path (`datetime`s or other non-`date` dates, a tenor that
isn't `<n><unit>`, a zero frequency or base, a non-int frequency or period count, a
subclassed convention, calendar, day count or time fraction, empty lists, a schedule
that rolls past its iteration limit) go to the Python code, which then raises its own
errors.

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

Swap valuation agrees **to the bit on every platform**. The parity test covers all 1,140
factory swap cases on markets of Python curves and of C++ curves: leg PVs, swap MtMs,
leg flows, and the simplified and non-started variants. The arithmetic is IEEE `+ * /`
with no libm call.

To make that possible, the Python reference sums PVs with `math.fsum` instead of
`np.dot`/`np.sum`. BLAS kernels pick their own summation order and may use fused
multiply-adds, so their result can't be reproduced. The correctly rounded sum is unique,
and `fsum` in `swap_leg.cpp` is CPython's algorithm, special values included. The change
moves Python PVs by at most an ulp and costs no measurable time.

Rate factors, the accrual loops and the history walks agree to the bit on Windows; off
Windows an exponential rate may differ by a few ulps (Python's is `np.exp`), and the
tests allow 4. Duration, the par check and realised fixings agree to the bit: duration
now sums with `math.fsum` on both backends, as the PVs do.

Forwards, NDFs and options also agree to the bit: 764 parity cases (forwards and NDFs
over every collateral, fixing and payment date and reporting currency; option prices,
greeks, log-moneyness and strikes over maturities, moneyness, vols and conventions;
options valued from a surface). Here both sides call libm (`exp`, `log`, `sqrt`,
`erfc`) through CPython's `math` and C++'s `<cmath>`, the same functions in-process.
scipy's `ndtr` / `ndtri` could not be matched: its compiled Cephes kernels round
differently from a strict-IEEE build of the same code (up to 3 ulps on Windows). The
Python reference therefore uses `_normal.py`: `ndtr(x) = 0.5 * math.erfc(-x / sqrt(2))`
and a pure-Python port of the Cephes `ndtri`, which C++ mirrors. Against scipy they
differ by at most 6e-14 relative (ndtr, far tail) and 6e-16 (ndtri).

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

### Swap valuation

Each figure is the median time of one `Calculator.valuate` call on a swap from the test
factory, valued on its market. A "[NonStarted]" row is the same swap as a
`NonStartedSimplifiedSwap`.

| Swap | Python | C++ | Speed-up |
|---|---:|---:|---:|
| IRS USD OIS, SOFR CSA (par) | 10.8 µs | 6.0 µs | 1.8× |
| ↳ [NonStarted] | 5.9 µs | 4.0 µs | 1.5× |
| IRS CLP ICP, uncollateralised (par) | 10.7 µs | 5.6 µs | 1.9× |
| IRS CLP OIS, SOFR CSA (three-curve discount) | 26.1 µs | 8.7 µs | 3.0× |
| ↳ [NonStarted] | 15.2 µs | 5.9 µs | 2.6× |
| IRS USD OIS, started (realised fixings, now read in C++; a later run) | 27.6 µs | 13.7 µs | 2.0× |
| CCS deliverable | 13.2 µs | 6.7 µs | 2.0× |
| CCS non-deliverable | 55.2 µs | 23.1 µs | 2.4× |

Both columns use C++ curves. The C++ arithmetic is now a small part of each call. What
remains is the calculator's Python layers (swap, leg, settlement currency, conversion
at spot) and the market lookups, about 2–3 µs per leg. Moving those layers into C++
would mean porting `market`.

### FX forwards, NDFs and options

Median time of one call. "Call, from surface" is `Calculator.valuate` on a USD/CLP
call, volatility read from a `VannaVolga` surface.

| Instrument | Python curves: Python | C++ | C++ curves: Python | C++ | Speed-up |
|---|---:|---:|---:|---:|---:|
| Forward, uncollateralised | 8.9 µs | 9.7 µs | 5.4 µs | 3.2 µs | 1.7× |
| Forward, SOFR CSA | 14.7 µs | 16.0 µs | 7.9 µs | 3.8 µs | 2.1× |
| NDF, projected fixing | 40.3 µs | 24.8 µs | 35.4 µs | 14.5 µs | 2.4× |
| Call, from surface | 54.7 µs | 52.4 µs | 53.1 µs | 51.8 µs | 1.0× |
| Call delta | 4.1 µs | 2.5 µs | 4.2 µs | 2.6 µs | 1.6× |
| Strike from 25 delta | 2.5 µs | 2.2 µs | 2.5 µs | 2.2 µs | 1.1× |

The speed-up column is on C++ curves. On Python curves a C++ forward is slightly
slower: its two or three discount factors come back through Python callbacks. An
option's valuation is almost all volatility-surface interpolation, which is `market`
and stays Python.

### Curve building

The test builds every curve from `notebooks/quotes.json` with `build_curves`: 3
warm-ups, then 10 timed builds, dropping the 2 slowest, reported as mean ± std. It
compares three configurations: all Python, C++ curves only, and C++ curves and swaps.
All three build identical curves (the largest relative df difference is 0).

| Interpolation | Python | C++ curves | C++ curves + swaps | Speed-up |
|---|---:|---:|---:|---:|
| PiecewiseConstant | 60.4 ± 2.2 ms | 44.5 ± 1.1 ms | 37.9 ± 0.4 ms | 1.59× |
| CubicSpline | 312.2 ± 3.1 ms | 172.1 ± 1.4 ms | 108.5 ± 1.6 ms | 2.88× |
| ConvexMonotone | 338.4 ± 3.1 ms | 171.0 ± 1.4 ms | 107.9 ± 1.3 ms | 3.14× |

With simplified swaps: the same build, with the builder valuing every swap as a
`NonStartedSimplifiedSwap` (`simplified_swaps` in the benchmark), all Python vs all C++
(rates and derivatives). All four configurations build identical curves.

| Interpolation | Python | Python simplified | C++ | C++ simplified |
|---|---:|---:|---:|---:|
| PiecewiseConstant | 65.5 ± 1.2 ms | 60.6 ± 0.9 ms (1.08×) | 40.6 ± 1.3 ms (1.61×) | 40.7 ± 1.3 ms (1.61×) |
| CubicSpline | 334.5 ± 4.2 ms | 280.8 ± 4.2 ms (1.19×) | 107.2 ± 1.4 ms (3.12×) | 93.0 ± 1.2 ms (3.60×) |

With the market and dates ports, each port added in turn (a later run, on a slower
moment of the machine: compare the speed-ups, each against its own Python row). All configurations build identical curves.

| Interpolation | Python | C++ rates + derivatives | + market | + dates (all C++) | all C++, simplified |
|---|---:|---:|---:|---:|---:|
| PiecewiseConstant | 61.4 ± 1.0 ms | 38.9 ± 0.9 ms (1.58×) | 35.4 ± 0.7 ms (1.73×) | 22.7 ± 0.8 ms (2.70×) | 23.6 ± 1.2 ms (2.60×) |
| CubicSpline | 328.1 ± 4.2 ms | 104.4 ± 1.7 ms (3.14×) | 72.1 ± 1.2 ms (4.55×) | 59.8 ± 0.9 ms (5.48×) | 57.3 ± 1.2 ms (5.72×) |
| ConvexMonotone | 341.4 ± 3.7 ms | 107.4 ± 4.3 ms (3.18×) | 72.5 ± 1.4 ms (4.71×) | 60.0 ± 1.5 ms (5.69×) | 58.3 ± 1.3 ms (5.85×) |

With the rest of `rates`, `derivatives` and `market` ported (Rate factors, year
fractions and knot checks, the par check, realised fixings, histories; a later run):

| Interpolation | Python | C++ rates + derivatives | + market | + dates (all C++) | all C++, simplified |
|---|---:|---:|---:|---:|---:|
| PiecewiseConstant | 56.7 ± 0.8 ms | 28.4 ± 0.1 ms (1.99×) | 26.4 ± 0.3 ms (2.15×) | 15.7 ± 0.2 ms (3.62×) | 15.9 ± 0.2 ms (3.56×) |
| CubicSpline | 288.6 ± 1.6 ms | 85.1 ± 0.9 ms (3.39×) | 57.0 ± 0.5 ms (5.06×) | 46.4 ± 0.2 ms (6.22×) | 45.6 ± 0.3 ms (6.33×) |
| ConvexMonotone | 316.4 ± 1.6 ms | 85.4 ± 0.9 ms (3.71×) | 57.6 ± 0.2 ms (5.49×) | 47.9 ± 1.5 ms (6.60×) | 48.1 ± 0.6 ms (6.58×) |

Then without the daily grid while solving C++ spline curves (below; a later run):

| Interpolation | Python | C++ rates + derivatives | + market | + dates (all C++) | all C++, simplified |
|---|---:|---:|---:|---:|---:|
| PiecewiseConstant | 55.4 ± 0.5 ms | 28.7 ± 0.2 ms (1.93×) | 26.3 ± 0.3 ms (2.11×) | 15.6 ± 0.1 ms (3.54×) | 15.9 ± 0.1 ms (3.48×) |
| CubicSpline | 283.7 ± 1.2 ms | 72.0 ± 0.2 ms (3.94×) | 44.7 ± 0.2 ms (6.35×) | 34.0 ± 0.1 ms (8.34×) | 32.7 ± 0.2 ms (8.68×) |
| ConvexMonotone | 309.4 ± 1.5 ms | 72.5 ± 0.4 ms (4.27×) | 45.4 ± 0.1 ms (6.82×) | 34.8 ± 0.2 ms (8.88×) | 33.5 ± 0.2 ms (9.24×) |

**The grid while solving.** The solver rebuilds its curves on every iteration. A
Python spline curve gets its daily grid on each rebuild: scipy's evaluation is slow
per call, so the grid pays for itself (without it the Python spline builds take
1.8x longer). A C++ curve evaluates its formula as fast as it reads a grid, so the
grid filled on each rebuild was wasted, about a quarter of the build: the solver now
skips it on the "cpp" backend of "rates" and the builder adds it to the final curves,
which are identical.

**Forwards updated in place.** The solver used to construct new curves on every
iteration. Now each solve builds its curves once and then sets the new forwards on
them (`PillarCurve._set_forward_rates`; on C++ curves `PillarCurveCore.set_forwards`
refits in place with the constructor's arithmetic). The curves are bit-identical; a
curve the solve did not build (its rollback snapshot) is never changed. Before and
after, same benchmark (3 warm-ups, 10 builds, 2 slowest dropped):

| Interpolation | Backend | New curves each iteration | In place | Gain |
|---|---|---:|---:|---:|
| PiecewiseConstant | Python | 54.8 ± 0.5 ms | 51.5 ± 0.2 ms | 6% |
| PiecewiseConstant | C++ | 15.6 ± 0.2 ms | 13.5 ± 0.1 ms | 14% |
| CubicSpline | Python | 291.9 ± 5.1 ms | 273.3 ± 0.6 ms | 6% |
| CubicSpline | C++ | 35.0 ± 0.7 ms | 32.5 ± 0.2 ms | 7% |

**Batch valuation.** The solver's residuals come from a `ValuationBatch`
(`derivatives/calculator/_batch.py`, also `Calculator.valuate_many`), compiled once per
list of instruments (the whole solve, and each date block of a sweep) and called on
every iteration. Each value is, to the bit, `Calculator.valuate`'s for that instrument
alone:

- Instruments valued in one C++ market call go to `market_valuate_many`: one call and
  one `MarketView` for the batch. The view remembers the curves, known-until dates and
  spot rates it has looked up, so each is looked up once per call rather than once per
  instrument. A batch of 68 swaps and NDFs takes 66 µs instead of 159 µs.
- Python swaps (`Swap` itself) and NDFs are compiled. The parts that depend only on the
  market's date, histories and risk-free index are computed once: the unpaid coupons,
  the par-leg decision, fixed flows, settlement and spot rates. Each call then evaluates
  every curve source (a discounting, an index curve's dfs or accruals, an FX pair) once,
  on all the dates the batch needs from it. The flows, FX and dfs are multiplied
  elementwise and each leg summed with `math.fsum` on its slice, which is the
  calculator's arithmetic. For 68 instruments a call takes 196 µs instead of
  1,419 µs.
- The rest goes to `Calculator.valuate`: other instrument types, coupons already
  running or fixed, and C++ legs on a Python market. If a call raises, it is redone
  instrument by instrument, so the error is the calculator's.

Same benchmark, before and after (the curves are bit-identical within a process):

| Interpolation | Backend | One by one | Batch | Gain |
|---|---|---:|---:|---:|
| PiecewiseConstant | Python | 51.5 ± 0.2 ms | 45.8 ± 0.6 ms | 1.12x |
| PiecewiseConstant | C++ | 13.5 ± 0.1 ms | 13.5 ± 0.1 ms | 1.00x |
| CubicSpline | Python | 273.3 ± 0.6 ms | 128.1 ± 0.9 ms | 2.13x |
| CubicSpline | C++ | 32.5 ± 0.2 ms | 22.3 ± 0.3 ms | 1.46x |

PiecewiseConstant gains little because its sweep solves blocks of 1 to 3 instruments,
each valued about 6 times, so compiling a batch costs about as much as it saves. A
build's curves depend on `PYTHONHASHSEED`: they can land on one of two results about
4e-15 apart, with or without the batch, because some set or dict of strings in the
builder is iterated in hash order. So compare curves from two processes with a fixed
seed.

**Analytic Jacobian.** A joint solve hands scipy's `hybr` the Jacobian of its residuals
(`_solver._Jacobian`) instead of letting it difference them, which cost one residual
evaluation per unknown each time hybr rebuilt the Jacobian. It is a chain rule through
the values a batch reads off the market, `dr/dy = (dr/dv)(dv/dy)`:

- `dr/dv` is analytic (`ValuationBatch.gradient`): each MTM is a sum of products of the
  values (`flow * fx * df`, `residual_0 * df` for a par head, `a * (s_t - strike) * df *
  spot` for an NDF) with constant coefficients. A C++ batch is compiled again in Python
  for it (`compile_all=True`).
- Every value is a product of curve dfs to the powers +-1 (collateral `D * I / D`, FX
  `anchor * D_b / D_q` from the known-until date, `1 + accrual = I(s) / I(e)`), so `dv/dy`
  is a signed sum of `d log df / dy` of the curves solved. That is read off a private copy
  of each curve, bumping one forward at a time, at the dates the batch reads. On
  piecewise-constant/linear and spline curves `log df` is affine in the forwards, so it
  is computed once per solve and the bump is exact up to rounding. On the monotone
  methods it is a forward difference (1e-7) recomputed at each call.

It matches central differences of the residuals to about 4e-10 relative on splines
(1e-6 on ConvexMonotone, whose curve part is differenced). The date blocks of a
piecewise-constant/linear sweep keep hybr's differencing: with 1 to 3 unknowns it costs
1 to 3 evaluations, less than setting up a Jacobian.

On the notebook quotes (3 warm-ups, 10 builds, 2 slowest dropped, `PYTHONHASHSEED=0`),
a CubicSpline build goes from 165 residual evaluations to 29 plus 5 Jacobians. The
forwards differ from the differenced solve's by at most 9e-15 (9e-11 bp), the largest
repricing error stays below 1e-10, and Python and C++ builds stay bit-identical:

| Interpolation | Backend | Differenced | Analytic | Gain |
|---|---|---:|---:|---:|
| CubicSpline | Python | 127.6 ± 0.8 ms | 65.9 ± 0.5 ms | 1.94x |
| CubicSpline | C++ | 22.4 ± 0.2 ms | 18.3 ± 0.8 ms | 1.22x |
| HermiteCubicSpline | Python | 142.9 ± 0.5 ms | 89.3 ± 0.4 ms | 1.60x |
| HermiteCubicSpline | C++ | 24.5 ± 0.1 ms | 20.0 ± 0.3 ms | 1.22x |
| ConvexMonotone | Python | 157.3 ± 1.0 ms | 113.3 ± 0.8 ms | 1.39x |
| ConvexMonotone | C++ | 23.3 ± 0.2 ms | 20.0 ± 0.3 ms | 1.17x |

PiecewiseConstant and PiecewiseLinear are unchanged (their sweeps converge here
without a joint solve). On C++ most of what is left of the Jacobian's cost is
compiling the Python plan (about 1 ms per solve).

**Block Newton.** The date blocks of a piecewise-constant/linear sweep (1 or 2
unknowns here, 55 per build) are solved by `_solver._newton` instead of hybr. The
Jacobian is a forward difference at the seed (1e-7), or the block's own from the
previous sweep, and each step updates it by Broyden's rank-one formula, so a step
costs one evaluation. A step below 1e-12 is taken without evaluating it. If the steps
stop lowering the residuals, or a residual is not finite or raises, the block goes to
hybr from its seed. The algebra is on Python floats: numpy's per-call cost (about 11 µs
for `linalg.solve`) exceeded a C++ evaluation's and cancelled the gain in a first
version. hybr took 5 to 7 evaluations per block, even 5 for a block already solved by
the previous sweep, because it stops on its step size alone.

Same benchmark (`PYTHONHASHSEED=0`): 237 evaluations per PiecewiseConstant build
instead of 363 (240 instead of 370 for PiecewiseLinear), no hybr fallback. The
forwards move by at most 2.2e-12 (2e-8 bp), the largest repricing error drops from
3.2e-10 to 2.9e-11 (2.4e-9 to 1.5e-11 for PiecewiseLinear), and Python and C++
builds stay bit-identical:

| Interpolation | Backend | hybr | Newton | Gain |
|---|---|---:|---:|---:|
| PiecewiseConstant | Python | 46.3 ± 0.5 ms | 40.6 ± 0.3 ms | 1.14x |
| PiecewiseConstant | C++ | 13.6 ± 0.2 ms | 12.1 ± 0.1 ms | 1.12x |
| PiecewiseLinear | Python | 91.3 ± 1.0 ms | 72.3 ± 2.0 ms | 1.26x |
| PiecewiseLinear | C++ | 14.8 ± 0.2 ms | 14.1 ± 1.4 ms | 1.05x |

A C++ evaluation of a block costs about 30 µs, so the 126 evaluations saved are about
4 ms of a 13.6 ms build; the rest of a C++ piecewise build is mostly building the
instruments from the quotes. The joint solves keep hybr with the analytic Jacobian:
5 calls per build, coupled unknowns and, for the monotone methods, kinks where hybr's
trust region matters.

The market port takes the per-valuation Python lookups out of the solver loop (most of
a fitted build); the dates port takes schedules, conventions, tenors, day counts and
`date_ordinals` out of building the instruments from the quotes (most of what was
left of a PiecewiseConstant build).
Simplified swaps no longer matter on C++: what they saved was Python glue. What
remains is mostly Python: building the leg and coupon objects, the solver's own loop
(scipy `hybr` on the joint solves, Newton on the sweep blocks) and, on Python curves,
the refits.

## Not ported yet / caveats

- Not ported, by design: the volatility surface (calibration, interpolation) and its
  vectorised `ndtri` (scipy); `ZeroCouponCurve` and the legacy curves (deprecated; C++
  swaps read them through `_dfs_at_ordinals`); the data and declarative classes
  (quotes, conventions, currencies, localities, indexes, `Market` construction and
  setters); building leg, coupon and instrument objects; one-line methods
  (`get_rate`, `get_index`, `convert_value_currency`, `UFConvention.reajuste`,
  `implied_monthly_cpi`), where a C++ call would cost as much as the Python.
- `Market` methods route to C++ on a market created under the `market` backend;
  `Rate` factors on the `rates` backend at call time; the history walks on the
  `market` and `dates` backends at call time.
- The core copies a curve's knots and forwards when the curve is built. Mutating
  `curve.forward_rates` in place afterwards is not seen by the core. That was never
  supported anyway, because the Python fit is also built once.
- `forward_integral` is exposed on the core for tests and debugging only.
- Options valued on a C++ market still resolve their curves, spot and volatility in
  Python (`_native_fx`): the volatility surface dominates their cost.
- A `CalendarCore` caches the years it loaded. Changing a calendar's holidays other than
  through `add_custom_holiday` (e.g. on its private `_calendar`) isn't seen by it: set
  `calendar._core = None` after such a change.
- The first C++ use of a calendar populates about 70 years of its `holidays` object
  (CL about 0.1 s), where Python populates only the years it touches.
- nanobind's leak report at exit is off: a shared calendar's core, held by a module
  global, can outlive the interpreter's last collection.
- Leg and coupon objects are still built in Python from the C++ schedule dates.
