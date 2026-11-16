"""Python vs C++ curve builder: timings, forward-rate differences, valuation error.

Run from the repository root, after ``uv sync --group dev``::

    uv run python cpp/benchmarks/bench_curve_builder.py [--quotes notebook|factory]

Both builders run on the C++ valuation backend (``settings.set_backend("cpp")``),
the reference with ``settings.set_backend("python", "curve_builder")`` on top. For
every interpolation method it reports the branch's acceptance criteria
(``BRANCH_BRANCH_README.md``):

1. timings: 3 warm-up builds and 10 timed ones, the 2 slowest dropped, mean +- std;
2. forward rates: max |forward C++ builder - forward Python builder| over every curve,
   against the 1e-8 limit;
3. total valuation error: every quote's instrument valued by ``Calculator.valuate`` on
   the final curves, in the builder's report currency: max |MTM|, RMS, sum |MTM| of
   the fitted quotes, and apart the ditched ones' (``RedundantQuoteWarning``: quotes
   owning no pillar, not fitted by design).

Exits with status 1 if a method's curves differ in their ids or by more than 1e-8, or
the builders ditch different quotes.
"""

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cpp" / "benchmarks"))
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

import bench_curves as nb  # noqa: E402

from fintoolsom import settings  # noqa: E402
from fintoolsom.curve_builder import RedundantQuoteWarning, build_curves  # noqa: E402
from fintoolsom.curve_builder._keys import _instrument_of  # noqa: E402
from fintoolsom.curve_builder.builder import _REPORT_CURRENCY  # noqa: E402
from fintoolsom.derivatives.calculator import Calculator  # noqa: E402

FORWARD_LIMIT = 1e-8
BUILDERS = ("python", "cpp")


def sources(kind):
    """``(quotes, market factory)`` of the notebook or the test factories."""
    if kind == "notebook":
        quotes = nb.build_quotes(nb.T, str(ROOT / "notebooks" / "quotes.json"))
        return quotes, nb.quote_market
    from tests.factories import quote_factory

    return quote_factory.load_quotes(), quote_factory.quote_market


def backend(builder):
    """Every subpackage on C++, the curve builder on `builder`."""
    settings.set_backend("cpp")
    settings.set_backend(builder, "curve_builder")


def build(quotes, new_market, method):
    """The built market and the ids of the quotes it ditched (``RedundantQuoteWarning``)."""
    market = new_market()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_curves(quotes, market, interp_method=method)
    ditched = {
        id(d[0])
        for w in caught
        if isinstance(w.message, RedundantQuoteWarning)
        for d in w.message.ditched
    }
    return market, ditched


def timed(quotes, new_market, method):
    times = []
    for i in range(13):
        market = new_market()
        t0 = time.perf_counter()
        build_curves(quotes, market, interp_method=method)
        if i >= 3:
            times.append(time.perf_counter() - t0)
    kept = sorted(times)[:8]
    return statistics.mean(kept) * 1e3, statistics.stdev(kept) * 1e3


def valuation_error(quotes, market, ditched):
    """|MTM| of every quote's instrument on the built curves: the fitted quotes', and
    the ditched ones' (not fitted, so not a solver error)."""
    mtm = {
        id(q): abs(
            Calculator.valuate(_instrument_of(q, market), market, _REPORT_CURRENCY)
        )
        for q in quotes
    }
    fitted = np.array([v for k, v in mtm.items() if k not in ditched])
    return fitted, np.array([v for k, v in mtm.items() if k in ditched])


def forward_difference(ref, other):
    """Max |forward difference| over the curves; inf if the curve ids differ."""
    if ref.curves.keys() != other.curves.keys():
        return float("inf")
    return max(
        float(np.abs(np.asarray(c.forward_rates) - other.curves[k].forward_rates).max())
        for k, c in ref.curves.items()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quotes", choices=("notebook", "factory"), default="notebook")
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args()
    quotes, new_market = sources(args.quotes)
    failed = False
    print(f"quotes: {args.quotes} ({len(quotes)}); report currency {_REPORT_CURRENCY}")
    print(
        f"{'method':18} {'python builder ms':>17} {'cpp builder ms':>17} {'gain':>6} "
        f"{'max |dfwd|':>10} {'ok':>3}  fitted quotes |MTM| max / RMS / sum "
        "(python | cpp); ditched"
    )
    for method in METHODS:
        built, errors, times = {}, {}, {}
        for builder in BUILDERS:
            backend(builder)
            built[builder] = build(quotes, new_market, method)
            errors[builder] = valuation_error(quotes, *built[builder])
            if not args.no_timing:
                times[builder] = timed(quotes, new_market, method)
        diff = forward_difference(built["python"][0], built["cpp"][0])
        ok = diff <= FORWARD_LIMIT and built["python"][1] == built["cpp"][1]
        failed |= not ok
        timing = (
            f"{times['python'][0]:8.2f} +- {times['python'][1]:4.2f} "
            f"{times['cpp'][0]:8.2f} +- {times['cpp'][1]:4.2f} "
            f"{times['python'][0] / times['cpp'][0]:5.2f}x"
            if times
            else f"{'':17} {'':17} {'':6}"
        )
        err = " | ".join(
            f"{e.max():.1e} / {np.sqrt(np.mean(e**2)):.1e} / {e.sum():.1e}"
            for e, _ in (errors["python"], errors["cpp"])
        )
        ditched = errors["cpp"][1]
        tail = f"{len(ditched)}, max {ditched.max():.1e}" if len(ditched) else "0"
        print(
            f"{method.name:18} {timing} {diff:10.1e} {'yes' if ok else 'NO':>3}  "
            f"{err}; {tail}"
        )
    settings.set_backend("python")
    return 1 if failed else 0


METHODS = [
    nb.M.PiecewiseConstant,
    nb.M.PiecewiseLinear,
    nb.M.CubicSpline,
    nb.M.HermiteCubicSpline,
    nb.M.ConvexMonotone,
]

if __name__ == "__main__":
    sys.exit(main())
