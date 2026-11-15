"""Python vs C++ backend of the rates curves: atomized operations and curve building.

Run from the repository root, after ``uv sync --group dev``::

    uv run python cpp/benchmarks/bench_curves.py

Atomized: median time per call over many calls. Curve building: the quotes of
notebooks/quotes.json, 3 warm-up builds and 10 timed ones, dropping the 2 slowest, as
mean +- std. Both backends must build the same curves; the script checks it.
"""

import statistics
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
warnings.simplefilter("ignore")

from build_quotes import build_quotes  # noqa: E402

from fintoolsom import settings  # noqa: E402
from fintoolsom.curve_builder import build_curves  # noqa: E402
from fintoolsom.market import Currency, CurrencyName, CurrencyPair, Market  # noqa: E402
from fintoolsom.market.currencies import FX_Rate, FX_RateData  # noqa: E402
from fintoolsom.market.index import (  # noqa: E402
    OvernightInterestPriceIndex,
    OvernightRateIndex,
)
from fintoolsom.market.index_history import (  # noqa: E402
    OvernightInterestPriceHistory,
    OvernightRateHistory,
)
from fintoolsom.rates import (  # noqa: E402
    IndexCurve,
    InterpolationMethod,
    LinearInterestConvention,
    Rate,
    RateConvention,
)

T = date(2026, 6, 29)
M = InterpolationMethod
METHODS = [
    M.PiecewiseConstant,
    M.PiecewiseLinear,
    M.HermiteCubicSpline,
    M.CubicSpline,
    M.ConvexMonotone,
]
BACKENDS = ("python", "cpp")
SPOT = Rate(
    RateConvention(
        interest_convention=LinearInterestConvention, time_fraction_base=360
    ),
    0.0525,
)
SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)
OFFSETS = [
    1,
    7,
    30,
    61,
    92,
    183,
    274,
    365,
    548,
    730,
    1095,
    1461,
    1826,
    2557,
    3652,
    5479,
    7305,
    10957,
]
KNOTS = [T + timedelta(1), *(T + timedelta(d + 1) for d in OFFSETS)]
FORWARDS = 0.04 + 0.01 * np.sin(np.arange(len(OFFSETS)))


def curve(method, precalculate):
    return IndexCurve(T, SOFR, KNOTS, FORWARDS, SPOT, method, precalculate=precalculate)


def per_call(fn, min_time=0.2):
    """Median seconds per call of fn()."""
    fn()
    n, t0 = 1, time.perf_counter()
    while time.perf_counter() - t0 < 0.02:
        fn()
        n += 1
    loops = max(1, n)
    samples = []
    t_end = time.perf_counter() + min_time
    while time.perf_counter() < t_end or len(samples) < 5:
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter() - t0) / loops)
    return statistics.median(samples)


def fmt(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} us"
    return f"{seconds * 1e3:8.2f} ms"


def atomized():
    rng = np.random.default_rng(0)
    few = T.toordinal() + np.sort(rng.integers(0, 10950, 10))
    many = T.toordinal() + np.sort(rng.integers(0, 10950, 10_000))
    dates = [date.fromordinal(int(o)) for o in few]
    rows = []
    for method in METHODS:
        cases = {}
        for backend in BACKENDS:
            with settings.use_backend(backend):
                plain, grid = curve(method, False), curve(method, True)
                cases[backend] = {
                    "build": per_call(lambda: curve(method, False)),
                    "build + grid": per_call(lambda: curve(method, True)),
                    "grid only": per_call(plain.precalculate_dfs),
                    "10 dfs, formula": per_call(lambda: plain._dfs_at_ordinals(few)),
                    "10k dfs, formula": per_call(lambda: plain._dfs_at_ordinals(many)),
                    "10k dfs, grid": per_call(lambda: grid._dfs_at_ordinals(many)),
                    "get_dfs 10 dates": per_call(lambda: grid.get_dfs(dates)),
                }
        for op in cases["python"]:
            py, cpp = cases["python"][op], cases["cpp"][op]
            rows.append((method.name, op, py, cpp))
    print(f"{'method':20}{'operation':20}{'python':>12}{'c++':>12}{'speed-up':>10}")
    for method, op, py, cpp in rows:
        print(f"{method:20}{op:20}{fmt(py):>12}{fmt(cpp):>12}{py / cpp:9.1f}x")
    return rows


def quote_market():  # as notebooks/main.py
    usd, clp = Currency(CurrencyName.USD), Currency(CurrencyName.CLP)
    sofr = OvernightRateIndex("SOFR", currency=usd)
    sofr_data = OvernightRateHistory(sofr, {T: SPOT})
    icp = OvernightInterestPriceIndex("ICP", currency=clp)
    icp_data = OvernightInterestPriceHistory(icp, {T: 100.0})
    pair = CurrencyPair(usd, clp)
    return Market(
        T,
        fx_history={pair: FX_RateData(pair, {T: FX_Rate(pair, 900)})},
        indexes_history={"SOFR": sofr_data, "ICP": icp_data},
        risk_free_index=icp,
    )


def curve_building(warmups=3, runs=10, drop=2):
    quotes = build_quotes(T, str(ROOT / "notebooks" / "quotes.json"))
    probe = [T + timedelta(days=d) for d in range(0, 365 * 31, 7)]
    print(f"\n{'method':20}{'backend':10}{'mean':>12}{'std':>12}{'speed-up':>10}")
    rows = []
    for method in (M.PiecewiseConstant, M.CubicSpline, M.ConvexMonotone):
        stats, dfs = {}, {}
        for backend in BACKENDS:
            with settings.use_backend(backend):
                times = []
                for run in range(warmups + runs):
                    market = quote_market()
                    t0 = time.perf_counter()
                    build_curves(quotes, market, interp_method=method)
                    if run >= warmups:
                        times.append(time.perf_counter() - t0)
            kept = sorted(times)[: runs - drop]
            stats[backend] = (statistics.mean(kept), statistics.stdev(kept))
            dfs[backend] = {cid: c.get_dfs(probe) for cid, c in market.curves.items()}
        worst = max(
            float(np.max(np.abs(dfs["python"][cid] / dfs["cpp"][cid] - 1)))
            for cid in dfs["python"]
        )
        for backend in BACKENDS:
            mean, std = stats[backend]
            up = stats["python"][0] / mean
            print(
                f"{method.name:20}{backend:10}{fmt(mean):>12}{fmt(std):>12}{up:9.2f}x"
            )
        print(f"{'':20}max relative df difference between backends: {worst:.1e}")
        rows.append((method.name, stats, worst))
    return rows


if __name__ == "__main__":
    atomized()
    curve_building()
