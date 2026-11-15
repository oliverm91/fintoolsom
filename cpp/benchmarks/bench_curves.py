"""Python vs C++ backends: curves, swap valuation and curve building.

Run from the repository root, after ``uv sync --group dev``::

    uv run python cpp/benchmarks/bench_curves.py

Atomized: median time per call over many calls (curves; swaps of the test factory,
valued on its market). Curve building: the quotes of notebooks/quotes.json, 3 warm-up
builds and 10 timed ones, dropping the 2 slowest, as mean +- std, for Python, C++
curves only, and C++ curves and swaps. Every configuration must build the same curves;
the script checks it.
"""

import contextlib
import statistics
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from build_quotes import build_quotes  # noqa: E402

from fintoolsom import settings  # noqa: E402
from fintoolsom.curve_builder import build_curves  # noqa: E402
from fintoolsom.derivatives.calculator import Calculator  # noqa: E402
from fintoolsom.derivatives.swaps import NonStartedSimplifiedSwap  # noqa: E402
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


SWAP_CASES = {
    "IRS USD OIS, SOFR CSA (par)": "irs-USD-fixed-vs-USD-overnight-not_started-nospread-coll_sofr",
    "IRS CLP ICP, uncoll (par)": "irs-CLP-fixed-vs-CLP-price-not_started-nospread-uncoll",
    "IRS CLP OIS, SOFR CSA": "irs-CLP-fixed-vs-CLP-overnight-not_started-spread-coll_sofr",
    "IRS USD OIS started": "irs-USD-fixed-vs-USD-overnight-started-spread-coll_sofr",
    "CCS deliverable": "ccs-CLP-fixed-vs-USD-overnight-not_started-nospread-coll_sofr-deliv",
    "CCS non-deliverable": "ccs-CLP-overnight-vs-USD-overnight-not_started-spread-coll_sofr-nondeliv",
}


def swaps():
    from tests.factories.swap_factory import (
        REPORTING_CURRENCY,
        build_market,
        build_swap,
        swap_cases,
    )

    cases = {c.id: c for c in swap_cases()}
    print()
    print(f"{'swap valuation':34}{'python':>12}{'c++':>12}{'speed-up':>10}")
    rows = []
    for rates in BACKENDS:
        with settings.use_backend(rates, "rates"):
            market = build_market()
        print(f"  on {rates} curves")
        for label, case_id in SWAP_CASES.items():
            variants = [("", lambda s: s)]
            if "not_started" in case_id and "nondeliv" not in case_id:
                variants.append((" [NonStarted]", _non_started))
            for suffix, wrap in variants:
                timing = {}
                for backend in BACKENDS:
                    with settings.use_backend(backend, "derivatives"):
                        swap = wrap(build_swap(cases[case_id]))
                    timing[backend] = per_call(
                        lambda: Calculator.valuate(swap, market, REPORTING_CURRENCY)
                    )
                py, cpp = timing["python"], timing["cpp"]
                print(
                    f"  {label + suffix:32}{fmt(py):>12}{fmt(cpp):>12}{py / cpp:9.1f}x"
                )
                rows.append((rates, label + suffix, py, cpp))
    return rows


def _non_started(swap):
    return NonStartedSimplifiedSwap(
        receive_leg=swap.receive_leg,
        pay_leg=swap.pay_leg,
        collateral_index=swap.collateral_index,
    )


def fx():
    """FX forward, NDF and option valuation (spot and greeks), per call."""
    from fintoolsom.derivatives.calculator import _options
    from fintoolsom.derivatives.forwards.forwards import NDF, Forward
    from fintoolsom.derivatives.options.options import Call
    from fintoolsom.market.volatility_surface import VolatilitySurface
    from tests.factories.swap_factory import SOFR as F_SOFR
    from tests.factories.swap_factory import VAL_DATE, build_market

    usd_clp = CurrencyPair(Currency.USD, Currency.CLP)
    pay, fix = VAL_DATE + timedelta(days=200), VAL_DATE + timedelta(days=198)
    cases = {
        "Forward USD/CLP, uncoll": lambda: Forward(1e6, 951.0, pay, True, usd_clp),
        "Forward USD/CLP, SOFR CSA": lambda: Forward(
            1e6, 951.0, pay, True, usd_clp, collateral_index=F_SOFR
        ),
        "NDF USD/CLP, projected": lambda: NDF(
            1e6, 951.0, pay, True, usd_clp, fixing_date=fix
        ),
        "Call USD/CLP, from surface": lambda: Call(1e6, 960.0, pay, usd_clp),
    }
    print()
    print(f"{'fx valuation':34}{'python':>12}{'c++':>12}{'speed-up':>10}")
    rows = []
    for rates in BACKENDS:
        with settings.use_backend(rates, "rates"):
            market = build_market()
        domestic = market.get_discount_curve(Currency.CLP)
        foreign = market.get_discount_curve(Currency.USD)
        market.volatility_surfaces[usd_clp] = VolatilitySurface(
            [30, 91, 182, 365, 730],
            [10, 25, "atm", 75, 85],
            [[0.14, 0.12, 0.11, 0.115, 0.13]] * 5,
            foreign,
        )
        print(f"  on {rates} curves")
        for label, build in cases.items():
            timing = {}
            for backend in BACKENDS:
                with settings.use_backend(backend, "derivatives"):
                    instrument = build()
                timing[backend] = per_call(
                    lambda: Calculator.valuate(instrument, market, Currency.CLP)
                )
            rows.append(_fx_row(rates, label, timing))
        timing = {}
        for backend in BACKENDS:
            with settings.use_backend(backend, "derivatives"):
                call = cases["Call USD/CLP, from surface"]()
            timing[backend] = per_call(
                lambda: _options.get_option_delta(
                    call, VAL_DATE, 951.0, 0.12, domestic, foreign
                )
            )
        rows.append(_fx_row(rates, "Call delta", timing))
        timing = {}
        for backend in BACKENDS:
            with settings.use_backend(backend, "derivatives"):
                timing[backend] = per_call(
                    lambda: _options.get_strike_from_delta(
                        0.25, 951.0, 0.12, domestic, foreign, pay, 1
                    )
                )
        rows.append(_fx_row(rates, "strike from 25 delta", timing))
    return rows


def _fx_row(rates, label, timing):
    py, cpp = timing["python"], timing["cpp"]
    print(f"  {label:32}{fmt(py):>12}{fmt(cpp):>12}{py / cpp:9.1f}x")
    return rates, label, py, cpp


# (label, subpackages on the C++ backend (the rest Python), simplified swaps)
CONFIGS = [
    ("python", (), False),
    ("python simplified", (), True),
    ("c++ rates+derivatives", ("rates", "derivatives"), False),
    ("  + market", ("rates", "derivatives", "market"), False),
    ("  + dates (all c++)", ("rates", "derivatives", "market", "dates"), False),
    ("all c++ simplified", ("rates", "derivatives", "market", "dates"), True),
]


@contextlib.contextmanager
def simplified_swaps(enabled):
    """Have the curve builder value its swaps as NonStartedSimplifiedSwap."""
    if not enabled:
        yield
        return
    from fintoolsom.curve_builder import _plan, builder
    from fintoolsom.derivatives.swaps import Swap

    original = _plan._instrument_of

    def wrapped(quote, market):
        instrument = original(quote, market)
        return _non_started(instrument) if isinstance(instrument, Swap) else instrument

    _plan._instrument_of = builder._instrument_of = wrapped
    try:
        yield
    finally:
        _plan._instrument_of = builder._instrument_of = original


def curve_building(
    warmups=3,
    runs=10,
    drop=2,
    methods=(M.PiecewiseConstant, M.CubicSpline, M.ConvexMonotone),
):
    quotes = build_quotes(T, str(ROOT / "notebooks" / "quotes.json"))
    probe = [T + timedelta(days=d) for d in range(0, 365 * 31, 7)]
    print()
    print(f"{'method':20}{'backend':24}{'mean':>12}{'std':>12}{'speed-up':>10}")
    rows = []
    for method in methods:
        stats, dfs = {}, {}
        for label, ported, simplified in CONFIGS:
            with (
                settings.use_backend("python"),
                settings.use_backend("cpp", *ported)
                if ported
                else contextlib.nullcontext(),
                simplified_swaps(simplified),
            ):
                times = []
                for run in range(warmups + runs):
                    market = quote_market()
                    t0 = time.perf_counter()
                    build_curves(quotes, market, interp_method=method)
                    if run >= warmups:
                        times.append(time.perf_counter() - t0)
            kept = sorted(times)[: runs - drop]
            stats[label] = (statistics.mean(kept), statistics.stdev(kept))
            dfs[label] = {cid: c.get_dfs(probe) for cid, c in market.curves.items()}
        reference = dfs["python"]
        worst = max(
            float(np.max(np.abs(reference[cid] / other[cid] - 1)))
            for other in dfs.values()
            for cid in reference
        )
        for label, *_ in CONFIGS:
            mean, std = stats[label]
            up = stats["python"][0] / mean
            print(f"{method.name:20}{label:24}{fmt(mean):>12}{fmt(std):>12}{up:9.2f}x")
        print(f"{'':20}max relative df difference between backends: {worst:.1e}")
        rows.append((method.name, stats, worst))
    return rows


if __name__ == "__main__":
    atomized()
    swaps()
    fx()
    curve_building()
