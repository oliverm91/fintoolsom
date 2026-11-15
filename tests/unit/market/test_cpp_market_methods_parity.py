"""The C++ Market methods, index histories and UF loops match the Python ones exactly.

Covers what runs in C++ beyond the one-call valuations of test_cpp_market_parity:
the realised fixings of started coupons (no Python callback), leg duration and DV01,
the par check of floating legs, Market._dfs_at_ordinals,
_future_fx_rates_at_ordinals, get_fx_known_until and the rate-accrual loops, the gap
filling and level compounding of the index histories, and the UF extension. Each
case runs all in Python and all in C++ and must agree to the bit, in value, type,
dict order and error; off Windows, values that went through exp may differ by a few
ulps (numpy's exp is not the C library's there), as in tests/unit/rates.
"""

import math
import random
import sys
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.dates import Calendar, get_calendar
from fintoolsom.dates.date_counts import ActualDayCountConvention, date_ordinals
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.calculator import _swap_pv
from fintoolsom.derivatives.swaps import LegSide
from fintoolsom.market import (
    OvernightInterestPriceHistory,
    OvernightInterestPriceIndex,
    OvernightRateHistory,
    OvernightRateIndex,
)
from fintoolsom.market.currencies import Currency, CurrencyPair
from fintoolsom.market.uf import UFConvention, extend_with_cpi
from fintoolsom.rates import (
    CompoundedInterestConvention,
    ExponentialInterestConvention,
    LinearInterestConvention,
    Rate,
    RateConvention,
)
from tests.factories.swap_factory import (
    COLLATERAL_INDEXES,
    ICP,
    REPORTING_CURRENCY,
    SOFR,
    UF_CURRENCY,
    VAL_DATE,
    SwapStatus,
    build_market,
    build_swap,
    swap_cases,
)

native = pytest.importorskip("fintoolsom._native")

USD_CLP = CurrencyPair(Currency.USD, Currency.CLP)
UF_CLP = CurrencyPair(UF_CURRENCY, Currency.CLP)
CASES = [pytest.param(c, id=c.id) for c in swap_cases()]
STARTED = [
    pytest.param(c, id=c.id) for c in swap_cases() if c.status is SwapStatus.STARTED
]


MAX_ULP = 0 if sys.platform == "win32" else 4


def _outcome(f):
    try:
        out = f()
    except Exception as e:  # noqa: BLE001 - both backends must fail alike
        return type(e), str(e)
    return type(out), out


def _close(a: float, b: float) -> bool:
    if a == b or (a != a and b != b):
        return True
    return MAX_ULP > 0 and math.isfinite(a) and abs(a - b) <= MAX_ULP * math.ulp(a)


def _same(a, b) -> bool:
    """Equal (to the bit on Windows), through tuples, lists and arrays, types included."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and all(map(_close, a.tolist(), b.tolist()))
        )
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, (float, np.floating)):
        return _close(float(a), float(b))
    return a == b


def _build(backend, rates="python"):
    """A market under `backend`, its curves under `rates`."""
    with settings.use_backend(backend), settings.use_backend(rates, "rates"):
        return build_market()


@pytest.fixture(scope="module", params=["python", "cpp"], ids=lambda b: f"{b}-curves")
def markets(request):
    """(Python market, C++ market) on the same curves, with the same histories."""
    py, cpp = _build("python", request.param), _build("cpp", request.param)
    assert not py._native and cpp._native
    return py, cpp


def _both(build):
    out = []
    for backend in ("python", "cpp"):
        with settings.use_backend(backend):
            out.append(build())
    return out


# ── Swaps ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", STARTED)
def test_realised_fixings_need_no_python_callback(case, markets, monkeypatch):
    py_swap, cpp_swap = _both(lambda: build_swap(case))
    py_market, cpp_market = markets
    expected = [
        Calculator.valuate(py_swap, py_market, REPORTING_CURRENCY),
        *(Calculator.get_leg_duration(py_swap, s, py_market) for s in LegSide),
    ]

    def python_callback(*args):
        raise AssertionError("realised fixings read in Python")

    monkeypatch.setattr(_swap_pv, "_running_flows", python_callback)
    monkeypatch.setattr(_swap_pv, "_index_history", python_callback)
    assert [
        Calculator.valuate(cpp_swap, cpp_market, REPORTING_CURRENCY),
        *(Calculator.get_leg_duration(cpp_swap, s, cpp_market) for s in LegSide),
    ] == expected


@pytest.mark.parametrize("case", CASES)
def test_duration_and_dv01_match(case, markets):
    py_swap, cpp_swap = _both(lambda: build_swap(case))
    py_market, cpp_market = markets
    for side in LegSide:
        assert _same(
            Calculator.get_leg_duration(cpp_swap, side, cpp_market),
            Calculator.get_leg_duration(py_swap, side, py_market),
        )
        assert _same(
            _outcome(
                lambda: Calculator.get_leg_dv01(
                    cpp_swap, side, cpp_market, Currency.CLP
                )
            ),
            _outcome(
                lambda: Calculator.get_leg_dv01(py_swap, side, py_market, Currency.CLP)
            ),
        )


@pytest.mark.parametrize("case", CASES)
def test_par_check_matches(case):
    py_swap, cpp_swap = _both(lambda: build_swap(case))
    for py_leg, cpp_leg in zip(
        (py_swap.receive_leg, py_swap.pay_leg), (cpp_swap.receive_leg, cpp_swap.pay_leg)
    ):
        assert getattr(cpp_leg, "_par_eligible", None) == getattr(
            py_leg, "_par_eligible", None
        )


def _python_repays_at_par(pay, start, end, r, a):
    return bool(
        np.array_equal(pay, end)
        and np.array_equal(start[1:], end[:-1])
        and np.allclose(r[1:], r[:-1] - a[:-1], rtol=1e-12, atol=0.0)
        and np.isclose(r[-1], a[-1], rtol=1e-12, atol=0.0)
    )


def test_repays_at_par_matches_numpy():
    rng = random.Random(7)
    specials = [0.0, -0.0, 1e-300, math.inf, -math.inf, math.nan]
    for _ in range(3000):
        n = rng.randint(1, 5)
        end = np.cumsum(np.array([rng.randint(1, 3) for _ in range(n)], dtype=np.int64))
        start = np.concatenate(([0], end[:-1])).astype(np.int64)
        if rng.random() < 0.2:
            start[rng.randrange(n)] += 1
        pay = end + (rng.random() < 0.2)
        a = np.array([rng.choice([1.0, 0.5, 2.0 / 3.0]) for _ in range(n)])
        r = np.cumsum(a[::-1])[::-1].copy()
        for x in (r, a):
            if rng.random() < 0.3:
                x[rng.randrange(n)] *= 1 + rng.choice([1e-13, 1e-11, 0.0])
            if rng.random() < 0.1:
                x[rng.randrange(n)] = rng.choice(specials)
        assert native.repays_at_par(pay, start, end, r, a) == _python_repays_at_par(
            pay, start, end, r, a
        ), (pay, start, end, r, a)


# ── Market methods ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("collateral", list(COLLATERAL_INDEXES.values()))
@pytest.mark.parametrize("currency", [Currency.CLP, Currency.USD, UF_CURRENCY])
@pytest.mark.parametrize("risk_free_index", [None, SOFR, ICP])
def test_dfs_match(markets, collateral, currency, risk_free_index):
    py, cpp = markets
    days = [VAL_DATE + timedelta(days=d) for d in (0, 1, 30, 365, 4000)]
    for f in (
        lambda m: m.get_dfs(
            currency, days, collateral=collateral, risk_free_index=risk_free_index
        ),
        lambda m: m._dfs_at_ordinals(
            currency, date_ordinals(days)[::-1].copy(), collateral=collateral
        ),
    ):
        assert _same(_outcome(lambda: f(cpp)), _outcome(lambda: f(py)))


@pytest.mark.parametrize(
    "pair",
    [
        USD_CLP,
        UF_CLP,
        UF_CLP.invert(),
        CurrencyPair(Currency.CLP, Currency.CLP),
        CurrencyPair(Currency.EUR, Currency.USD),
    ],
)
def test_future_fx_and_known_until_match(markets, pair):
    py, cpp = markets
    days = [VAL_DATE + timedelta(days=d) for d in (-3, 0, 1, 20, 32, 33, 400)]
    ordinals = date_ordinals(days)
    assert _same(
        _outcome(lambda: cpp._future_fx_rates_at_ordinals(pair, ordinals)),
        _outcome(lambda: py._future_fx_rates_at_ordinals(pair, ordinals)),
    )
    assert _outcome(lambda: cpp.get_fx_known_until(pair)) == _outcome(
        lambda: py.get_fx_known_until(pair)
    )
    for d in days[2:]:
        assert _same(
            _outcome(lambda: cpp.get_future_fx_rate(d, pair).value),
            _outcome(lambda: py.get_future_fx_rate(d, pair).value),
        )


def _accrual_markets(convention):
    """(Python, C++) markets with a daily rate series in `convention`."""
    out = []
    start = date(2024, 1, 1)
    for backend in ("python", "cpp"):
        with settings.use_backend(backend):
            series = {
                start + timedelta(days=i): Rate(convention, 0.05 + 1e-4 * (i % 7))
                for i in range(0, 120)
                if i % 11 != 3  # a few missing fixings
            }
            m = _build(backend)
            m.interest_rates["R"] = series
            out.append(m)
    return out


CONVENTIONS = [
    RateConvention(LinearInterestConvention, ActualDayCountConvention, 360),
    RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365),
    RateConvention(ExponentialInterestConvention, ActualDayCountConvention, 365),
]


@pytest.mark.parametrize(
    "convention", CONVENTIONS, ids=lambda c: c.interest_convention.__name__
)
@pytest.mark.parametrize("notional", [1_000_000.0, 7, np.float64(3.5)])
@pytest.mark.parametrize("closest", [False, True])
def test_accrual_loops_match(convention, notional, closest):
    py, cpp = _accrual_markets(convention)
    cal = get_calendar("CL")
    cases = [
        (date(2024, 1, 2), date(2024, 3, 1), 0),
        (date(2024, 1, 3), date(2024, 2, 20), 2),
        (date(2024, 2, 1), date(2024, 2, 1), 1),
        (date(2024, 1, 1), date(2024, 1, 5), 0),
        (date(2024, 3, 20), date(2024, 6, 1), 0),  # past the series
    ]
    for start, end, lag in cases:
        assert _same(
            _outcome(
                lambda: cpp.accrue_rates_reset_business_days(
                    notional, "R", start, end, lag, cal, closest
                )
            ),
            _outcome(
                lambda: py.accrue_rates_reset_business_days(
                    notional, "R", start, end, lag, cal, closest
                )
            ),
        ), (start, end, lag)
        for resets in (
            [start, start + timedelta(days=9), start + timedelta(days=30)],
            [start - timedelta(days=5), start - timedelta(days=1)],
            [start, end + timedelta(days=5)],
            [start + timedelta(days=4), start],
        ):
            assert _same(
                _outcome(
                    lambda: cpp.accrue_rates_custom_reset_days(
                        notional, "R", start, end, list(resets), closest
                    )
                ),
                _outcome(
                    lambda: py.accrue_rates_custom_reset_days(
                        notional, "R", start, end, list(resets), closest
                    )
                ),
            ), (start, end, resets)


def test_accrual_of_an_unknown_rate_raises_the_same_error():
    py, cpp = _accrual_markets(CONVENTIONS[0])
    for m in (py, cpp):
        m.interest_rates.pop("R")
    for f in (
        lambda m: m.accrue_rates_reset_business_days(
            1.0, "R", date(2024, 1, 2), date(2024, 1, 9)
        ),
        lambda m: m.accrue_rates_custom_reset_days(
            1.0, "R", date(2024, 1, 2), date(2024, 1, 9), [date(2024, 1, 2)]
        ),
    ):
        assert _outcome(lambda: f(cpp)) == _outcome(lambda: f(py))


# ── Index histories ─────────────────────────────────────────────────────────


def _rate_history(convention, gaps=True):
    rates = {}
    for i in range(400):
        d = date(2023, 1, 1) + timedelta(days=i)
        if gaps and (i % 13 == 5 or 40 < i < 55):
            continue
        rates[d] = Rate(convention, 0.04 + 1e-4 * (i % 9))
    return OvernightRateHistory(OvernightRateIndex("X", currency=Currency.CLP), rates)


def _state(h):
    if isinstance(h, OvernightRateHistory):
        return (
            list(h.rates.items()),
            sorted(h._filled_dates),
            [(d, type(v), v) for d, v in h._index_values.items()],
        )
    return [(d, type(v), v) for d, v in h.values.items()]


def _histories(build):
    py, cpp = _both(build)
    assert _state(cpp) == _state(py)
    return py, cpp


@pytest.mark.parametrize(
    "convention", CONVENTIONS, ids=lambda c: c.interest_convention.__name__
)
def test_overnight_rate_histories_match(convention):
    py, cpp = _histories(lambda: _rate_history(convention))
    other = Rate(convention, 0.07)
    updates = [
        date(2024, 2, 6),  # after the end
        date(2024, 2, 20),  # past a gap after the end
        date(2023, 2, 14),  # inside: replaces a fixing
        date(2023, 2, 15),  # a filled date
        date(2022, 12, 1),  # before the first fixing
        date(2024, 3, 2),  # a Saturday: rebuild
    ]
    for d in updates:
        for backend, h in (("python", py), ("cpp", cpp)):
            with settings.use_backend(backend):
                h.add_rate(d, other)
        assert _state(cpp) == _state(py), d
    for d0, d1 in (
        (date(2023, 1, 3), date(2023, 12, 1)),
        (date(2023, 3, 1), date(2024, 1, 9)),
    ):
        assert _same(
            cpp.get_accrued_interest(1e6, d0, d1), py.get_accrued_interest(1e6, d0, d1)
        )


def test_a_mixed_rate_history_carries_on_in_python():
    """A fixing the C++ walk does not take (a numpy rate) hands over to Python mid-walk."""

    def build():
        h = _rate_history(CONVENTIONS[0], gaps=False)
        rates = dict(h.rates)
        rates[date(2023, 6, 1)] = Rate(CONVENTIONS[0], np.float64(0.05))
        return OvernightRateHistory(h.index, rates)

    _histories(build)


def test_verbose_gap_filling_warns_alike():
    py, cpp = _both(lambda: _rate_history(CONVENTIONS[0]))
    out = []
    for backend, h in (("python", py), ("cpp", cpp)):
        h.rates[date(2025, 1, 6)] = h.rates[max(h.rates)]
        with settings.use_backend(backend), pytest.warns(UserWarning) as record:
            h._check_for_gaps(verbose=True)
        out.append([str(w.message) for w in record])
    assert out[0] == out[1] and _state(cpp) == _state(py)


def _price_history(values):
    return OvernightInterestPriceHistory(
        OvernightInterestPriceIndex("P", currency=Currency.CLP), dict(values)
    )


@pytest.mark.parametrize(
    "values",
    [
        {
            date(2024, 1, 2): 10000.0,
            date(2024, 1, 19): 10031.7,
            date(2024, 3, 1): 10100.25,
        },
        {date(2024, 1, 2): 100, date(2024, 2, 1): 101},
        {date(2024, 1, 2): 100.0, date(2024, 2, 1): -1.0},  # complex in Python
        {date(2024, 1, 2): 0.0, date(2024, 1, 3): 1.0},  # ZeroDivisionError
        {date(2024, 1, 2): np.float64(100.0), date(2024, 2, 1): np.float64(101.0)},
    ],
)
def test_price_histories_match(values):
    py = _outcome(lambda: _state(_both_one("python", values)))
    cpp = _outcome(lambda: _state(_both_one("cpp", values)))
    assert py == cpp


def _both_one(backend, values):
    with settings.use_backend(backend):
        h = _price_history(values)
        h.add_value(date(2024, 4, 2), 10150.0)
        return h


def test_the_history_walks_run_in_cpp(monkeypatch):
    calls = []
    for name in ("fill_rate_gaps", "compound_levels", "fill_price_gaps"):
        original = getattr(native, name)
        monkeypatch.setattr(
            native, name, lambda *a, f=original, n=name: calls.append(n) or f(*a)
        )
    with settings.use_backend("cpp"):
        _rate_history(CONVENTIONS[0])
        _price_history({date(2024, 1, 2): 1.0, date(2024, 2, 1): 1.01})
    assert set(calls) == {"fill_rate_gaps", "compound_levels", "fill_price_gaps"}
    calls.clear()
    with settings.use_backend("python"):
        _rate_history(CONVENTIONS[0])
    assert not calls


def test_a_calendar_subclass_stays_in_python(monkeypatch):
    class MyCalendar(Calendar):
        pass

    index = OvernightRateIndex("X", currency=Currency.CLP)
    index.calendar = MyCalendar.__new__(MyCalendar)
    index.calendar.__setstate__(get_calendar("CL").__getstate__())
    with settings.use_backend("cpp"):
        h = OvernightRateHistory(index, {})
        assert h._native_calendar() is None


# ── UF ──────────────────────────────────────────────────────────────────────


def test_uf_reajuste_rounds_as_python():
    rng = random.Random(3)
    for _ in range(20000):
        base = rng.uniform(20000.0, 40000.0)
        cpi = rng.uniform(-0.01, 0.02)
        period = rng.choice([28, 29, 30, 31])
        elapsed = rng.randint(1, period)
        assert native.uf_reajuste(base, cpi, elapsed, period) == UFConvention.reajuste(
            base, cpi, elapsed, period
        )
    # Exact halves in binary: round half to even, as Python.
    for x in (0.125, 0.375, 2.675, 1.005, 37000.125):
        assert native.uf_reajuste(x, 0.0, 1, 30) == round(x, 2)


@pytest.mark.parametrize("base_date", [None, date(2024, 7, 9), date(2024, 2, 9)])
@pytest.mark.parametrize("cpi", [0.004, -0.002, 0, -2.0])
def test_extend_with_cpi_matches(base_date, cpi):
    levels = {
        date(2024, 7, 9): 37571.86,
        date(2024, 2, 9): 36789.01,
        date(2024, 7, 12): 1.0,
    }

    def run():
        out = extend_with_cpi(levels, cpi, base_date)
        return [(d, type(v), v) for d, v in out.items()]

    py, cpp = _both(lambda: _outcome(run))
    assert py == cpp
