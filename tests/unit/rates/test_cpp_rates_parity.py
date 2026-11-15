"""The C++ Rate factors, year fractions and knot checks match the Python ones exactly.

Rate.get_wealth_factor, get_discount_factor and get_accrued_interest run in C++
under the "cpp" backend of "rates" (fintoolsom._native.rate_factors), with Python's
result types (np.float64 for exponential rates and numpy scalar inputs); so do
Curve.year_fractions and the knot checks of PillarCurve. Off-path inputs (datetimes,
empty or unequal lists, conventions of other classes) stay in Python, errors
included. Off Windows, exponential factors may differ by a few ulps (numpy's exp is
not the C library's there).
"""

import math
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.dates import date_counts
from fintoolsom.market import Currency, OvernightRateIndex
from fintoolsom.rates import (
    CompoundedInterestConvention,
    ExponentialInterestConvention,
    IndexCurve,
    LinearInterestConvention,
    Rate,
    RateConvention,
)
from fintoolsom.rates.Curve import year_fractions

pytest.importorskip("fintoolsom._native")
# numpy's overflow and zero-division warnings of the Python results (base 0, huge rates).
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

MAX_ULP = 0 if sys.platform == "win32" else 4
INTERESTS = [
    LinearInterestConvention,
    CompoundedInterestConvention,
    ExponentialInterestConvention,
]
DAY_COUNTS = list(date_counts._KINDS)
D0 = date(2024, 1, 31)
DATES = [D0 + timedelta(days=d) for d in (1, 29, 59, 366, 3653)]


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
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and all(map(_close, a.ravel().tolist(), b.ravel().tolist()))
        )
    if isinstance(a, tuple):
        return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, (float, np.floating)):
        return _close(float(a), float(b))
    return a == b


def _both(f):
    out = []
    for backend in ("python", "cpp"):
        with settings.use_backend(backend, "rates"):
            out.append(_outcome(f))
    return out


@pytest.mark.parametrize("interest", INTERESTS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("day_count", DAY_COUNTS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("base", [360, 365, 365.25, 0])
@pytest.mark.parametrize("value", [0.05, 3, np.float64(0.031), -0.99, 250.0])
def test_rate_factors_match(interest, day_count, base, value):
    rate = Rate(RateConvention(interest, day_count, base), value)
    inputs = [
        (D0, DATES[1]),
        (D0, D0),
        (DATES[3], D0),
        ([D0] * 5, DATES),
        (D0, DATES),
        (DATES, DATES[-1]),
        (DATES[:2], DATES),  # unequal
        ([], []),
        (datetime(2024, 1, 31), DATES[1]),
        (tuple(DATES), tuple(DATES)),
    ]
    for start, end in inputs:
        for f in (
            lambda: rate.get_wealth_factor(start, end),
            lambda: rate.get_discount_factor(start, end),
            lambda: rate.get_accrued_interest(1e6, start, end),
            lambda: rate.get_accrued_interest(7, start, end),
            lambda: rate.get_accrued_interest(np.float64(2.5), start, end),
        ):
            py, cpp = _both(f)
            assert _same(cpp, py), (start, end, py, cpp)


def test_rate_factors_run_in_cpp(monkeypatch):
    native = settings.native()
    calls = []
    original = native.rate_factors
    monkeypatch.setattr(
        native, "rate_factors", lambda *a: calls.append(1) or original(*a)
    )
    rate = Rate(RateConvention(), 0.05)
    with settings.use_backend("cpp", "rates"):
        rate.get_wealth_factor(D0, DATES[1])
    with settings.use_backend("python", "rates"):
        rate.get_wealth_factor(D0, DATES[1])
    assert calls == [1]


@pytest.mark.parametrize(
    "dates",
    [
        DATES,
        tuple(DATES),
        [],
        [D0 - timedelta(days=3)],
        [datetime(2024, 3, 1)],
        np.array(DATES),
    ],
    ids=["list", "tuple", "empty", "past", "datetime", "array"],
)
def test_year_fractions_match(dates):
    py, cpp = _both(lambda: year_fractions(D0, dates))
    assert _same(cpp, py)


SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)


@pytest.mark.parametrize(
    "knots",
    [
        [D0, DATES[0], DATES[2], DATES[3]],
        [DATES[0], DATES[2], DATES[3]],  # first knot after the curve date
        [D0, DATES[0]],  # too few
        [D0, DATES[2], DATES[0]],  # not increasing
        [D0, DATES[0], DATES[0]],  # repeated
        [D0 - timedelta(days=1), DATES[0], DATES[2]],  # before the curve date
        (D0, DATES[0], DATES[2]),
    ],
)
def test_knot_checks_match(knots):
    def build():
        curve = IndexCurve(D0, SOFR, knots, np.full(len(knots) - 1, 0.04))
        return curve._knot_tau, curve.get_dfs(DATES)

    py, cpp = _both(build)
    assert _same(cpp, py)
