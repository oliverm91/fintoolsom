"""The C++ curves (fintoolsom._native) give the Python curves' results.

On Windows (MSVC, the scipy wheels) they agree to the bit. Other platforms may
differ by an ulp or two where numpy's or scipy's builds use another exp/log or fused
multiply-adds, so the comparisons allow a few ulps.
"""

import warnings
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.curve_builder import build_curves
from fintoolsom.market import Currency
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.rates import (
    DiscountCurve,
    IndexCurve,
    InterpolationMethod,
    LinearInterestConvention,
    Rate,
    RateConvention,
)
from tests.factories import quote_factory

pytest.importorskip("fintoolsom._native")

MAX_ULP = 4
T = date(2026, 6, 29)
SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)
ICP = OvernightRateIndex("ICP", currency=Currency.CLP)
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
_rng = np.random.default_rng(0)
FORWARDS = {
    "smooth": 0.04 + 0.01 * np.sin(np.arange(len(OFFSETS))),
    "noisy": 0.03 + 0.02 * _rng.standard_normal(len(OFFSETS)),
    "negative": -0.01 + 0.005 * _rng.standard_normal(len(OFFSETS)),
}
SPOT = Rate(
    RateConvention(
        interest_convention=LinearInterestConvention, time_fraction_base=360
    ),
    0.0525,
)
METHODS = [m for m in InterpolationMethod if m is not InterpolationMethod.LogLinear]
KINDS = ["discount", "index", "index-spot"]
# Before the curve date, every day of the curve, and past the last knot.
DAYS = np.arange(-40, OFFSETS[-1] + 400, dtype=np.int64)


def _build(kind, method, forwards, backend, precalculate=False):
    with settings.use_backend(backend), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PiecewiseLinear warns once
        if kind == "discount":
            knots = [T, *(T + timedelta(d) for d in OFFSETS)]
            return DiscountCurve(
                T, Currency.USD, ICP, knots, forwards, method, precalculate=precalculate
            )
        knots = [T + timedelta(1), *(T + timedelta(d + 1) for d in OFFSETS)]
        spot = SPOT if kind == "index-spot" else None
        return IndexCurve(
            T, SOFR, knots, forwards, spot, method, precalculate=precalculate
        )


def _pair(kind, method, forwards, precalculate=False):
    return tuple(
        _build(kind, method, FORWARDS[forwards], b, precalculate)
        for b in ("python", "cpp")
    )


def _same(a, b):
    np.testing.assert_array_max_ulp(np.asarray(a), np.asarray(b), maxulp=MAX_ULP)


params = pytest.mark.parametrize(
    "kind, method, forwards",
    [(k, m, f) for k in KINDS for m in METHODS for f in FORWARDS],
    ids=lambda v: getattr(v, "name", v),
)


@params
def test_backends_choose_their_implementation(kind, method, forwards):
    py, cpp = _pair(kind, method, forwards)
    assert py._core is None and cpp._core is not None


@params
@pytest.mark.parametrize("precalculate", [False, True])
def test_dfs_match_on_every_day(kind, method, forwards, precalculate):
    py, cpp = _pair(kind, method, forwards, precalculate)
    _same(py._dfs_at_days(DAYS), cpp._dfs_at_days(DAYS))
    _same(py._dfs_from_days(DAYS), cpp._dfs_from_days(DAYS))
    ordinals = DAYS + T.toordinal()
    _same(py._dfs_at_ordinals(ordinals), cpp._dfs_at_ordinals(ordinals))


@params
def test_grids_match(kind, method, forwards):
    py, cpp = _pair(kind, method, forwards, precalculate=True)
    assert len(py._grid) == len(cpp._grid)
    _same(py._grid, cpp._grid)
    assert not cpp._grid.flags.writeable


@params
def test_date_methods_match(kind, method, forwards):
    py, cpp = _pair(kind, method, forwards, precalculate=True)
    dates = [T + timedelta(int(d)) for d in (-3, 0, 1, 15, 400, 4000, OFFSETS[-1] + 30)]
    _same(py.get_dfs(dates), cpp.get_dfs(dates))
    for d in dates:
        _same(py.get_df(d), cpp.get_df(d))
        _same(py.get_df_fwd(T + timedelta(1), d), cpp.get_df_fwd(T + timedelta(1), d))


@params
def test_derived_curves_match(kind, method, forwards):
    py, cpp = _pair(kind, method, forwards)
    for derive in (
        lambda c: c.copy(),
        lambda c: c.parallel_bump_bps(10.0),
        lambda c: c.get_aged_curve(T + timedelta(200)),
    ):
        with settings.use_backend("python"):
            a = derive(py)
        with settings.use_backend("cpp"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = derive(cpp)
        assert b._core is not None
        _same(a._dfs_at_days(DAYS), b._dfs_at_days(DAYS))


def test_a_curve_keeps_its_backend_when_the_setting_changes():
    cpp = _build("discount", InterpolationMethod.CubicSpline, FORWARDS["smooth"], "cpp")
    with settings.use_backend("python"):
        assert cpp._core is not None
        assert cpp.copy()._core is None  # a new curve reads the setting


@pytest.mark.parametrize(
    "method", [InterpolationMethod.PiecewiseConstant, InterpolationMethod.CubicSpline]
)
def test_curve_building_matches(method):
    built = {}
    for backend in ("python", "cpp"):
        with settings.use_backend(backend):
            market = quote_factory.quote_market()
            build_curves(quote_factory.load_quotes(), market, interp_method=method)
        built[backend] = market.curves
    py, cpp = built["python"], built["cpp"]
    assert set(py) == set(cpp)
    for cid, curve in py.items():
        other = cpp[cid]
        assert other._core is not None
        _same(curve.forward_rates, other.forward_rates)
        _same(curve._grid, other._grid)
