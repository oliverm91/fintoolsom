"""The C++ FX forward, NDF and option valuation gives the Python one's results, to the bit.

Each instrument is built twice, under the Python and the C++ ``"derivatives"``
backend, and valued on the same market (of Python or of C++ curves). Besides IEEE
``+ * /`` both call the C library's exp, log, sqrt and erfc, CPython through
:mod:`math` and C++ through ``<cmath>``; ``ndtr`` / ``ndtri`` are computed the same way
on both sides (:mod:`fintoolsom.derivatives.calculator._normal`).
"""

import math
from datetime import timedelta

import numpy as np
import pytest
from scipy.special import ndtr as scipy_ndtr
from scipy.special import ndtri as scipy_ndtri

from fintoolsom import settings
from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.derivatives.calculator import _normal, _options
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.forwards.forwards import NDF, Forward
from fintoolsom.derivatives.options.options import Call, Put
from fintoolsom.market.currencies import Currency, CurrencyPair
from fintoolsom.market.volatility_surface import VolatilitySurface
from fintoolsom.rates import LinearInterestConvention, RateConvention
from tests.factories.swap_factory import (
    COLLATERAL_INDEXES,
    UF_CURRENCY,
    UF_KNOWN_UNTIL,
    VAL_DATE,
    build_market,
)

native = pytest.importorskip("fintoolsom._native")

USD_CLP = CurrencyPair(Currency.USD, Currency.CLP)
UF_CLP = CurrencyPair(UF_CURRENCY, Currency.CLP)
OFFSETS = [-1, 0, 1, 7, 33, 180, 400, 1100]


@pytest.fixture(scope="module", params=["python", "cpp"], ids=lambda b: f"{b}-curves")
def market(request):
    with settings.use_backend(request.param, "rates"):
        return build_market()


def _both(build):
    """`build()` under the Python and the C++ derivatives backend."""
    out = []
    for backend in ("python", "cpp"):
        with settings.use_backend(backend, "derivatives"):
            out.append(build())
    assert not out[0]._native and out[1]._native
    return out


def _outcome(f):
    """``f()``, or its exception's type and message."""
    try:
        return f()
    except Exception as e:  # noqa: BLE001 - both backends must fail alike
        return type(e), str(e)


def _assert_same(a, b):
    assert type(a) is type(b) or isinstance(a, float) and isinstance(b, float)
    assert a == b or (a != a and b != b), (a, b)


# ── ndtr / ndtri ───────────────────────────────────────────────────────────


def test_ndtr_and_ndtri_match_to_the_bit_and_scipy_closely():
    rng = np.random.default_rng(7)
    xs = np.concatenate(
        [rng.normal(0, 3, 50_000), rng.uniform(-38, 38, 20_000), [0, 1, -1, 8, -8]]
    ).tolist() + [math.inf, -math.inf, math.nan]
    for x in xs:
        _assert_same(_normal.ndtr(x), native.ndtr(x))
    ys = np.concatenate(
        [
            rng.uniform(0, 1, 50_000),
            10.0 ** rng.uniform(-300, 0, 20_000),
            1 - 10.0 ** rng.uniform(-16, 0, 10_000),
        ]
    ).tolist() + [0.0, 1.0, 0.5, 0.13533528323661269, -0.5, 1.5, math.nan, 5e-324]
    for y in ys:
        _assert_same(_normal.ndtri(y), native.ndtri(y))

    finite = [x for x in xs if -30 < x < 30]
    np.testing.assert_allclose(
        [_normal.ndtr(x) for x in finite], scipy_ndtr(finite), rtol=1e-13, atol=0
    )
    inner = [y for y in ys if 1e-300 < y < 1]
    np.testing.assert_allclose(
        [_normal.ndtri(y) for y in inner], scipy_ndtri(inner), rtol=1e-14, atol=0
    )


# ── Forwards and NDFs ──────────────────────────────────────────────────────

COLLATERALS = [pytest.param(c, id=k) for k, c in COLLATERAL_INDEXES.items()]
REPORTING = [Currency.CLP, Currency.USD]


@pytest.mark.parametrize("collateral", COLLATERALS)
@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("is_buy", [True, False])
def test_forward_mtm_matches(market, collateral, offset, is_buy):
    py, cpp = _both(
        lambda: Forward(
            1_250_000.0,
            951.37,
            VAL_DATE + timedelta(days=offset),
            is_buy,
            USD_CLP,
            collateral_index=collateral,
        )
    )
    for currency in REPORTING:
        _assert_same(
            _outcome(lambda: Calculator.valuate(cpp, market, currency)),
            _outcome(lambda: Calculator.valuate(py, market, currency)),
        )


NDF_DATES = [
    (VAL_DATE - timedelta(days=3), VAL_DATE + timedelta(days=2)),  # fixed
    (VAL_DATE, VAL_DATE + timedelta(days=1)),  # fixing today
    (VAL_DATE + timedelta(days=5), VAL_DATE + timedelta(days=7)),
    (UF_KNOWN_UNTIL, UF_KNOWN_UNTIL + timedelta(days=2)),  # UF published
    (VAL_DATE + timedelta(days=200), VAL_DATE + timedelta(days=202)),
    (VAL_DATE + timedelta(days=900), VAL_DATE + timedelta(days=930)),
    (VAL_DATE - timedelta(days=9), VAL_DATE),  # paid
]


@pytest.mark.parametrize("pair", [USD_CLP, UF_CLP], ids=["USDCLP", "CLFCLP"])
@pytest.mark.parametrize("collateral", COLLATERALS)
@pytest.mark.parametrize("dates", NDF_DATES)
@pytest.mark.parametrize("is_buy", [True, False])
def test_ndf_mtm_matches(market, pair, collateral, dates, is_buy):
    fixing, payment = dates
    strike = 37_731.2 if pair is UF_CLP else 948.15
    py, cpp = _both(
        lambda: NDF(
            1_000.0,
            strike,
            payment,
            is_buy,
            pair,
            collateral_index=collateral,
            fixing_date=fixing,
        )
    )
    for currency in REPORTING:
        _assert_same(
            _outcome(lambda: Calculator.valuate(cpp, market, currency)),
            _outcome(lambda: Calculator.valuate(py, market, currency)),
        )


def test_a_missing_curve_raises_the_same_error():
    empty = build_market()
    empty.curves.clear()
    for build in (
        lambda: Forward(1.0, 950.0, VAL_DATE + timedelta(days=30), True, USD_CLP),
        lambda: NDF(
            1.0,
            950.0,
            VAL_DATE + timedelta(days=32),
            True,
            USD_CLP,
            fixing_date=VAL_DATE + timedelta(days=30),
        ),
    ):
        py, cpp = _both(build)
        a = _outcome(lambda: Calculator.valuate(py, empty, Currency.CLP))
        b = _outcome(lambda: Calculator.valuate(cpp, empty, Currency.CLP))
        assert isinstance(a, tuple) and a == b


# ── Options ────────────────────────────────────────────────────────────────

LINEAR_360 = RateConvention(
    interest_convention=LinearInterestConvention,
    day_count_convention=ActualDayCountConvention,
    time_fraction_base=360,
)
CONVENTIONS = [pytest.param(None, id="default"), pytest.param(LINEAR_360, id="lin360")]
GREEKS = ["get_option_mtm", "get_option_delta", "get_option_gamma", "get_option_vega"]


def _curves(market):
    return (
        market.get_discount_curve(Currency.CLP),
        market.get_discount_curve(Currency.USD),
    )


@pytest.mark.parametrize("kind", [Call, Put])
@pytest.mark.parametrize("days", [1, 30, 91, 365, 1095])
@pytest.mark.parametrize("moneyness", [0.7, 0.95, 1.0, 1.08, 1.6])
@pytest.mark.parametrize("rc", CONVENTIONS)
def test_option_price_and_greeks_match(market, kind, days, moneyness, rc):
    domestic, foreign = _curves(market)
    spot = market.get_fx_rate(VAL_DATE, USD_CLP).value
    maturity = VAL_DATE + timedelta(days=days)
    py, cpp = _both(lambda: kind(2_500_000.0, spot * moneyness, maturity, USD_CLP))
    _assert_same(
        _options.get_option_log_moneyness(cpp, spot, domestic, foreign),
        _options.get_option_log_moneyness(py, spot, domestic, foreign),
    )
    for vol in (0.03, 0.115, 0.6):
        for name in GREEKS:
            f = getattr(_options, name)
            _assert_same(
                f(cpp, VAL_DATE, spot, vol, domestic, foreign, rc),
                f(py, VAL_DATE, spot, vol, domestic, foreign, rc),
            )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("delta", [0.05, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("days", [7, 182, 730])
@pytest.mark.parametrize("rc", CONVENTIONS)
def test_strike_from_delta_matches(market, sign, delta, days, rc):
    domestic, foreign = _curves(market)
    maturity = VAL_DATE + timedelta(days=days)
    args = (delta, 951.0, 0.12, domestic, foreign, maturity, sign, rc)
    out = []
    for backend in ("python", "cpp"):
        with settings.use_backend(backend, "derivatives"):
            out.append(_outcome(lambda: _options.get_strike_from_delta(*args)))
    _assert_same(*out)


def test_option_mtm_from_the_market_surface_matches(market):
    _, foreign = _curves(market)
    spot = market.get_fx_rate(VAL_DATE, USD_CLP).value
    days = [30, 91, 182, 365, 730]
    deltas = [10, 25, "atm", 75, 85]
    vols = [[0.14, 0.12, 0.11, 0.115, 0.13]] * len(days)
    surfaces = []
    for backend in ("python", "cpp"):
        with settings.use_backend(backend, "derivatives"):
            surfaces.append(VolatilitySurface(days, deltas, vols, foreign))
    for lm in (-0.2, -0.01, 0.0, 0.05, 0.3):
        for d in (15, 200, 700):
            _assert_same(
                surfaces[1].get_volatility(lm, d), surfaces[0].get_volatility(lm, d)
            )
    market.volatility_surfaces[USD_CLP] = surfaces[0]
    try:
        for d in (20, 100, 400, 700):
            for k in (0.9, 1.0, 1.1):
                for kind in (Call, Put):
                    py, cpp = _both(
                        lambda: kind(
                            1e6, spot * k, VAL_DATE + timedelta(days=d), USD_CLP
                        )
                    )
                    _assert_same(
                        Calculator.valuate(cpp, market, Currency.CLP),
                        Calculator.valuate(py, market, Currency.CLP),
                    )
    finally:
        market.volatility_surfaces.pop(USD_CLP)
