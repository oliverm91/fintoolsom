"""The C++ market lookups give the Python ones' results and errors, to the bit.

On a market created under the ``"cpp"`` backend of ``"market"``, an instrument with a
C++ core (legs, forwards) is valued in one C++ call that also resolves the curves and
FX rates (``fintoolsom._native.market_*``). Each case is valued all in Python (Python
instrument, Python market) and all in C++ (C++ instrument, C++ market), on the same
curves, and must agree exactly; so must the errors of a missing curve or FX rate.
"""

from datetime import timedelta

import pytest

from fintoolsom import settings
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.forwards.forwards import NDF, Forward
from fintoolsom.derivatives.swaps import (
    LegSide,
    NonStartedSimplifiedSwap,
    SimplifiedSwap,
)
from fintoolsom.market.currencies import Currency, CurrencyPair
from tests.factories.swap_factory import (
    COLLATERAL_INDEXES,
    FLOAT_INDEXES,
    REPORTING_CURRENCY,
    UF_CURRENCY,
    UF_KNOWN_UNTIL,
    VAL_DATE,
    SwapStatus,
    build_market,
    build_swap,
    swap_cases,
)

native = pytest.importorskip("fintoolsom._native")

USD_CLP = CurrencyPair(Currency.USD, Currency.CLP)
UF_CLP = CurrencyPair(UF_CURRENCY, Currency.CLP)


def _market(backend, rates):
    with settings.use_backend(rates, "rates"), settings.use_backend(backend, "market"):
        return build_market()


@pytest.fixture(scope="module", params=["python", "cpp"], ids=lambda b: f"{b}-curves")
def markets(request):
    """(Python market, C++ market) on the same curves."""
    py, cpp = _market("python", request.param), _market("cpp", request.param)
    assert not py._native and cpp._native
    return py, cpp


def _both(build):
    out = []
    for backend in ("python", "cpp"):
        with settings.use_backend(backend, "derivatives"):
            out.append(build())
    return out


def _outcome(f):
    try:
        return f()
    except Exception as e:  # noqa: BLE001 - both backends must fail alike
        return type(e), str(e)


def _swaps(case, wrap=None):
    def build():
        swap = build_swap(case)
        if wrap is None:
            return swap
        return wrap(
            receive_leg=swap.receive_leg,
            pay_leg=swap.pay_leg,
            collateral_index=swap.collateral_index,
        )

    return _both(build)


CASES = [pytest.param(c, id=c.id) for c in swap_cases()]
SIMPLIFIED = [pytest.param(c, id=c.id) for c in swap_cases() if not c.uses_uf]
NON_STARTED = [
    pytest.param(c, id=c.id)
    for c in swap_cases()
    if not c.uses_uf and c.status is SwapStatus.NOT_STARTED
]


@pytest.mark.parametrize("case", CASES)
def test_swap_mtm_and_leg_pvs_match(case, markets):
    (py_swap, cpp_swap), (py_market, cpp_market) = _swaps(case), markets
    for currency in (REPORTING_CURRENCY, Currency.USD):
        expected = _outcome(lambda: Calculator.valuate(py_swap, py_market, currency))
        assert _outcome(lambda: Calculator.valuate(cpp_swap, cpp_market, currency)) == (
            expected
        )
    for side in LegSide:
        assert Calculator.get_leg_pv(
            cpp_swap, side, cpp_market, REPORTING_CURRENCY
        ) == Calculator.get_leg_pv(py_swap, side, py_market, REPORTING_CURRENCY)


@pytest.mark.parametrize("case", SIMPLIFIED)
def test_simplified_swap_matches(case, markets):
    (py_swap, cpp_swap), (py_market, cpp_market) = _swaps(case, SimplifiedSwap), markets
    assert Calculator.valuate(cpp_swap, cpp_market, REPORTING_CURRENCY) == (
        Calculator.valuate(py_swap, py_market, REPORTING_CURRENCY)
    )


@pytest.mark.parametrize("case", NON_STARTED)
def test_non_started_swap_matches(case, markets):
    (py_swap, cpp_swap) = _swaps(case, NonStartedSimplifiedSwap)
    py_market, cpp_market = markets
    assert Calculator.valuate(cpp_swap, cpp_market, REPORTING_CURRENCY) == (
        Calculator.valuate(py_swap, py_market, REPORTING_CURRENCY)
    )


def test_a_started_non_started_swap_raises_the_same_error():
    case = NON_STARTED[0].values[0]
    py_swap, cpp_swap = _swaps(case, NonStartedSimplifiedSwap)
    errors = []
    for swap, backend in ((py_swap, "python"), (cpp_swap, "cpp")):
        market = _market(backend, "python")
        market.t = swap.effective_date + timedelta(days=1)
        errors.append(_outcome(lambda: Calculator.valuate(swap, market, Currency.CLP)))
    assert errors[0] == errors[1] and errors[0][0] is ValueError


@pytest.mark.parametrize("collateral", list(COLLATERAL_INDEXES.values()))
@pytest.mark.parametrize("offset", [-1, 0, 1, 33, 400])
def test_forwards_and_ndfs_match(markets, collateral, offset):
    py_market, cpp_market = markets
    pay = VAL_DATE + timedelta(days=offset)
    builds = [
        lambda: Forward(1e6, 951.4, pay, True, USD_CLP, collateral_index=collateral),
        lambda: NDF(
            1e6,
            949.0,
            pay + timedelta(days=2),
            False,
            USD_CLP,
            collateral_index=collateral,
            fixing_date=pay,
        ),
        lambda: NDF(
            10.0,
            37_800.0,
            pay + timedelta(days=2),
            True,
            UF_CLP,
            collateral_index=collateral,
            fixing_date=pay,
        ),
        lambda: NDF(
            10.0,
            37_800.0,
            UF_KNOWN_UNTIL + timedelta(days=1),
            True,
            UF_CLP,
            collateral_index=collateral,
            fixing_date=UF_KNOWN_UNTIL,
        ),
    ]
    for build in builds:
        py, cpp = _both(build)
        for currency in (Currency.CLP, Currency.USD):
            assert _outcome(lambda: Calculator.valuate(cpp, cpp_market, currency)) == (
                _outcome(lambda: Calculator.valuate(py, py_market, currency))
            )


def _missing(market, what):
    if what == "curves":
        market.curves.clear()
    elif what == "usd curve":
        market.curves.pop(next(k for k in market.curves if k.startswith("USD_")))
    elif what == "fx":
        market.fx_history.clear()
    elif what == "spot":
        for data in market.fx_history.values():
            data.historic_data.pop(VAL_DATE, None)
    elif what == "risk free":
        market.risk_free_index = None
    return market


@pytest.mark.parametrize("what", ["curves", "usd curve", "fx", "spot", "risk free"])
def test_missing_market_data_raises_the_same_error(what):
    instruments = [
        pytest.param(c, id=c.id).values[0]
        for c in swap_cases()
        if c.collateral != "uncoll" and "USD" in c.id
    ][:12]
    for case in instruments:
        for wrap in (None, SimplifiedSwap):
            if wrap is not None and case.uses_uf:
                continue
            py_swap, cpp_swap = _swaps(case, wrap)
            py = _outcome(
                lambda: Calculator.valuate(
                    py_swap, _missing(_market("python", "cpp"), what), Currency.CLP
                )
            )
            cpp = _outcome(
                lambda: Calculator.valuate(
                    cpp_swap, _missing(_market("cpp", "cpp"), what), Currency.CLP
                )
            )
            assert cpp == py
    py, cpp = _both(
        lambda: Forward(1.0, 950.0, VAL_DATE + timedelta(30), True, USD_CLP)
    )
    assert _outcome(
        lambda: Calculator.valuate(
            cpp, _missing(_market("cpp", "cpp"), what), Currency.CLP
        )
    ) == _outcome(
        lambda: Calculator.valuate(
            py, _missing(_market("python", "cpp"), what), Currency.CLP
        )
    )


def test_the_lookups_match_the_market_methods():
    market = _market("cpp", "cpp")
    for index in {*FLOAT_INDEXES.values(), *COLLATERAL_INDEXES.values()} - {None}:
        assert native.market_index_curve(market, index) is market.get_index_curve(index)
    for ccy in (Currency.CLP, Currency.USD, Currency.EUR, UF_CURRENCY, Currency.JPY):
        assert _outcome(lambda: native.market_discount_curve(market, ccy)) == _outcome(
            lambda: market.get_discount_curve(ccy)
        )
    for pair in (
        USD_CLP,
        USD_CLP.invert(),
        UF_CLP,
        CurrencyPair(Currency.CLP, Currency.CLP),
    ):
        assert native.market_fx_known_until(market, pair) == market.get_fx_known_until(
            pair
        )
        for d in (VAL_DATE, VAL_DATE - timedelta(days=40), UF_KNOWN_UNTIL):
            assert _outcome(lambda: native.market_fx_rate(market, d, pair)) == _outcome(
                lambda: market.get_fx_rate(d, pair).value
            )
    assert native.market_convert(market, 3.7, Currency.USD, Currency.CLP) == (
        market.convert_value_currency(3.7, Currency.USD, Currency.CLP)
    )


def test_a_cpp_market_values_in_one_cpp_call(monkeypatch):
    calls = []
    original = native.market_swap_mtm
    monkeypatch.setattr(
        native, "market_swap_mtm", lambda *a: calls.append(1) or original(*a)
    )
    case = CASES[0].values[0]
    _, swap = _swaps(case)
    Calculator.valuate(swap, _market("cpp", "cpp"), Currency.CLP)
    Calculator.valuate(swap, _market("python", "cpp"), Currency.CLP)
    assert calls == [1]
