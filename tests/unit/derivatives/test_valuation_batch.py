"""A ValuationBatch gives, to the bit, what Calculator.valuate gives one by one.

Every swap case of the factory (started, not started, finished; with or without
spread; every collateral; deliverable or not), simplified swaps, forwards and NDFs (the
compiled batch, and the instruments left to the calculator), before and after the
curves move.
"""

from datetime import timedelta

import numpy as np
import pytest

from fintoolsom.curve_builder import _solver, build_curves
from fintoolsom.derivatives.calculator import Calculator, ValuationBatch
from fintoolsom.derivatives.forwards.forwards import NDF, Forward
from fintoolsom.derivatives.swaps import NonStartedSimplifiedSwap, SimplifiedSwap
from fintoolsom.market.currencies import Currency, CurrencyPair
from fintoolsom.rates import InterpolationMethod
from tests.factories import quote_factory
from tests.factories.swap_factory import (
    COLLATERAL_INDEXES,
    UF_CURRENCY,
    UF_KNOWN_UNTIL,
    VAL_DATE,
    SwapStatus,
    build_market,
    build_swap,
    swap_cases,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

USD_CLP = CurrencyPair(Currency.USD, Currency.CLP)
UF_CLP = CurrencyPair(UF_CURRENCY, Currency.CLP)


def _outcome(f):
    try:
        return f()
    except Exception as e:  # noqa: BLE001 - both ways must fail alike
        return type(e), str(e)


def _fx_instruments():
    out = []
    for collateral in COLLATERAL_INDEXES.values():
        for offset in (-1, 0, 1, 33, 400):
            pay = VAL_DATE + timedelta(days=offset)
            out += [
                Forward(1e6, 951.4, pay, True, USD_CLP, collateral_index=collateral),
                NDF(
                    1e6,
                    949.0,
                    pay + timedelta(days=2),
                    False,
                    USD_CLP,
                    collateral_index=collateral,
                    fixing_date=pay,
                ),
                NDF(
                    10.0,
                    37_800.0,
                    pay + timedelta(days=2),
                    True,
                    UF_CLP,
                    collateral_index=collateral,
                    fixing_date=pay,
                ),
            ]
        out.append(
            NDF(
                10.0,
                37_800.0,
                UF_KNOWN_UNTIL + timedelta(days=1),
                True,
                UF_CLP,
                collateral_index=collateral,
                fixing_date=UF_KNOWN_UNTIL,
            )
        )
    return out


def _instruments():
    out = []
    for case in swap_cases():
        swap = build_swap(case)
        out.append(swap)
        if not case.uses_uf:
            legs = dict(
                receive_leg=swap.receive_leg,
                pay_leg=swap.pay_leg,
                collateral_index=swap.collateral_index,
            )
            out.append(SimplifiedSwap(**legs))
            if case.status is SwapStatus.NOT_STARTED:
                out.append(NonStartedSimplifiedSwap(**legs))
    return out + _fx_instruments()


def _check(instruments, market, currency):
    """The batch, compiled once, against the calculator: values to the bit, or the
    calculator's error."""
    expected = [
        _outcome(lambda i=i: Calculator.valuate(i, market, currency))
        for i in instruments
    ]
    batch = ValuationBatch(instruments, market, currency)
    if all(isinstance(e, float) for e in expected):
        assert batch.values().tolist() == expected
        assert (
            Calculator.valuate_many(instruments, market, currency).tolist() == expected
        )
    else:
        first_error = next(e for e in expected if not isinstance(e, float))
        assert _outcome(batch.values) == first_error
    return batch


def test_every_instrument_matches_the_calculator():
    market = build_market()
    instruments = _instruments()
    for currency in (Currency.CLP, Currency.USD):
        batch = _check(instruments, market, currency)
    # Started swaps are left to the calculator; the rest is compiled.
    assert batch._compiled and batch._fallback


def test_the_batch_reads_the_curves_as_they_move():
    market = build_market()
    instruments = _instruments()
    batch = ValuationBatch(instruments, market, Currency.CLP)
    before = batch.values()
    for curve in market.curves.values():
        curve._set_forward_rates(curve.forward_rates * 1.25 + 0.001)
    after = batch.values()
    assert not np.array_equal(before, after)
    assert after.tolist() == [
        Calculator.valuate(i, market, Currency.CLP) for i in instruments
    ]


@pytest.mark.parametrize("what", ["curves", "usd curve", "spot"])
def test_missing_market_data_raises_the_calculator_error(what):
    market = build_market()
    instruments = _instruments()
    batch = ValuationBatch(instruments, market, Currency.CLP)
    if what == "curves":
        market.curves.clear()
    elif what == "usd curve":
        market.curves.pop(next(k for k in market.curves if k.startswith("USD_")))
    else:
        for data in market.fx_history.values():
            data.historic_data.pop(VAL_DATE, None)
        # The spot rates were read when compiling: compile again.
        batch = ValuationBatch(instruments, market, Currency.CLP)
    expected = next(
        e
        for e in (
            _outcome(lambda i=i: Calculator.valuate(i, market, Currency.CLP))
            for i in instruments
        )
        if not isinstance(e, float)
    )
    assert _outcome(batch.values) == expected


@pytest.mark.parametrize(
    "method", [InterpolationMethod.PiecewiseConstant, InterpolationMethod.CubicSpline]
)
def test_the_solver_values_through_batches(monkeypatch, method):
    """The solver's residuals come from batches, which are the calculator's values."""
    calls = []
    values = ValuationBatch.values

    def checked(self):
        out = values(self)
        expected = [
            Calculator.valuate(i, self.market, self.currency) for i in self.instruments
        ]
        assert out.tolist() == expected
        calls.append(len(out))
        return out

    monkeypatch.setattr(ValuationBatch, "values", checked)
    build_curves(
        quote_factory.load_quotes(),
        quote_factory.quote_market(),
        interp_method=method,
    )
    assert calls and _solver.ValuationBatch is ValuationBatch
