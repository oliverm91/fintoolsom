"""Initial notional exchange of deliverable swaps.

``Swap(initial_exchange=True)`` exchanges the notionals on the first start date as
well as at maturity: each leg pays its first residual, in its own currency, on its
first start date. Physical (deliverable) swaps only.
"""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import Calculator, ValuationBatch
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.simplified import SimplifiedSwap
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.market import Currency
from fintoolsom.rates import Rate, RateConvention
from tests.factories.swap_factory import (
    FX_TO_CLP,
    ICP,
    SOFR,
    Ccy,
    adj_convention,
    build_market,
)

USD_NOTIONAL = 1_000_000.0
CLP_NOTIONAL = USD_NOTIONAL * FX_TO_CLP[Ccy.USD]
FORWARD_START = date(2024, 10, 8)
STARTED = date(2024, 4, 8)


@pytest.fixture(scope="module")
def market():
    return build_market()


def _legs(start: date):
    usd, clp = adj_convention(Ccy.USD), adj_convention(Ccy.CLP)
    term = Term(2, TermUnit.Y, usd)
    receive = builders.fixed_leg(
        USD_NOTIONAL,
        start,
        term,
        "SEMIANNUAL",
        usd,
        Rate(RateConvention(), 0.045),
        Currency.USD,
    )
    pay = builders.overnight_leg(
        CLP_NOTIONAL, start, Term(2, TermUnit.Y, clp), "SEMIANNUAL", clp, ICP
    )
    return receive, pay


def _swap(start=FORWARD_START, collateral=None, **kwargs) -> Swap:
    receive, pay = _legs(start)
    return Swap(
        receive_leg=receive,
        pay_leg=pay,
        is_deliverable=True,
        collateral_index=collateral,
        **kwargs,
    )


def test_a_non_deliverable_swap_cannot_exchange_its_notionals_at_the_start():
    receive, pay = _legs(FORWARD_START)
    with pytest.raises(ValueError, match="only for deliverable"):
        Swap(
            receive_leg=receive,
            pay_leg=pay,
            payment_currency=Currency.CLP,
            initial_exchange=True,
        )


@pytest.mark.parametrize("collateral", [None, SOFR, ICP], ids=["uncoll", "sofr", "icp"])
def test_a_forward_start_adds_each_legs_notional_paid_on_its_start(market, collateral):
    without = Calculator.valuate(_swap(collateral=collateral), market, Currency.CLP)
    swap = _swap(collateral=collateral, initial_exchange=True)
    with_exchange = Calculator.valuate(swap, market, Currency.CLP)

    df_usd = market.get_df(Currency.USD, FORWARD_START, collateral=collateral)
    df_clp = market.get_df(Currency.CLP, FORWARD_START, collateral=collateral)
    # Receive USD leg pays its notional at the start; pay CLP leg receives its own.
    exchange = -USD_NOTIONAL * df_usd * FX_TO_CLP[Ccy.USD] + CLP_NOTIONAL * df_clp
    assert with_exchange - without == pytest.approx(exchange, rel=1e-9)
    assert abs(exchange) > 1_000  # USD and CLP discount differently


def test_the_exchange_is_paid_once_the_swap_has_started(market):
    started = Calculator.valuate(
        _swap(STARTED, initial_exchange=True), market, Currency.CLP
    )
    assert started == Calculator.valuate(_swap(STARTED), market, Currency.CLP)


def test_single_currency_exchanges_cancel(market):
    clp = adj_convention(Ccy.CLP)
    term = Term(2, TermUnit.Y, clp)
    fixed = builders.fixed_leg(
        CLP_NOTIONAL,
        FORWARD_START,
        term,
        "SEMIANNUAL",
        clp,
        Rate(RateConvention(), 0.06),
        Currency.CLP,
    )
    floating = builders.overnight_leg(
        CLP_NOTIONAL, FORWARD_START, term, "SEMIANNUAL", clp, ICP
    )

    def mtm(exchange: bool) -> float:
        swap = Swap(
            receive_leg=fixed,
            pay_leg=floating,
            is_deliverable=True,
            initial_exchange=exchange,
        )
        return Calculator.valuate(swap, market, Currency.CLP)

    assert mtm(True) == pytest.approx(mtm(False), abs=1e-6)


def test_a_simplified_swap_values_the_exchange_like_the_swap(market):
    swap = _swap(initial_exchange=True)
    simplified = SimplifiedSwap(
        receive_leg=swap.receive_leg, pay_leg=swap.pay_leg, initial_exchange=True
    )
    assert Calculator.valuate(simplified, market, Currency.CLP) == pytest.approx(
        Calculator.valuate(swap, market, Currency.CLP), rel=1e-12
    )


def test_the_swap_says_it_exchanges_at_the_start():
    assert "initial exchange" in str(_swap(initial_exchange=True))


@pytest.mark.parametrize("currency", [Currency.CLP, Currency.USD])
def test_the_batch_values_initial_exchanges_as_the_calculator_to_the_bit(
    market, currency
):
    swaps = [
        _swap(initial_exchange=True),
        _swap(collateral=SOFR, initial_exchange=True),
        _swap(STARTED, initial_exchange=True),
        _swap(),
    ]
    batch = ValuationBatch(swaps, market, currency)  # the started one falls back
    assert batch.values().tolist() == [
        Calculator.valuate(s, market, currency) for s in swaps
    ]


def test_the_batch_gradient_includes_the_exchange(market):
    batch = ValuationBatch([_swap(initial_exchange=True)], market, Currency.CLP)
    values, base = batch.source_values(), batch.values()[0]
    gradient = batch.gradient(values)[0]
    usd_curve = market.get_discount_curve(Currency.USD)
    market.set_curve(Currency.USD, usd_curve.parallel_bump_bps(1))
    try:
        moved, change = batch.source_values() - values, batch.values()[0] - base
    finally:
        market.set_curve(Currency.USD, usd_curve)
    assert gradient @ moved == pytest.approx(change, rel=1e-3)
    # The exchange's df term is in the gradient: dropping it misses the change.
    no_exchange = ValuationBatch([_swap()], market, Currency.CLP)
    assert not np.allclose(gradient, no_exchange.gradient(values)[0])


def test_replacing_a_swap_keeps_its_initial_exchange():
    swap = replace(_swap(initial_exchange=True), collateral_index=SOFR)
    assert swap.initial_exchange
