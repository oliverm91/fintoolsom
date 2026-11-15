from datetime import date

import pytest

from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.swaps import (
    NonStartedSimplifiedSwap,
    SimplifiedSwap,
    Swap,
    TermRateLeg,
    builders,
)
from fintoolsom.market import Currency
from tests.factories.swap_factory import (
    CAMARA,
    REPORTING_CURRENCY,
    SPREAD,
    VAL_DATE,
    Ccy,
    SwapStatus,
    _rate,
    adj_convention,
    build_market,
    build_swap,
    swap_cases,
)

ADJ_CLP = adj_convention(Ccy.CLP)


@pytest.fixture(scope="module")
def market():
    return build_market()


def _same_path_cases(status: SwapStatus | None = None):
    """Cases the full path values like a simplified swap: no UF leg, each leg in
    its own currency (IRS, or deliverable CCS) and no payment lag."""
    return [
        pytest.param(c, id=c.id)
        for c in swap_cases()
        if not c.uses_uf
        and (c.swap_type == "irs" or c.deliverable)
        and (status is None or c.status is status)
    ]


def _simplified(swap: Swap, cls=SimplifiedSwap):
    return cls(
        receive_leg=swap.receive_leg,
        pay_leg=swap.pay_leg,
        collateral_index=swap.collateral_index,
    )


def _no_term_fixing_set(swap: Swap) -> bool:
    return all(
        min(leg.fixing_dates) > VAL_DATE
        for leg in (swap.receive_leg, swap.pay_leg)
        if isinstance(leg, TermRateLeg)
    )


@pytest.mark.parametrize("case", _same_path_cases())
def test_simplified_swap_values_as_the_full_path(case, market):
    swap = build_swap(case)
    expected = Calculator.valuate(swap, market, REPORTING_CURRENCY)
    simplified = _simplified(swap)
    assert Calculator.valuate(simplified, market, REPORTING_CURRENCY) == pytest.approx(
        expected, rel=1e-12, abs=1e-9
    )
    # The full path agrees too: a simplified swap settles every leg in its currency.
    assert Calculator.get_swap_mtm(
        simplified, market, REPORTING_CURRENCY
    ) == pytest.approx(expected, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("case", _same_path_cases(SwapStatus.NOT_STARTED))
def test_non_started_swap_values_as_the_full_path(case, market):
    swap = build_swap(case)
    if not _no_term_fixing_set(swap):
        pytest.skip("a term coupon already fixed: the non-started path projects it")
    expected = Calculator.valuate(swap, market, REPORTING_CURRENCY)
    non_started = _simplified(swap, NonStartedSimplifiedSwap)
    assert non_started.valuate(market, REPORTING_CURRENCY) == pytest.approx(
        expected, rel=1e-12, abs=1e-9
    )


def _clp_leg(builder, **kw):
    return builder(
        notional=1_000.0,
        start_date=date(2024, 7, 15),
        term=Term(2, TermUnit.Y, ADJ_CLP),
        frequency="QUARTERLY",
        adj_convention=ADJ_CLP,
        **kw,
    )


def test_payment_lag_is_dropped_without_touching_the_original_legs(market):
    fixed = _clp_leg(
        builders.fixed_leg, rate=_rate(0.05), currency=Currency.CLP, payment_lag=2
    )
    floating = _clp_leg(
        builders.overnight_leg, index=CAMARA, spread=SPREAD, payment_lag=2
    )
    swap = SimplifiedSwap(receive_leg=fixed, pay_leg=floating)
    for leg, original in ((swap.receive_leg, fixed), (swap.pay_leg, floating)):
        assert leg is not original and leg.payment_dates == leg.end_dates
        assert leg.fx_fixing_dates == leg.end_dates
        assert original.payment_dates != original.end_dates

    no_lag = Swap(
        receive_leg=_clp_leg(builders.fixed_leg, rate=_rate(0.05), currency=Currency.CLP),
        pay_leg=_clp_leg(builders.overnight_leg, index=CAMARA, spread=SPREAD),
        payment_currency=Currency.CLP,
    )
    assert swap.valuate(market, REPORTING_CURRENCY) == pytest.approx(
        no_lag.valuate(market, REPORTING_CURRENCY), rel=1e-12
    )


def test_legs_without_a_payment_lag_are_kept():
    leg = _clp_leg(builders.fixed_leg, rate=_rate(0.05), currency=Currency.CLP)
    other = _clp_leg(builders.overnight_leg, index=CAMARA)
    swap = SimplifiedSwap(receive_leg=leg, pay_leg=other)
    assert swap.receive_leg is leg and swap.pay_leg is other


def test_simplified_swap_has_no_settlement_currency():
    with pytest.raises(TypeError):
        SimplifiedSwap(  # type: ignore[call-arg]
            receive_leg=_clp_leg(builders.overnight_leg, index=CAMARA),
            pay_leg=_clp_leg(builders.overnight_leg, index=CAMARA),
            payment_currency=Currency.CLP,
        )
    swap = SimplifiedSwap(
        receive_leg=_clp_leg(builders.overnight_leg, index=CAMARA),
        pay_leg=_clp_leg(builders.overnight_leg, index=CAMARA),
    )
    assert swap.is_deliverable and swap.settlement_currency is None
    assert swap.leg_settlement_currency(swap.pay_leg) == Currency.CLP


def test_a_uf_leg_settles_in_uf(market):
    uf_leg = _clp_leg(builders.fixed_leg, rate=_rate(0.02), currency=Currency.CLF)
    clp_leg = _clp_leg(builders.overnight_leg, index=CAMARA)
    with pytest.raises(ValueError, match="cannot be delivered"):
        Swap(receive_leg=uf_leg, pay_leg=clp_leg, is_deliverable=True)

    swap = NonStartedSimplifiedSwap(receive_leg=uf_leg, pay_leg=clp_leg)
    assert swap.leg_settlement_currency(uf_leg) == Currency.CLF
    uf_pv = float(
        uf_leg.flows @ market._dfs_at_ordinals(Currency.CLF, uf_leg._payment_ordinals)
    )
    clp_pv = Calculator._leg_pv(clp_leg, None, market)
    expected = (
        market.convert_value_currency(uf_pv, Currency.CLF, REPORTING_CURRENCY) - clp_pv
    )
    assert swap.valuate(market, REPORTING_CURRENCY) == pytest.approx(
        expected, rel=1e-12
    )


def test_effective_date_is_the_first_accrual_start():
    later = _clp_leg(builders.overnight_leg, index=CAMARA)
    earlier = builders.overnight_leg(
        notional=1_000.0,
        start_date=date(2024, 7, 10),
        term=Term(2, TermUnit.Y, ADJ_CLP),
        frequency="QUARTERLY",
        adj_convention=ADJ_CLP,
        index=CAMARA,
    )
    swap = NonStartedSimplifiedSwap(receive_leg=later, pay_leg=earlier)
    assert swap.effective_date == date(2024, 7, 10)


def _started_on(start: date) -> NonStartedSimplifiedSwap:
    leg = builders.overnight_leg(
        notional=1_000.0,
        start_date=start,
        term=Term(2, TermUnit.Y, ADJ_CLP),
        frequency="QUARTERLY",
        adj_convention=ADJ_CLP,
        index=CAMARA,
        spread=SPREAD,
    )
    fixed = builders.fixed_leg(
        notional=1_000.0,
        start_date=start,
        term=Term(2, TermUnit.Y, ADJ_CLP),
        frequency="QUARTERLY",
        adj_convention=ADJ_CLP,
        rate=_rate(0.05),
        currency=Currency.CLP,
    )
    return NonStartedSimplifiedSwap(receive_leg=fixed, pay_leg=leg)


def test_non_started_swap_raises_when_valued_after_it_starts(market):
    swap = _started_on(date(2024, 7, 1))
    assert swap.effective_date < VAL_DATE
    with pytest.raises(ValueError, match="after it starts"):
        Calculator.valuate(swap, market, REPORTING_CURRENCY)
    # The other paths still value it.
    simplified = _simplified(swap)
    assert Calculator.valuate(simplified, market, REPORTING_CURRENCY) == pytest.approx(
        Calculator.get_swap_mtm(swap, market, REPORTING_CURRENCY), rel=1e-12
    )


def test_non_started_swap_starting_on_the_valuation_date_values_as_the_full_path(
    market,
):
    swap = _started_on(VAL_DATE)
    expected = Calculator.get_swap_mtm(swap, market, REPORTING_CURRENCY)
    assert Calculator.valuate(swap, market, REPORTING_CURRENCY) == pytest.approx(
        expected, rel=1e-12
    )
