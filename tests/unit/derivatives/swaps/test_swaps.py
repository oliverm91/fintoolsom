from typing import Any
import pytest
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency
from tests.factories.swap_factory import (
    FIXED_RATES,
    NOTIONALS,
    PAYMENT_FREQUENCY,
    SOFR,
    SPREAD,
    START_DATES,
    SWAP_TERM_YEARS,
    Ccy,
    SwapStatus,
    _rate,
    adj_convention,
)


USD, CLP = Currency.USD, Currency.CLP


def _common(ccy: Ccy, status: SwapStatus) -> dict:
    return dict(
        notional=NOTIONALS[ccy],
        start_date=START_DATES[status],
        term=Term(SWAP_TERM_YEARS, TermUnit.Y, adj_convention(ccy)),
        frequency=PAYMENT_FREQUENCY,
        adj_convention=adj_convention(ccy),
    )


def _xccy_usd(
    status: SwapStatus = SwapStatus.NOT_STARTED, fx_fixing_lag: int = 0
) -> Any:
    return builders.xccy_floating_leg(
        index=SOFR,
        spread=SPREAD,
        fx_fixing_lag=fx_fixing_lag,
        **_common(Ccy.USD, status),
    )


def _fixed(
    ccy: Ccy, status: SwapStatus = SwapStatus.NOT_STARTED, fx_fixing_lag: int = 0
) -> Any:
    return builders.fixed_leg(
        rate=_rate(FIXED_RATES[ccy]),
        currency=USD if ccy is Ccy.USD else CLP,
        fx_fixing_lag=fx_fixing_lag,
        **_common(ccy, status),
    )


def test_settlement_currency_validation():
    usd, clp = _xccy_usd(), _fixed(Ccy.CLP)
    with pytest.raises(ValueError, match="payment_currency"):
        Swap(receive_leg=usd, pay_leg=clp)  # non-deliverable XCCY needs one
    with pytest.raises(ValueError, match="payment_currency"):
        Swap(receive_leg=usd, pay_leg=clp, is_deliverable=True, payment_currency=CLP)
    assert (
        Swap(receive_leg=usd, pay_leg=clp, is_deliverable=True).settlement_currency
        is None
    )
    assert (
        Swap(receive_leg=usd, pay_leg=clp, payment_currency=CLP).settlement_currency
        == CLP
    )
    Swap(receive_leg=clp, pay_leg=_fixed(Ccy.CLP))  # single currency: nothing to settle


def test_str_is_a_compact_summary():
    from tests.factories.swap_factory import build_swap, swap_cases

    for case in swap_cases():
        text = str(build_swap(case))
        lines = text.splitlines()
        assert lines[0].startswith("Swap  (")
        assert len(lines) == 5
        assert lines[1].lstrip().startswith("Receive")
        assert lines[3].lstrip().startswith("Pay")
        assert "coupons)" in lines[2] and "coupons)" in lines[4]
