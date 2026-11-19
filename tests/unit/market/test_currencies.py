import typing
from datetime import date

import pytest

from fintoolsom.market import Currency, CurrencyPair
from fintoolsom.market.currencies import (
    DEFAULT_FX_SPOT_LAG,
    FX_SPOT_LAGS,
    CurrencyName,
)


CLF, CLP, USD = Currency.CLF, Currency.CLP, Currency.USD


def test_clf_is_a_non_deliverable_currency_settled_in_clp():
    assert not CLF.is_deliverable and CLF.settlement_currency is CLP
    assert CLP.is_deliverable and CLP.settlement_currency is CLP


def test_every_currency_name_has_a_declared_constant():
    # The constants are assigned after the class; the ClassVar declarations let
    # type checkers see them, so they must list exactly the CurrencyName members.
    hints = typing.get_type_hints(Currency)
    declared = {n for n, h in hints.items() if typing.get_origin(h) is typing.ClassVar}
    assert declared == {c.name for c in CurrencyName}
    for c in CurrencyName:
        constant = getattr(Currency, c.name)
        assert isinstance(constant, Currency) and constant.name is c


@pytest.mark.parametrize(
    ("base", "quote", "lag"),
    [
        (Currency.USD, Currency.CLP, 1),
        (Currency.USD, Currency.CAD, 1),
        (Currency.EUR, Currency.USD, 2),
        (Currency.GBP, Currency.USD, 2),
        (Currency.USD, Currency.JPY, 2),
        (Currency.USD, Currency.BRL, DEFAULT_FX_SPOT_LAG),
    ],
)
def test_a_pair_spots_at_its_market_lag_in_either_order(base, quote, lag):
    assert CurrencyPair(base, quote).spot_lag == lag
    assert CurrencyPair(quote, base).spot_lag == lag


def test_the_spot_date_skips_holidays_of_both_currencies():
    # Thanksgiving 2026 (Thu 26 Nov) closes USD: a USD/CLP deal on Wed spots Fri.
    usdclp = CurrencyPair(Currency.USD, Currency.CLP)
    assert usdclp.get_spot_date(date(2026, 11, 25)) == date(2026, 11, 27)
    # EUR/USD, T+2 from Fri 20 Nov: Tue 24 Nov.
    eurusd = CurrencyPair(Currency.EUR, Currency.USD)
    assert eurusd.get_spot_date(date(2026, 11, 20)) == date(2026, 11, 24)


def test_a_spot_lag_convention_can_be_changed(monkeypatch):
    pair = CurrencyPair(Currency.USD, Currency.MXN)
    monkeypatch.setitem(FX_SPOT_LAGS, frozenset((Currency.USD, Currency.MXN)), 1)
    assert pair.spot_lag == 1


def test_a_spot_lag_does_not_change_pair_equality():
    assert CurrencyPair(Currency.USD, Currency.CLP) == CurrencyPair(
        Currency.USD, Currency.CLP
    )
    assert len({CurrencyPair(Currency.USD, Currency.CLP)}) == 1
