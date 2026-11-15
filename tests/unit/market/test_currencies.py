import typing

from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyName


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
