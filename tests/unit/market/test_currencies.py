from fintoolsom.market import Currency



CLF, CLP, USD = Currency.CLF, Currency.CLP, Currency.USD


def test_clf_is_a_non_deliverable_currency_settled_in_clp():
    assert not CLF.is_deliverable and CLF.settlement_currency is CLP
    assert CLP.is_deliverable and CLP.settlement_currency is CLP
