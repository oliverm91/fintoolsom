from datetime import date
import pytest
from fintoolsom.derivatives.forwards.forwards import Forward
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair


CLF, CLP, USD = Currency.CLF, Currency.CLP, Currency.USD


CLFCLP = CurrencyPair(CLF, CLP)


def test_a_deliverable_uf_forward_is_rejected():
    with pytest.raises(ValueError, match="NDF"):
        Forward(1_000, 38_000.0, date(2025, 1, 10), is_buy=True, currency_pair=CLFCLP)
