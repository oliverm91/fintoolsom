from datetime import date
import pytest
from fintoolsom.derivatives.forwards.forwards import NDF, Forward
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair


CLF, CLP, USD = Currency.CLF, Currency.CLP, Currency.USD


CLFCLP = CurrencyPair(CLF, CLP)


@pytest.mark.parametrize(
    "pair", [CLFCLP, CurrencyPair(USD, CLF)], ids=["clf-base", "clf-quote"]
)
def test_a_deliverable_uf_forward_is_rejected(pair):
    with pytest.raises(ValueError, match="CLF cannot be delivered.*must be an NDF"):
        Forward(1_000, 38_000.0, date(2025, 1, 10), is_buy=True, currency_pair=pair)


def test_a_uf_forward_is_accepted_as_an_ndf():
    ndf = NDF(
        1_000,
        38_000.0,
        date(2025, 1, 10),
        is_buy=True,
        currency_pair=CLFCLP,
        fixing_date=date(2025, 1, 10),
    )
    assert ndf.currency_pair is CLFCLP
