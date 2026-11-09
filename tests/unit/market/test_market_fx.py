from datetime import date
import pytest
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from tests.factories.swap_factory import (
    FX_TO_CLP,
    VAL_DATE,
    Ccy,
    build_market,
)


USD, CLP = Currency.USD, Currency.CLP


SPOT = FX_TO_CLP[Ccy.USD]


@pytest.fixture(scope="module")
def market():
    return build_market()


def test_market_fx_lookup_inverts_and_is_identity_on_same_currency(market):
    assert market.get_fx_rate(VAL_DATE, CurrencyPair(CLP, USD)).value == pytest.approx(
        1 / SPOT
    )
    assert market.get_fx_rate(VAL_DATE, CurrencyPair(CLP, CLP)).value == 1.0
    market.add_fx_rate(date(2000, 1, 3), FX_Rate(CurrencyPair(Currency.EUR, USD), 1.1))
    assert market.get_fx_rate(
        date(2000, 1, 3), CurrencyPair(USD, Currency.EUR)
    ).value == pytest.approx(1 / 1.1)
