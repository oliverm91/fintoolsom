from datetime import date, timedelta
import numpy as np
import pytest
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.market import Currency
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from tests.factories.swap_factory import (
    FX_TO_CLP,
    PAST_FX_FACTOR,
    UF_KNOWN_UNTIL,
    VAL_DATE,
    Ccy,
    build_market,
    uf_level,
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


# --- FX projection and conversion ---------------------------------------------

USDCLP, CLFCLP = CurrencyPair(USD, CLP), CurrencyPair(Currency.CLF, CLP)
FUTURE = date(2025, 7, 8)


def _funding_df(market, ccy, d):
    return market.get_discount_curve(ccy).get_df(d)


def test_future_fx_rate_is_spot_carried_on_the_funding_curves(market):
    fx = market.get_future_fx_rate(FUTURE, USDCLP)
    expected = (
        SPOT * _funding_df(market, USD, FUTURE) / _funding_df(market, CLP, FUTURE)
    )
    assert isinstance(fx, FX_Rate) and fx.currency_pair == USDCLP
    assert fx.value == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("t", [date(2024, 1, 8), VAL_DATE], ids=["past", "today"])
def test_future_fx_rate_needs_a_date_after_the_market_date(market, t):
    with pytest.raises(ValueError, match="after the market date"):
        market.get_future_fx_rate(t, USDCLP)
    with pytest.raises(ValueError, match="after the market date"):
        market.convert_future_value_currency(2.0, USD, CLP, t)


def test_future_uf_reads_published_levels_then_projects_from_the_last_one(market):
    published = date(2024, 8, 1)
    assert published <= UF_KNOWN_UNTIL
    assert market.get_future_fx_rate(published, CLFCLP).value == pytest.approx(
        uf_level(published), rel=1e-15
    )
    clf, L = Currency.CLF, UF_KNOWN_UNTIL
    expected = (
        uf_level(L)
        * (_funding_df(market, clf, FUTURE) / _funding_df(market, clf, L))
        / (_funding_df(market, CLP, FUTURE) / _funding_df(market, CLP, L))
    )
    assert market.get_future_fx_rate(FUTURE, CLFCLP).value == pytest.approx(
        expected, rel=1e-14
    )


def test_future_fx_rate_of_the_inverse_pair_is_the_inverse(market):
    direct = market.get_future_fx_rate(FUTURE, USDCLP).value
    inverse = market.get_future_fx_rate(FUTURE, CurrencyPair(CLP, USD)).value
    assert inverse == pytest.approx(1 / direct, rel=1e-14)


def test_future_fx_rate_of_a_currency_against_itself_is_one(market):
    assert market.get_future_fx_rate(FUTURE, CurrencyPair(USD, USD)).value == 1.0


def test_future_fx_rate_of_an_unloaded_pair_raises(market):
    with pytest.raises(KeyError):
        market.get_future_fx_rate(FUTURE, CurrencyPair(Currency.EUR, CLP))


def test_vectorised_fx_rates_read_the_history_then_project(market):
    past, future = [date(2024, 1, 8), VAL_DATE], [date(2024, 12, 9), FUTURE]
    fx = market._future_fx_rates_at_ordinals(
        USDCLP, np.array([d.toordinal() for d in past + future])
    )
    expected = [market.get_fx_rate(d, USDCLP).value for d in past] + [
        market.get_future_fx_rate(d, USDCLP).value for d in future
    ]
    np.testing.assert_array_equal(fx, expected)
    assert fx[0] == pytest.approx(SPOT * PAST_FX_FACTOR, rel=1e-15)


def test_calculator_fx_fixings_delegate_to_the_market(market):
    np.testing.assert_array_equal(
        Calculator.get_fx_fixings(USD, CLP, [VAL_DATE, FUTURE], market),
        [SPOT, market.get_future_fx_rate(FUTURE, USDCLP).value],
    )


def test_convert_value_currency_uses_spot(market):
    assert market.convert_value_currency(2.0, USD, CLP) == pytest.approx(2.0 * SPOT)
    assert market.convert_value_currency(2.0, CLP, USD) == pytest.approx(2.0 / SPOT)
    assert market.convert_value_currency(2.0, CLP, CLP) == 2.0


def test_convert_future_value_currency_uses_the_fx_fixing_of_its_date(market):
    fx = market.get_future_fx_rate(FUTURE, USDCLP).value
    assert market.convert_future_value_currency(2.0, USD, CLP, FUTURE) == 2.0 * fx


def test_future_uf_on_its_last_known_date_is_the_published_level(market):
    L = UF_KNOWN_UNTIL
    assert market.get_fx_known_until(CLFCLP) == L
    assert market.get_future_fx_rate(L, CLFCLP).value == uf_level(L)
    # The day after is the first projected one, carried from UF(L).
    nxt = L + timedelta(days=1)
    clf = Currency.CLF
    expected = (
        uf_level(L)
        * (_funding_df(market, clf, nxt) / _funding_df(market, clf, L))
        / (_funding_df(market, CLP, nxt) / _funding_df(market, CLP, L))
    )
    assert market.get_future_fx_rate(nxt, CLFCLP).value == pytest.approx(
        expected, rel=1e-14
    )


def test_future_fx_rate_without_the_funding_curve_raises():
    market = build_market()
    del market.curves["CLF_ICP"]
    with pytest.raises(KeyError, match="CLF_ICP"):
        market.get_future_fx_rate(FUTURE, CLFCLP)


def test_future_fx_rate_without_the_rate_at_the_last_known_date_raises():
    market = build_market()
    del market.fx_history[USDCLP].historic_data[VAL_DATE]
    assert market.get_fx_known_until(USDCLP) == VAL_DATE
    with pytest.raises(KeyError, match=str(VAL_DATE)):
        market.get_future_fx_rate(FUTURE, USDCLP)


def test_convert_value_currency_of_an_unloaded_pair_raises(market):
    with pytest.raises(KeyError):
        market.convert_value_currency(2.0, Currency.EUR, CLP)
