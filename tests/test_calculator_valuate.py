from datetime import date

import pytest

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.forwards.forwards import Forward, NDF
from fintoolsom.market import Currency, Market, RateIndex, UFIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate

USDCLP = CurrencyPair(Currency.USD, Currency.CLP)
USD_INDEX = RateIndex("USD_OIS", currency=Currency.USD)
CLP_INDEX = RateIndex("CLP_OIS", currency=Currency.CLP)


def _market_with_fx_forward_setup(curve_date, zero_coupon_curve, spot: float) -> Market:
    market = Market(
        t=curve_date,
        curves={
            (USD_INDEX, Currency.USD): zero_coupon_curve,
            (CLP_INDEX, Currency.CLP): zero_coupon_curve,
        },
    )
    market.add_fx_rate(curve_date, FX_Rate(USDCLP, spot))
    return market


def test_valuate_forward_matches_get_forward_mtm(sample_zero_coupon_curve, curve_date):
    spot = 900.0
    strike = 800.0
    payment_date = date(2025, 1, 10)
    market = _market_with_fx_forward_setup(curve_date, sample_zero_coupon_curve, spot)

    forward = Forward(
        1_000_000,
        strike,
        payment_date,
        is_buy=True,
        currency_pair=USDCLP,
        domestic_index=CLP_INDEX,
        foreign_index=USD_INDEX,
    )
    dispatched_mtm = Calculator.valuate(forward, market)
    direct_mtm = Calculator.get_forward_mtm(
        forward, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert dispatched_mtm == direct_mtm


def test_valuate_ndf_matches_get_ndf_mtm(sample_zero_coupon_curve, curve_date):
    spot = 900.0
    strike = 800.0
    payment_date = date(2025, 1, 10)
    market = _market_with_fx_forward_setup(curve_date, sample_zero_coupon_curve, spot)

    ndf = NDF(
        1_000_000,
        strike,
        payment_date,
        is_buy=True,
        currency_pair=USDCLP,
        domestic_index=CLP_INDEX,
        foreign_index=USD_INDEX,
        fixing_date=payment_date,
    )
    dispatched_mtm = Calculator.valuate(ndf, market)
    direct_mtm = Calculator.get_ndf_mtm(
        ndf, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert dispatched_mtm == direct_mtm


def test_valuate_uf_ndf_uses_market_uf_history(sample_zero_coupon_curve, curve_date):
    fixing_date = date(2024, 8, 10)
    payment_date = date(2025, 1, 10)
    strike = 39_000.0
    known_uf = 39_500.0
    uf_index = UFIndex("UF", currency=Currency.CLP)

    market = Market(
        t=curve_date,
        curves={
            (CLP_INDEX, Currency.CLP): sample_zero_coupon_curve,
            (uf_index, Currency.CLP): sample_zero_coupon_curve,
        },
        uf_history={fixing_date: known_uf},
    )

    ndf = NDF(
        1_000,
        strike,
        payment_date,
        is_buy=True,
        is_uf_indexed=True,
        domestic_index=CLP_INDEX,
        foreign_index=uf_index,
        fixing_date=fixing_date,
    )
    mtm = Calculator.valuate(ndf, market)
    expected = Calculator.get_uf_forward_mtm(
        ndf, market.uf_history, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert mtm == expected


def test_valuate_raises_for_unsupported_instrument_type():
    with pytest.raises(NotImplementedError):
        Calculator.valuate(object(), market=None)
