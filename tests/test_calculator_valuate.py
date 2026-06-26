from datetime import date

import pytest

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.forwards.forwards import Forward, NDF
from fintoolsom.market import Currency, Locality, Market
from fintoolsom.market.currencies import CurrencyPair, FX_Rate

USDCLP = CurrencyPair(Currency.USD, Currency.CLP)


def _market_with_fx_forward_setup(curve_date, zero_coupon_curve, spot: float) -> Market:
    market = Market(
        t=curve_date,
        zero_coupon_curve_mapper={
            "USD": {Locality.CL: zero_coupon_curve},
            "CLP": {Locality.CL: zero_coupon_curve},
        },
    )
    market.add_fx_rate(curve_date, FX_Rate(USDCLP, spot))
    return market, Locality.CL


def test_valuate_forward_matches_get_forward_mtm(sample_zero_coupon_curve, curve_date):
    spot = 900.0
    strike = 800.0
    payment_date = date(2025, 1, 10)
    market, locality = _market_with_fx_forward_setup(
        curve_date, sample_zero_coupon_curve, spot
    )

    forward = Forward(
        1_000_000, strike, payment_date, is_buy=True, currency_pair=USDCLP
    )
    dispatched_mtm = Calculator.valuate(forward, market, locality=locality)
    direct_mtm = Calculator.get_forward_mtm(
        forward, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert dispatched_mtm == direct_mtm


def test_valuate_ndf_matches_get_ndf_mtm(sample_zero_coupon_curve, curve_date):
    spot = 900.0
    strike = 800.0
    payment_date = date(2025, 1, 10)
    market, locality = _market_with_fx_forward_setup(
        curve_date, sample_zero_coupon_curve, spot
    )

    ndf = NDF(
        1_000_000,
        strike,
        payment_date,
        is_buy=True,
        currency_pair=USDCLP,
        fixing_date=payment_date,
    )
    dispatched_mtm = Calculator.valuate(ndf, market, locality=locality)
    direct_mtm = Calculator.get_ndf_mtm(
        ndf, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert dispatched_mtm == direct_mtm


def test_valuate_uf_ndf_uses_market_uf_history(sample_zero_coupon_curve, curve_date):
    fixing_date = date(2024, 8, 10)
    payment_date = date(2025, 1, 10)
    strike = 39_000.0
    known_uf = 39_500.0

    market = Market(
        t=curve_date,
        zero_coupon_curve_mapper={
            "CLP": {
                "UF": sample_zero_coupon_curve,
                Locality.CL: sample_zero_coupon_curve,
            }
        },
        uf_history={fixing_date: known_uf},
    )
    locality = Locality.CL

    ndf = NDF(
        1_000,
        strike,
        payment_date,
        is_buy=True,
        is_uf_indexed=True,
        fixing_date=fixing_date,
    )
    mtm = Calculator.valuate(ndf, market, locality=locality)
    expected = Calculator.get_uf_forward_mtm(
        ndf, market.uf_history, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert mtm == expected


def test_valuate_raises_for_unsupported_instrument_type():
    with pytest.raises(NotImplementedError):
        Calculator.valuate(object(), market=None)
