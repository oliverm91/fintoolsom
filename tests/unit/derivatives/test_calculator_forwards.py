from datetime import date

import pytest

from fintoolsom.derivatives.forwards.forwards import Forward, NDF
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from tests.factories.swap_factory import build_market
from tests.factories.curve_factory import (
    index_curve_from_zero_coupon_curve,
    discount_curve_from_zero_coupon_curve,
)


USDCLP = CurrencyPair(Currency.USD, Currency.CLP)


# CLP riskless index; USD discounts on its own (non-riskless) currency curve. get_forward_mtm/get_ndf_mtm derive the
# quote-ccy (domestic) and base-ccy (foreign) curves and the spot off the market.
RISKLESS = OvernightRateIndex("OIS", currency=Currency.CLP)


def _market(curve_date, zero_coupon_curve, spot: float) -> Market:
    market = Market(
        t=curve_date,
        risk_free_index=RISKLESS,
        curves=[
            index_curve_from_zero_coupon_curve(
                zero_coupon_curve, RISKLESS
            ),  # CLP: the riskless index's own curve
            discount_curve_from_zero_coupon_curve(
                zero_coupon_curve, Currency.USD, RISKLESS
            ),  # USD: uncollateralised
        ],
    )
    market.add_fx_rate(curve_date, FX_Rate(USDCLP, spot))
    return market


def test_forward_mtm_is_zero_at_fair_strike(sample_zero_coupon_curve, curve_date):
    spot = 900.0
    payment_date = date(2025, 1, 10)
    df_d = sample_zero_coupon_curve.get_df(payment_date)
    df_f = sample_zero_coupon_curve.get_df(payment_date)
    fair_strike = spot * df_f / df_d

    market = _market(curve_date, sample_zero_coupon_curve, spot)
    forward = Forward(
        1_000_000, fair_strike, payment_date, is_buy=True, currency_pair=USDCLP
    )
    mtm = Calculator.get_forward_mtm(forward, market, Currency.CLP)
    assert mtm == 0


def test_forward_mtm_sign_flips_with_is_buy(sample_zero_coupon_curve, curve_date):
    spot = 900.0
    strike = 800.0
    payment_date = date(2025, 1, 10)
    market = _market(curve_date, sample_zero_coupon_curve, spot)

    buy_forward = Forward(
        1_000_000, strike, payment_date, is_buy=True, currency_pair=USDCLP
    )
    sell_forward = Forward(
        1_000_000, strike, payment_date, is_buy=False, currency_pair=USDCLP
    )

    buy_mtm = Calculator.get_forward_mtm(buy_forward, market, Currency.CLP)
    sell_mtm = Calculator.get_forward_mtm(sell_forward, market, Currency.CLP)
    assert buy_mtm == -sell_mtm
    assert buy_mtm > 0  # spot > strike, buying is in the money


def test_ndf_mtm_matches_forward_when_fixing_equals_payment(
    sample_zero_coupon_curve, curve_date
):
    spot = 900.0
    strike = 800.0
    payment_date = date(2025, 1, 10)
    market = _market(curve_date, sample_zero_coupon_curve, spot)

    forward = Forward(
        1_000_000, strike, payment_date, is_buy=True, currency_pair=USDCLP
    )
    ndf = NDF(
        1_000_000,
        strike,
        payment_date,
        is_buy=True,
        currency_pair=USDCLP,
        fixing_date=payment_date,
    )

    forward_mtm = Calculator.get_forward_mtm(forward, market, Currency.CLP)
    ndf_mtm = Calculator.get_ndf_mtm(ndf, market, Currency.CLP)
    assert ndf_mtm == forward_mtm


@pytest.mark.parametrize(
    "fixing_date", [date(2024, 7, 1), date(2024, 7, 8)], ids=["past", "today"]
)
def test_ndf_fixed_but_unpaid_settles_at_the_realised_fixing(fixing_date):
    # get_future_fx_rate only takes dates after market.t: a fixed NDF reads the history.
    market = build_market()
    payment = date(2025, 7, 10)
    ndf = NDF(1e6, 900.0, payment, True, USDCLP, fixing_date=fixing_date)
    realised = market.get_fx_rate(fixing_date, USDCLP).value
    expected = 1e6 * (realised - 900.0) * market.get_df(Currency.CLP, payment)
    assert Calculator.valuate(ndf, market, Currency.CLP) == pytest.approx(
        expected, rel=1e-14
    )
