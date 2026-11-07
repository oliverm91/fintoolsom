from datetime import date
from fintoolsom.derivatives.forwards.forwards import Forward, NDF
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate



USDCLP = CurrencyPair(Currency.USD, Currency.CLP)


# CLP riskless index; USD discounts on its own (non-riskless) currency curve. get_forward_mtm/get_ndf_mtm derive the
# quote-ccy (domestic) and base-ccy (foreign) curves and the spot off the market.
RISKLESS = OvernightRateIndex("OIS", currency=Currency.CLP)


def _market(curve_date, zero_coupon_curve, spot: float) -> Market:
    market = Market(
        t=curve_date,
        riskless_index=RISKLESS,
        curves={
            RISKLESS: zero_coupon_curve,  # CLP: the riskless index's own curve
            Currency.USD: zero_coupon_curve,  # USD: uncollateralised, non-riskless currency
        },
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


def test_ndf_mtm_matches_forward_when_fixing_equals_payment(sample_zero_coupon_curve, curve_date):
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
