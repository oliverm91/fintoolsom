from datetime import date

from fintoolsom.derivatives.forwards.forwards import Forward, NDF
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.market.currencies import Currency, CurrencyPair

USDCLP = CurrencyPair(Currency.USD, Currency.CLP)


def test_forward_mtm_is_zero_at_fair_strike(sample_zero_coupon_curve, curve_date):
    spot = 900.0
    payment_date = date(2025, 1, 10)
    df_d = sample_zero_coupon_curve.get_df(payment_date)
    df_f = sample_zero_coupon_curve.get_df(payment_date)
    fair_strike = spot * df_f / df_d

    forward = Forward(
        1_000_000, fair_strike, payment_date, is_buy=True, currency_pair=USDCLP
    )
    mtm = Calculator.get_forward_mtm(
        forward, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert mtm == 0


def test_forward_mtm_sign_flips_with_is_buy(sample_zero_coupon_curve):
    spot = 900.0
    strike = 800.0
    payment_date = date(2025, 1, 10)

    buy_forward = Forward(
        1_000_000, strike, payment_date, is_buy=True, currency_pair=USDCLP
    )
    sell_forward = Forward(
        1_000_000, strike, payment_date, is_buy=False, currency_pair=USDCLP
    )

    buy_mtm = Calculator.get_forward_mtm(
        buy_forward, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    sell_mtm = Calculator.get_forward_mtm(
        sell_forward, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert buy_mtm == -sell_mtm
    assert buy_mtm > 0  # spot > strike, buying is in the money


def test_ndf_mtm_matches_forward_when_fixing_equals_payment(sample_zero_coupon_curve):
    spot = 900.0
    strike = 800.0
    payment_date = date(2025, 1, 10)

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

    forward_mtm = Calculator.get_forward_mtm(
        forward, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    ndf_mtm = Calculator.get_ndf_mtm(
        ndf, spot, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    assert ndf_mtm == forward_mtm


def test_uf_forward_uses_known_uf_value_when_fixing_is_known(sample_zero_coupon_curve):
    fixing_date = date(2024, 8, 10)
    payment_date = date(2025, 1, 10)
    strike = 39_000.0
    known_uf = 39_500.0

    ndf = NDF(
        1_000,
        strike,
        payment_date,
        is_buy=True,
        is_uf_indexed=True,
        fixing_date=fixing_date,
    )
    uf_history = {fixing_date: known_uf}

    mtm = Calculator.get_uf_forward_mtm(
        ndf, uf_history, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    expected = (
        ndf.sign
        * ndf.notional
        * (known_uf - strike)
        * sample_zero_coupon_curve.get_df(payment_date)
    )
    assert mtm == expected


def test_uf_forward_projects_when_fixing_is_unknown(sample_zero_coupon_curve):
    last_known_date = date(2024, 8, 1)
    fixing_date = date(2024, 8, 10)
    payment_date = date(2025, 1, 10)
    strike = 39_000.0
    last_known_uf = 39_500.0

    ndf = NDF(
        1_000,
        strike,
        payment_date,
        is_buy=True,
        is_uf_indexed=True,
        fixing_date=fixing_date,
    )
    uf_history = {last_known_date: last_known_uf}

    mtm = Calculator.get_uf_forward_mtm(
        ndf, uf_history, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    # Same curve used for both legs => forward factor is 1, so the projected UF
    # should equal the last known UF value.
    expected = (
        ndf.sign
        * ndf.notional
        * (last_known_uf - strike)
        * sample_zero_coupon_curve.get_df(payment_date)
    )
    assert mtm == expected
