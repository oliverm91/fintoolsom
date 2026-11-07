from datetime import date

import pytest

from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.forwards.forwards import Forward, NDF
from fintoolsom.market import Currency, Market, OvernightRateIndex, UFIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate

USDCLP = CurrencyPair(Currency.USD, Currency.CLP)
# CLP riskless index; USD discounts on its own (non-riskless) currency curve.
# get_forward_mtm/get_ndf_mtm read both via market.get_funding_curve.
RISKLESS = OvernightRateIndex("OIS", currency=Currency.CLP)


def _market_with_fx_forward_setup(curve_date, zero_coupon_curve, spot: float) -> Market:
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
    )
    dispatched_mtm = Calculator.valuate(forward, market, Currency.CLP)
    direct_mtm = Calculator.get_forward_mtm(forward, market, Currency.CLP)
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
        fixing_date=payment_date,
    )
    dispatched_mtm = Calculator.valuate(ndf, market, Currency.CLP)
    direct_mtm = Calculator.get_ndf_mtm(ndf, market, Currency.CLP)
    assert dispatched_mtm == direct_mtm


def test_valuate_uf_ndf_is_not_routed(sample_zero_coupon_curve, curve_date):
    # A UF-indexed NDF needs a UF (CLF) curve + UF history, so valuate() deliberately
    # does not route it and raises — it must be valued via get_uf_forward_mtm directly.
    payment_date = date(2025, 1, 10)
    market = Market(t=curve_date)
    ndf = NDF(
        1_000,
        39_000.0,
        payment_date,
        is_buy=True,
        currency_pair=USDCLP,
        fixing_date=date(2024, 8, 10),
        is_uf_indexed=True,
    )
    with pytest.raises(NotImplementedError):
        Calculator.valuate(ndf, market, Currency.CLP)


def test_get_uf_forward_mtm_uses_known_uf(sample_zero_coupon_curve, curve_date):
    fixing_date = date(2024, 8, 10)
    payment_date = date(2025, 1, 10)
    strike = 39_000.0
    known_uf = 39_500.0
    uf_history = {fixing_date: known_uf}

    ndf = NDF(
        1_000,
        strike,
        payment_date,
        is_buy=True,
        currency_pair=USDCLP,
        fixing_date=fixing_date,
        is_uf_indexed=True,
    )
    mtm = Calculator.get_uf_forward_mtm(
        ndf, uf_history, sample_zero_coupon_curve, sample_zero_coupon_curve
    )
    # Known UF at the fixing → settlement is (known_uf - strike) discounted on the CLP curve.
    expected = 1_000 * (known_uf - strike) * sample_zero_coupon_curve.get_df(payment_date)
    assert mtm == pytest.approx(expected)


def test_valuate_raises_for_unsupported_instrument_type():
    with pytest.raises(NotImplementedError):
        Calculator.valuate(object(), market=None, currency=Currency.CLP)
