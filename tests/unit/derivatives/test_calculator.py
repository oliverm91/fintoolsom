from datetime import date
import pytest
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.forwards.forwards import Forward, NDF
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from fintoolsom.rates import DiscountCurve, IndexCurve


USDCLP = CurrencyPair(Currency.USD, Currency.CLP)


# CLP riskless index; USD discounts on its own (non-riskless) currency curve.
# get_forward_mtm/get_ndf_mtm read both via market.get_discount_curve.
RISKLESS = OvernightRateIndex("OIS", currency=Currency.CLP)


def _market_with_fx_forward_setup(curve_date, zero_coupon_curve, spot: float) -> Market:
    market = Market(
        t=curve_date,
        risk_free_index=RISKLESS,
        curves=[
            IndexCurve.from_zero_coupon_curve(
                zero_coupon_curve, RISKLESS
            ),  # CLP: the riskless index's own curve
            DiscountCurve.from_zero_coupon_curve(
                zero_coupon_curve, Currency.USD, RISKLESS
            ),  # USD: uncollateralised
        ],
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


def test_valuate_raises_for_unsupported_instrument_type():
    with pytest.raises(NotImplementedError):
        Calculator.valuate(object(), market=None, currency=Currency.CLP)
