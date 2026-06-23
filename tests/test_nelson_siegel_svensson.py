from datetime import date, timedelta

import pytest

from fintoolsom.fixedIncome import Bond, Coupon, Coupons, NelsonSiegelSvensson
from fintoolsom.rates import Rate, RateConvention, CompoundedInterestConvention
from fintoolsom.dates import ActualDayCountConvention


def _zero_coupon_bond(calibration_date: date, years: int, notional: float = 1_000_000) -> Bond:
    convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    maturity = calibration_date + timedelta(days=365 * years)
    coupon = Coupon(100, 5, 100, calibration_date, maturity, convention)
    return Bond(coupons=Coupons([coupon]), currency='clp', notional=notional)


def test_nss_calibrate_fits_market_irrs():
    calibration_date = date(2024, 7, 8)
    convention = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    bonds_irr_list = [
        (_zero_coupon_bond(calibration_date, 1), Rate(convention, 0.05)),
        (_zero_coupon_bond(calibration_date, 3), Rate(convention, 0.052)),
        (_zero_coupon_bond(calibration_date, 5), Rate(convention, 0.054)),
        (_zero_coupon_bond(calibration_date, 10), Rate(convention, 0.056)),
    ]

    nss = NelsonSiegelSvensson()
    nss.calibrate(calibration_date, bonds_irr_list)

    for bond, irr in bonds_irr_list:
        market_pv = bond.get_present_value(calibration_date, irr)
        t = (bond.get_maturity_date() - calibration_date).days / 365
        ns_pv = bond.coupons.get_flows()[0] * nss.get_df(t)
        assert ns_pv == pytest.approx(market_pv, rel=0.02)
