"""Tests against real market data for BTP0470930 (CLP nominal) and BTU0190930 (UF-indexed),
both 12.5y bullet Chilean treasury bonds issued 2018-03-01, maturing 2030-09-01, semi-annual
coupons on 01-03/01-09. Expected values taken from a market valuation tool screenshot dated
2026-06-24, with UF value 40.804 (Chilean format) = 40,804 CLP/UF.
"""

from datetime import date
import pytest
from fintoolsom.fixedIncome import CLBond
from tests.factories import clbond_factory
from fintoolsom.rates import (
    Rate,
    RateConvention,
    CompoundedInterestConvention,
)
from fintoolsom.dates import ActualDayCountConvention


VALUATION_DATE = date(2026, 6, 24)


UF_VALUE = 40_804.0


IRR_CONVENTION = RateConvention(
    CompoundedInterestConvention, ActualDayCountConvention, 365
)


@pytest.fixture
def btp0470930() -> CLBond:
    return clbond_factory.btp0470930()


@pytest.fixture
def btu0190930() -> CLBond:
    return clbond_factory.btu0190930()


def test_btp0470930_tera(btp0470930):
    assert btp0470930.tera.rate == pytest.approx(0.047503, abs=1e-6)


def test_btu0190930_tera(btu0190930):
    assert btu0190930.tera.rate == pytest.approx(0.019071, abs=1e-6)


def test_btp0470930_price_and_par_value(btp0470930):
    irr = Rate(IRR_CONVENTION, 0.051919)
    price, par_value = btp0470930.get_price(VALUATION_DATE, irr)
    assert price == pytest.approx(98.4054, abs=1e-4)
    assert par_value == pytest.approx(101.4729514, abs=1e-6)


def test_btu0190930_price_and_par_value(btu0190930):
    irr = Rate(IRR_CONVENTION, 0.023138)
    price, par_value = btu0190930.get_price(VALUATION_DATE, irr)
    assert price == pytest.approx(98.4054, abs=1e-4)
    assert par_value == pytest.approx(100.59698426, abs=1e-6)


def test_btp0470930_amount_clp(btp0470930):
    irr = Rate(IRR_CONVENTION, 0.051919)
    amount = btp0470930.get_amount_value(VALUATION_DATE, irr)
    assert amount == pytest.approx(379_448_482, abs=1)


def test_btu0190930_amount_clp(btu0190930):
    """Monto Liq. $ from the screenshot: 121.179.146 CLP, using fx = UF value (40,804 CLP/UF)."""
    irr = Rate(IRR_CONVENTION, 0.023138)
    amount_clp = btu0190930.get_amount_value(VALUATION_DATE, irr, fx=UF_VALUE)
    assert amount_clp == pytest.approx(121_179_146, abs=1)
