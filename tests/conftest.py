from datetime import date

import pytest

from fintoolsom.fixedIncome import CLBond
from fintoolsom.rates import ZeroCouponCurve
from tests.factories import clbond_factory, curve_factory


@pytest.fixture
def curve_date() -> date:
    return curve_factory.SAMPLE_CURVE_DATE


@pytest.fixture
def sample_zero_coupon_curve() -> ZeroCouponCurve:
    return curve_factory.sample_zero_coupon_curve()


@pytest.fixture
def sample_bond() -> CLBond:
    """A 1-year CLBond with two semi-annual coupons, built by hand (no fixed_income.db)."""
    return clbond_factory.two_coupon_bond()
