from datetime import date

import pytest

from fintoolsom.fixedIncome import CLBond
from fintoolsom.rates import PillarCurve
from tests.factories import clbond_factory, curve_factory


@pytest.fixture
def curve_date() -> date:
    return curve_factory.SAMPLE_CURVE_DATE


@pytest.fixture
def sample_curve() -> PillarCurve:
    return curve_factory.sample_curve()


@pytest.fixture
def sample_bond() -> CLBond:
    """A 1-year CLBond with two semi-annual coupons, built by hand (no fixed_income.db)."""
    return clbond_factory.two_coupon_bond()
