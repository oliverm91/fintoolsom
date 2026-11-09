from datetime import date
import pytest
from fintoolsom.fixedIncome import Coupons
from tests.factories import clbond_factory


@pytest.fixture
def sample_coupons() -> Coupons:
    return clbond_factory.two_coupon_coupons()


def test_get_remaining_flows_excludes_coupons_ended_on_date(sample_coupons):
    remaining = sample_coupons.get_remaining_flows(date(2024, 7, 1))
    assert list(remaining) == [102]


def test_get_current_coupon_at_period_boundary_belongs_to_next_coupon(
    sample_coupons,
):
    current = sample_coupons.get_current_coupon(date(2024, 7, 1))
    assert current.start_date == date(2024, 7, 1)
    assert current.end_date == date(2025, 1, 1)


def test_adjust_to_notional_rescales_amortizations_to_match_notional(sample_coupons):
    new_notional = 50_000_000
    sample_coupons.adjust_to_notional(new_notional)
    total_amortization = sum(c.amortization for c in sample_coupons.coupons)
    assert total_amortization == pytest.approx(new_notional, rel=1e-3)
