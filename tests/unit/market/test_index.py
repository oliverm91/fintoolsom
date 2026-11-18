import pytest

from fintoolsom.dates import ActualDayCountConvention, ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import (
    Currency,
    InterestIndex,
    OvernightRateIndex,
    OvernightInterestPriceIndex,
)
from fintoolsom.market.index import TermRateIndex
from fintoolsom.rates import LinearInterestConvention, RateConvention


SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)


ICP = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)


def test_rate_index_holds_name_and_currency():
    assert SOFR.name == "SOFR"
    assert SOFR.currency == Currency.USD


TERM3M = TermRateIndex(
    "TERM3M",
    currency=Currency.USD,
    term=Term(3, TermUnit.M, ModifiedFollowingConvention("US")),
)
LIN_ACT_360 = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)


@pytest.mark.parametrize("index", [SOFR, ICP, TERM3M], ids=lambda i: i.name)
def test_every_interest_index_has_a_rate_convention_linear_act_360_by_default(index):
    assert index.rate_convention == LIN_ACT_360


def test_each_index_gets_its_own_default_rate_convention():
    other = OvernightRateIndex("ESTR", currency=Currency.EUR)
    assert other.rate_convention == SOFR.rate_convention
    assert other.rate_convention is not SOFR.rate_convention


def test_the_rate_convention_is_keyword_only_and_not_part_of_the_identity():
    act_365 = RateConvention(LinearInterestConvention, ActualDayCountConvention, 365)
    sonia = OvernightRateIndex("SONIA", currency=Currency.GBP, rate_convention=act_365)
    assert sonia.rate_convention is act_365
    with pytest.raises(TypeError):  # name and term are the only positional fields
        TermRateIndex("TERM3M", TERM3M.term, act_365)
    # Two indexes are equal when their names are.
    assert OvernightRateIndex("SOFR", rate_convention=act_365) == SOFR


def test_definitions_carry_no_time_series():
    # The split: definitions hold no fixings, only histories do.
    assert not hasattr(SOFR, "rates")
    assert not hasattr(ICP, "values")


def test_interest_bearing_definitions_are_interest_indexes():
    # RateIndex (SOFR) and InterestPriceIndex (ICP) accrue → InterestIndex.
    assert isinstance(SOFR, InterestIndex)
    assert isinstance(ICP, InterestIndex)
