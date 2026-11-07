from fintoolsom.market import (
    Currency,
    InterestIndex,
    OvernightRateIndex,
    OvernightInterestPriceIndex,
)



SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)


ICP = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)


def test_rate_index_holds_name_and_currency():
    assert SOFR.name == "SOFR"
    assert SOFR.currency == Currency.USD


def test_definitions_carry_no_time_series():
    # The split: definitions hold no fixings, only histories do.
    assert not hasattr(SOFR, "rates")
    assert not hasattr(ICP, "values")


def test_interest_bearing_definitions_are_interest_indexes():
    # RateIndex (SOFR) and InterestPriceIndex (ICP) accrue → InterestIndex.
    assert isinstance(SOFR, InterestIndex)
    assert isinstance(ICP, InterestIndex)
