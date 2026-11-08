import math
import pytest
from datetime import date
from fintoolsom.market import (
    Currency,
    OvernightRateIndex,
    TermRateIndex,
    OvernightInterestPriceIndex,
    Term,
    TermUnit,
    IndexHistory,
    InterestHistory,
    RateHistory,
    OvernightRateHistory,
    TermRateHistory,
    PriceHistory,
    OvernightInterestPriceHistory,
)
from fintoolsom.rates import CompoundedInterestConvention, Rate, RateConvention, LinearInterestConvention
from fintoolsom.dates import ActualDayCountConvention, Calendar, ModifiedFollowingConvention



RATE_5PCT = Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 365), 0.05)


RATE_3PCT = Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 365), 0.03)


# Mon–Fri week
D0 = date(2024, 1, 2)  # Tue


D1 = date(2024, 1, 3)  # Wed


D2 = date(2024, 1, 4)  # Thu


D3 = date(2024, 1, 5)  # Fri


# Definitions (identity only — name + currency, value type fixed by class).
_ADJ = ModifiedFollowingConvention(Calendar())


SOFR = OvernightRateIndex("SOFR", currency=Currency.USD)


SOFR3M = TermRateIndex("SOFR3M", currency=Currency.USD, term=Term(3, TermUnit.M, _ADJ))


ICP = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)


def test_overnight_rate_history_builds_index_starting_at_100():
    h = OvernightRateHistory(index=SOFR, rates={D0: RATE_5PCT})
    assert h._index_values[D0] == 100.0


def test_overnight_rate_history_extends_one_bd_past_last_rate():
    h = OvernightRateHistory(index=SOFR, rates={D0: RATE_5PCT})
    assert D1 in h._index_values
    assert D2 not in h._index_values


def test_overnight_rate_history_gap_filled_by_repeating_last_rate():
    # D1 is missing — should be filled with D0's rate
    h = OvernightRateHistory(index=SOFR, rates={D0: RATE_5PCT, D2: RATE_3PCT})
    assert h.rates[D1] == RATE_5PCT


def test_overnight_rate_history_add_rate_rebuilds_index():
    h = OvernightRateHistory(index=SOFR, rates={D0: RATE_5PCT})
    h.add_rate(D1, RATE_3PCT)
    assert D2 in h._index_values


def test_overnight_rate_history_accrued_interest_one_day():
    h = OvernightRateHistory(index=SOFR, rates={D0: RATE_5PCT})
    notional = 1_000_000.0
    ai = h.get_accrued_interest(notional, D0, D1)
    expected = notional * RATE_5PCT.get_wealth_factor(D0, D1) - notional
    assert math.isclose(ai, expected, rel_tol=1e-9)


def test_overnight_rate_history_is_index_history():
    h = OvernightRateHistory(index=SOFR, rates={D0: RATE_5PCT})
    assert isinstance(h, RateHistory)
    assert isinstance(h, InterestHistory)
    assert isinstance(h, IndexHistory)
    assert h.name == "SOFR"
    assert h.currency == Currency.USD


def test_overnight_rate_history_spot_overnight_rate_is_todays_fixing():
    # A rate index anchors from today's published fixing (applies to [t, t+1bd]).
    h = OvernightRateHistory(index=SOFR, rates={D0: RATE_5PCT})
    assert h.spot_overnight_rate(D0) == RATE_5PCT


def test_overnight_price_history_raises_on_empty():
    with pytest.raises(ValueError):
        OvernightInterestPriceHistory(index=ICP, values={})


def test_overnight_price_history_fills_first_interval_gap():
    # Only D0 and D2 known — D1 must be filled by geometric interpolation
    h = OvernightInterestPriceHistory(index=ICP, values={D0: 35000.0, D2: 35014.0})
    expected_d1 = 35000.0 * (35014.0 / 35000.0) ** (1 / 2)
    assert D1 in h.values
    assert math.isclose(h.values[D1], expected_d1, rel_tol=1e-9)


def test_overnight_price_history_fills_multi_step_gap():
    # D0 and D3 known — D1 and D2 must be filled by geometric interpolation
    h = OvernightInterestPriceHistory(index=ICP, values={D0: 35000.0, D3: 35021.0})
    ratio = 35021.0 / 35000.0
    days_total = (D3 - D0).days
    expected_d1 = 35000.0 * ratio ** ((D1 - D0).days / days_total)
    expected_d2 = 35000.0 * ratio ** ((D2 - D0).days / days_total)
    assert math.isclose(h.values[D1], expected_d1, rel_tol=1e-9)
    assert math.isclose(h.values[D2], expected_d2, rel_tol=1e-9)


def test_overnight_price_history_accrued_interest():
    h = OvernightInterestPriceHistory(index=ICP, values={D0: 35000.0, D2: 35014.0})
    notional = 1_000_000.0
    ai = h.get_accrued_interest(notional, D0, D2)
    expected = notional * (35014.0 / 35000.0 - 1.0)
    assert math.isclose(ai, expected, rel_tol=1e-9)


def test_overnight_price_history_spot_overnight_rate_uses_last_realized_step():
    # A price index has no forward level, so it anchors from the most recent realised
    # one-business-day level ratio (D1→D2 here; D1 was gap-filled), in the index's
    # convention: ICP's default, linear act/360.
    h = OvernightInterestPriceHistory(index=ICP, values={D0: 35000.0, D2: 35014.0})
    r = h.spot_overnight_rate(D2)
    wf = h.values[D2] / h.values[D1]
    expected = (wf - 1) / ((D2 - D1).days / 360)
    assert r.rate == pytest.approx(expected)
    assert r.rate_convention.time_fraction_base == 360


def test_overnight_price_history_spot_overnight_rate_uses_the_index_convention():
    act365 = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)
    index = OvernightInterestPriceIndex("IDX", currency=Currency.CLP, rate_convention=act365)
    h = OvernightInterestPriceHistory(index=index, values={D0: 35000.0, D1: 35004.0})
    r = h.spot_overnight_rate(D1)
    assert r.rate == pytest.approx((35004.0 / 35000.0) ** (365 / (D1 - D0).days) - 1)
    assert r.rate_convention.interest_convention is CompoundedInterestConvention
    assert r.rate_convention is not index.rate_convention  # a copy: the index is not exposed
    # The implied rate reprices the realised growth.
    assert float(r.get_wealth_factor(D0, D1)) == pytest.approx(35004.0 / 35000.0, rel=1e-14)


def test_price_index_defaults_to_linear_act_360():
    rc = ICP.rate_convention
    assert (rc.interest_convention, rc.day_count_convention, rc.time_fraction_base) == (
        LinearInterestConvention, ActualDayCountConvention, 360
    )
    assert OvernightInterestPriceIndex("X").rate_convention is not rc  # no shared default


def test_overnight_price_history_spot_overnight_rate_needs_two_levels():
    h = OvernightInterestPriceHistory(index=ICP, values={D0: 35000.0})
    with pytest.raises(KeyError):
        h.spot_overnight_rate(D0)


def test_interest_history_abc_forces_accrual():
    # RateHistory leaves get_accrued_interest abstract, so it cannot be
    # instantiated, and neither can a subclass that forgets to implement it.
    with pytest.raises(TypeError):
        RateHistory(index=SOFR3M)

    class ForgetsAccrual(RateHistory):
        pass

    with pytest.raises(TypeError):
        ForgetsAccrual(index=SOFR3M)


def test_term_rate_history_add_and_get_rate():
    h = TermRateHistory(index=SOFR3M)
    h.add_rate(D0, RATE_5PCT)
    assert h.get_rate(D0) == RATE_5PCT


def test_term_rate_history_missing_date_raises():
    h = TermRateHistory(index=SOFR3M)
    with pytest.raises(KeyError):
        h.get_rate(D0)


def test_term_rate_history_accrued_interest_uses_fixing_date():
    h = TermRateHistory(index=SOFR3M, rates={D0: RATE_5PCT})
    notional = 1_000_000.0
    ai = h.get_accrued_interest(notional, D1, D2, fixing_date=D0)
    expected = RATE_5PCT.get_accrued_interest(notional, D1, D2)
    assert math.isclose(ai, expected, rel_tol=1e-9)


def test_price_history_get_value():
    h = PriceHistory(index=ICP, values={D0: 37000.0})
    assert h.get_value(D0) == 37000.0


def test_price_history_missing_date_raises():
    h = PriceHistory(index=ICP, values={D0: 37000.0})
    with pytest.raises(KeyError):
        h.get_value(D1)


def test_price_only_history_has_no_accrual_method():
    # A plain level history must not expose get_accrued_interest nor type as an
    # InterestHistory: accrual comes only from InterestPriceHistory.
    h = PriceHistory(index=ICP, values={D0: 37000.0})
    assert not hasattr(h, "get_accrued_interest")
    assert not isinstance(h, InterestHistory)


def test_overnight_interest_price_history_is_interest_history():
    h = OvernightInterestPriceHistory(index=ICP, values={D0: 35000.0, D2: 35014.0})
    assert isinstance(h, InterestHistory)
    assert isinstance(h, PriceHistory)  # still supports level lookup
    assert h.get_value(D0) == 35000.0
