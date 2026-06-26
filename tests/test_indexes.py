import math
import pytest
from datetime import date

from fintoolsom.market import (
    Currency,
    InterestIndex,
    OvernightRateIndex,
    OvernightPriceIndex,
    TermRateIndex,
)
from fintoolsom.rates import Rate, RateConvention, LinearInterestConvention
from fintoolsom.dates import ActualDayCountConvention

RATE_5PCT = Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 365), 0.05)
RATE_3PCT = Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 365), 0.03)

# Mon–Fri week
D0 = date(2024, 1, 2)  # Tue
D1 = date(2024, 1, 3)  # Wed
D2 = date(2024, 1, 4)  # Thu
D3 = date(2024, 1, 5)  # Fri


# ---------------------------------------------------------------------------
# OvernightRateIndex
# ---------------------------------------------------------------------------

def test_overnight_rate_index_builds_index_starting_at_100():
    idx = OvernightRateIndex(name="SOFR", currency=Currency.USD, overnight_rates={D0: RATE_5PCT})
    assert idx._index_values[D0] == 100.0


def test_overnight_rate_index_index_extends_one_bd_past_last_rate():
    idx = OvernightRateIndex(name="SOFR", currency=Currency.USD, overnight_rates={D0: RATE_5PCT})
    assert D1 in idx._index_values
    assert D2 not in idx._index_values


def test_overnight_rate_index_gap_filled_by_repeating_last_rate():
    # D1 is missing — should be filled with D0's rate
    idx = OvernightRateIndex(name="SOFR", currency=Currency.USD, overnight_rates={D0: RATE_5PCT, D2: RATE_3PCT})
    assert idx.overnight_rates[D1] == RATE_5PCT


def test_overnight_rate_index_add_rate_rebuilds_index():
    idx = OvernightRateIndex(name="SOFR", currency=Currency.USD, overnight_rates={D0: RATE_5PCT})
    idx.add_rate(D1, RATE_3PCT)
    assert D2 in idx._index_values


def test_overnight_rate_index_accrued_interest_one_day():
    idx = OvernightRateIndex(name="SOFR", currency=Currency.USD, overnight_rates={D0: RATE_5PCT})
    notional = 1_000_000.0
    ai = idx.get_accrued_interest(notional, D0, D1)
    expected = notional * RATE_5PCT.get_wealth_factor(D0, D1) - notional
    assert math.isclose(ai, expected, rel_tol=1e-9)


def test_overnight_rate_index_is_interest_index():
    idx = OvernightRateIndex(name="SOFR", currency=Currency.USD, overnight_rates={D0: RATE_5PCT})
    assert isinstance(idx, InterestIndex)


# ---------------------------------------------------------------------------
# OvernightPriceIndex
# ---------------------------------------------------------------------------

def test_overnight_price_index_raises_on_empty():
    with pytest.raises(ValueError):
        OvernightPriceIndex(name="ICP", currency=Currency.CLP, index_values={})


def test_overnight_price_index_fills_first_interval_gap():
    # Only D0 and D2 known — D1 must be filled by geometric interpolation
    idx = OvernightPriceIndex(name="ICP", currency=Currency.CLP, index_values={D0: 35000.0, D2: 35014.0})
    expected_d1 = 35000.0 * (35014.0 / 35000.0) ** (1 / 2)
    assert D1 in idx.index_values
    assert math.isclose(idx.index_values[D1], expected_d1, rel_tol=1e-9)


def test_overnight_price_index_fills_multi_step_gap():
    # D0 and D3 known — D1 and D2 must be filled by geometric interpolation
    idx = OvernightPriceIndex(name="ICP", currency=Currency.CLP, index_values={D0: 35000.0, D3: 35021.0})
    ratio = 35021.0 / 35000.0
    days_total = (D3 - D0).days
    expected_d1 = 35000.0 * ratio ** ((D1 - D0).days / days_total)
    expected_d2 = 35000.0 * ratio ** ((D2 - D0).days / days_total)
    assert math.isclose(idx.index_values[D1], expected_d1, rel_tol=1e-9)
    assert math.isclose(idx.index_values[D2], expected_d2, rel_tol=1e-9)


def test_overnight_price_index_accrued_interest():
    idx = OvernightPriceIndex(name="ICP", currency=Currency.CLP, index_values={D0: 35000.0, D2: 35014.0})
    notional = 1_000_000.0
    ai = idx.get_accrued_interest(notional, D0, D2)
    expected = notional * (35014.0 / 35000.0 - 1.0)
    assert math.isclose(ai, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# TermRateIndex
# ---------------------------------------------------------------------------

def test_term_rate_index_add_and_get_rate():
    idx = TermRateIndex(name="SOFR3M", currency=Currency.USD)
    idx.add_rate(D0, RATE_5PCT)
    assert idx.get_rate(D0) == RATE_5PCT


def test_term_rate_index_missing_date_raises():
    idx = TermRateIndex(name="SOFR3M", currency=Currency.USD)
    with pytest.raises(KeyError):
        idx.get_rate(D0)


def test_term_rate_index_accrued_interest_uses_fixing_date():
    idx = TermRateIndex(name="SOFR3M", currency=Currency.USD, historic_data={D0: RATE_5PCT})
    notional = 1_000_000.0
    ai = idx.get_accrued_interest(notional, D1, D2, fixing_date=D0)
    expected = RATE_5PCT.get_accrued_interest(notional, D1, D2)
    assert math.isclose(ai, expected, rel_tol=1e-9)
