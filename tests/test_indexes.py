import math
import pytest
from datetime import date

from fintoolsom.market import (
    Currency,
    InterestIndex,
    RateIndex,
    PriceIndex,
    InterestPriceIndex,
    UFIndex,
    IndexHistory,
    InterestHistory,
    RateHistory,
    OvernightRateHistory,
    TermRateHistory,
    PriceHistory,
    OvernightInterestPriceHistory,
    UFIndexHistory,
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

# Definitions (identity only — name + currency, value type fixed by class).
SOFR = RateIndex("SOFR", currency=Currency.USD)
SOFR3M = RateIndex("SOFR3M", currency=Currency.USD)
ICP = InterestPriceIndex("ICP", currency=Currency.CLP)
UF = UFIndex("UF", currency=Currency.CLP)


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

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


def test_price_only_definition_is_not_interest_index():
    # UF is price-only: it must NOT type as interest-bearing, so it cannot be
    # used where a FloatingLegSpec / collateral index is required.
    assert not isinstance(UF, InterestIndex)


# ---------------------------------------------------------------------------
# OvernightRateHistory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# OvernightPriceHistory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TermRateHistory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PriceHistory — price-only index (UF): lookup, but no interest accrual
# ---------------------------------------------------------------------------

def test_price_history_get_value():
    h = PriceHistory(index=UF, values={D0: 37000.0})
    assert h.get_value(D0) == 37000.0


def test_price_history_missing_date_raises():
    h = PriceHistory(index=UF, values={D0: 37000.0})
    with pytest.raises(KeyError):
        h.get_value(D1)


def test_price_only_history_has_no_accrual_method():
    # Point: UF and other price-only indexes never accrue interest, so their
    # history must not even expose get_accrued_interest, and must not type as an
    # InterestHistory.
    h = PriceHistory(index=UF, values={D0: 37000.0})
    assert not hasattr(h, "get_accrued_interest")
    assert not isinstance(h, InterestHistory)


def test_overnight_interest_price_history_is_interest_history():
    h = OvernightInterestPriceHistory(index=ICP, values={D0: 35000.0, D2: 35014.0})
    assert isinstance(h, InterestHistory)
    assert isinstance(h, PriceHistory)  # still supports level lookup
    assert h.get_value(D0) == 35000.0


# ---------------------------------------------------------------------------
# UFIndexHistory (UF): monthly inflation from published levels
# ---------------------------------------------------------------------------

def test_inflation_index_is_price_only_definition():
    # UF is inflation-linked but price-only: a PriceIndex that is NOT interest-bearing.
    assert isinstance(UF, PriceIndex)
    assert not isinstance(UF, InterestIndex)


def test_inflation_history_is_price_history_without_accrual():
    h = UFIndexHistory(index=UF, values={date(2026, 2, 9): 37000.0})
    assert isinstance(h, PriceHistory)
    assert not isinstance(h, InterestHistory)
    assert not hasattr(h, "get_accrued_interest")


def test_inflation_from_reajuste_boundaries():
    # Inflation of Jan 2026 = UF(9-Mar-2026) / UF(9-Feb-2026) - 1.
    h = UFIndexHistory(
        index=UF,
        values={date(2026, 2, 9): 37000.0, date(2026, 3, 9): 37185.0},
    )
    inflation = h.get_inflation(2026, 1)
    assert math.isclose(inflation, 37185.0 / 37000.0 - 1.0, rel_tol=1e-12)


def test_inflation_reads_future_boundary_level():
    # The later boundary (9-Mar) can be a future date relative to "today"; as long
    # as it is loaded (published in advance), get_inflation still resolves it.
    h = UFIndexHistory(
        index=UF,
        values={date(2026, 2, 9): 37000.0, date(2026, 3, 9): 37111.0},
    )
    assert h.get_value(date(2026, 3, 9)) == 37111.0  # future print is held
    assert math.isclose(h.get_inflation(2026, 1), 37111.0 / 37000.0 - 1.0, rel_tol=1e-12)


def test_inflation_rolls_year_for_december():
    # Inflation of Dec 2026 spans 9-Jan-2027 → 9-Feb-2027.
    h = UFIndexHistory(
        index=UF,
        values={date(2027, 1, 9): 38000.0, date(2027, 2, 9): 38152.0},
    )
    assert math.isclose(h.get_inflation(2026, 12), 38152.0 / 38000.0 - 1.0, rel_tol=1e-12)


def test_inflation_missing_boundary_raises():
    h = UFIndexHistory(index=UF, values={date(2026, 2, 9): 37000.0})
    with pytest.raises(ValueError):
        h.get_inflation(2026, 1)  # 9-Mar-2026 not loaded yet


def test_last_known_date_before_cpi_release():
    # 8-Jun-2026 is a Tuesday (business day), so the release proxy is the 8th.
    # On the 3rd (before release) only levels up to the 9th of the current month
    # are known.
    h = UFIndexHistory(index=UF, values={date(2026, 2, 9): 37000.0})
    assert h.get_last_known_date(date(2026, 6, 3)) == date(2026, 6, 9)


def test_last_known_date_after_cpi_release():
    # On/after the release, levels are known up to the 9th of next month.
    h = UFIndexHistory(index=UF, values={date(2026, 2, 9): 37000.0})
    assert h.get_last_known_date(date(2026, 6, 25)) == date(2026, 7, 9)


def test_last_known_date_rolls_year_in_december():
    h = UFIndexHistory(index=UF, values={date(2026, 2, 9): 37000.0})
    assert h.get_last_known_date(date(2026, 12, 20)) == date(2027, 1, 9)


def test_last_known_date_uses_chilean_calendar():
    # 8-Dec-2026 (Inmaculada Concepción) is a Chilean holiday, so the CPI-release
    # proxy (preceding business day of the 8th) is 7-Dec-2026 (Monday). On the 7th
    # we are already on/after release → horizon is next month. A weekends-only
    # calendar would place release on the 8th and keep us in the current month, so
    # this asserts the CL calendar (from CLP → Locality.CL) is in effect.
    h = UFIndexHistory(index=UF, values={date(2026, 12, 9): 39000.0})
    assert h.get_last_known_date(date(2026, 12, 7)) == date(2027, 1, 9)


def test_extend_with_cpi_matches_uf_formula():
    h = UFIndexHistory(index=UF, values={date(2026, 3, 9): 37000.0})
    h.extend_with_cpi(0.004)
    # UF is published rounded to 2 decimals, so the formula output is exact.
    # Worked example: UF(13-Mar) = round(UF(9-Mar)*(1.004)^(4/31), 2).
    assert h.get_value(date(2026, 3, 13)) == round(37000.0 * 1.004 ** (4 / 31), 2)
    # The next 9th carries the full month's variation, exactly.
    assert h.get_value(date(2026, 4, 9)) == round(37000.0 * 1.004, 2)
    assert h.get_value(date(2026, 4, 9)) == 37148.0


def test_extend_with_cpi_rounds_to_two_decimals():
    h = UFIndexHistory(index=UF, values={date(2026, 3, 9): 37123.45})
    h.extend_with_cpi(0.0037)
    for d, level in h.values.items():
        assert round(level, 2) == level  # every projected level is a 2-dp value


def test_extend_with_cpi_roundtrips_through_get_inflation():
    # Base is 9-Mar, so the period applies February's CPI; once reached, the
    # inflation of February reads back as that CPI.
    h = UFIndexHistory(index=UF, values={date(2026, 3, 9): 37000.0})
    h.extend_with_cpi(0.004)
    assert math.isclose(h.get_inflation(2026, 2), 0.004, rel_tol=1e-9)


def test_extend_with_cpi_does_not_overwrite_known_levels():
    known = 99999.0
    h = UFIndexHistory(
        index=UF, values={date(2026, 3, 9): 37000.0, date(2026, 3, 13): known}
    )
    h.extend_with_cpi(0.004)
    assert h.get_value(date(2026, 3, 13)) == known


def test_extend_with_cpi_requires_a_ninth_base():
    h = UFIndexHistory(index=UF, values={date(2026, 3, 10): 37000.0})
    with pytest.raises(ValueError):
        h.extend_with_cpi(0.004)


def test_extend_with_cpi_rejects_non_ninth_base():
    h = UFIndexHistory(index=UF, values={date(2026, 3, 9): 37000.0})
    with pytest.raises(ValueError):
        h.extend_with_cpi(0.004, base_date=date(2026, 3, 10))
