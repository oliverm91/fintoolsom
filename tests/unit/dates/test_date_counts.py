from datetime import date, timedelta
import numpy as np
import pytest
from fintoolsom.dates import (
    ActualDayCountConvention,
    Days30ADayCountConvention,
    Days30EDayCountConvention,
    Days30UDayCountConvention,
)



def test_actual_day_count_single_dates():
    t1 = date(2023, 10, 17)
    t2 = t1 + timedelta(days=7)
    assert ActualDayCountConvention.get_day_count(t1, t2) == 7


def test_actual_day_count_list_of_dates():
    t1 = date(2023, 10, 17)
    days_between = [7, 10]
    t2_list = [t1 + timedelta(days=d) for d in days_between]
    result = list(ActualDayCountConvention.get_day_count(t1, t2_list))
    assert result == days_between


def test_days_30e_time_fraction():
    t1 = date(2023, 1, 31)
    t2 = date(2023, 3, 31)
    # 30E: both day-of-month values are capped at 30, so Jan 31 -> Mar 31 is exactly 2 months.
    time_fraction = Days30EDayCountConvention.get_time_fraction(t1, t2, 360)
    assert time_fraction == 60 / 360


def test_days_30u_day_count_matches_30a_and_30e_when_both_days_are_month_end():
    # 30/360 US (Bond Basis): D1=31 must be set to 30 *before* checking whether
    # D2=31 and D1==30, so this should be exactly 2 months (60 days), same as
    # the 30A and 30E conventions for this exact case.
    t1 = date(2024, 1, 31)
    t2 = date(2024, 3, 31)
    assert Days30UDayCountConvention.get_day_count(t1, t2) == 60
    assert Days30UDayCountConvention.get_day_count(t1, t2) == (
        Days30ADayCountConvention.get_day_count(t1, t2)
    )


def test_get_time_fraction_scalar_and_list_have_consistent_types_and_values():
    t1 = date(2023, 10, 17)
    days_between = [7, 10]
    t2_list = [t1 + timedelta(days=d) for d in days_between]

    scalar_fraction = ActualDayCountConvention.get_time_fraction(t1, t2_list[0], 360)
    assert isinstance(scalar_fraction, float)
    assert scalar_fraction == pytest.approx(7 / 360)

    list_fraction = ActualDayCountConvention.get_time_fraction(t1, t2_list, 360)
    assert isinstance(list_fraction, np.ndarray)
    assert list_fraction == pytest.approx(np.array(days_between) / 360)


# ── Reference tables ────────────────────────────────────────────────────────
# Expected day counts copied (once, by hand) from OpenGamma Strata's
# DayCountTest.java; no network access is needed here.
from fintoolsom.dates import Days30EISDADayCountConvention  # noqa: E402
from tests.factories.day_counts import D, calc360  # noqa: E402

# Strata data_30U360, EOM column (30U/360 with the end-of-month rule).
THIRTY_U_360_EOM = [
    (D(2011, 12, 28), D(2012, 2, 28), calc360(2011, 12, 28, 2012, 2, 28)),
    (D(2011, 12, 28), D(2012, 2, 29), calc360(2011, 12, 28, 2012, 2, 29)),
    (D(2011, 12, 28), D(2012, 3, 1), calc360(2011, 12, 28, 2012, 3, 1)),
    (D(2011, 12, 28), D(2016, 2, 29), calc360(2011, 12, 28, 2016, 2, 29)),
    (D(2012, 2, 28), D(2012, 3, 28), calc360(2012, 2, 28, 2012, 3, 28)),
    (D(2012, 2, 29), D(2012, 3, 28), calc360(2012, 2, 30, 2012, 3, 28)),
    (D(2012, 2, 29), D(2012, 3, 30), calc360(2012, 2, 30, 2012, 3, 30)),
    (D(2012, 2, 29), D(2012, 3, 31), calc360(2012, 2, 30, 2012, 3, 30)),
    (D(2012, 2, 29), D(2013, 2, 28), calc360(2012, 2, 30, 2013, 2, 30)),
    (D(2011, 2, 28), D(2012, 2, 28), calc360(2011, 2, 30, 2012, 2, 28)),
    (D(2011, 2, 28), D(2012, 2, 29), calc360(2011, 2, 30, 2012, 2, 30)),
    (D(2012, 2, 29), D(2016, 2, 29), calc360(2012, 2, 30, 2016, 2, 30)),
    (D(2012, 3, 1), D(2012, 3, 28), calc360(2012, 3, 1, 2012, 3, 28)),
    (D(2012, 5, 30), D(2013, 8, 29), calc360(2012, 5, 30, 2013, 8, 29)),
    (D(2012, 5, 29), D(2013, 8, 30), calc360(2012, 5, 29, 2013, 8, 30)),
    (D(2012, 5, 29), D(2013, 8, 31), calc360(2012, 5, 29, 2013, 8, 31)),
    (D(2012, 5, 30), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 30), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
]

# Strata THIRTY_360_ISDA rows (30A/360, 2006 ISDA 4.16(f)).
THIRTY_A_360 = [
    (D(2011, 12, 28), D(2012, 2, 29), calc360(2011, 12, 28, 2012, 2, 29)),
    (D(2012, 2, 29), D(2012, 3, 28), calc360(2012, 2, 29, 2012, 3, 28)),
    (D(2012, 5, 29), D(2013, 8, 31), calc360(2012, 5, 29, 2013, 8, 31)),
    (D(2012, 5, 30), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 30), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
]

# Strata THIRTY_E_360 rows (2006 ISDA 4.16(g)).
THIRTY_E_360 = [
    (D(2011, 2, 28), D(2012, 2, 29), calc360(2011, 2, 28, 2012, 2, 29)),
    (D(2012, 5, 29), D(2013, 8, 31), calc360(2012, 5, 29, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 30), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
]

# Strata data_30E360ISDA, "maturity" column: an end date on the last of February
# is the termination date and is kept (D1 on the last of February still moves to 30).
THIRTY_E_360_ISDA_MATURITY = [
    (D(2011, 12, 28), D(2012, 2, 29), calc360(2011, 12, 28, 2012, 2, 29)),
    (D(2012, 2, 29), D(2012, 3, 28), calc360(2012, 2, 30, 2012, 3, 28)),
    (D(2011, 2, 28), D(2012, 2, 28), calc360(2011, 2, 30, 2012, 2, 28)),
    (D(2011, 2, 28), D(2012, 2, 29), calc360(2011, 2, 30, 2012, 2, 29)),
    (D(2012, 2, 29), D(2016, 2, 29), calc360(2012, 2, 30, 2016, 2, 29)),
    (D(2012, 5, 29), D(2013, 8, 31), calc360(2012, 5, 29, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
]


@pytest.mark.parametrize("start, end, days", THIRTY_U_360_EOM)
def test_30u_eom_matches_reference(start, end, days):
    assert Days30UDayCountConvention.get_day_count(start, end) == days


@pytest.mark.parametrize("start, end, days", THIRTY_A_360)
def test_30a_matches_reference(start, end, days):
    assert Days30ADayCountConvention.get_day_count(start, end) == days


@pytest.mark.parametrize("start, end, days", THIRTY_E_360)
def test_30e_matches_reference(start, end, days):
    assert Days30EDayCountConvention.get_day_count(start, end) == days


@pytest.mark.parametrize("start, end, days", THIRTY_E_360_ISDA_MATURITY)
def test_30e_isda_matches_reference_maturity_column(start, end, days):
    assert Days30EISDADayCountConvention.get_day_count(start, end) == days


# ── Input shapes ────────────────────────────────────────────────────────────

def test_day_count_accepts_every_scalar_list_combination():
    s, e = [date(2024, 1, 1), date(2024, 2, 1)], [date(2024, 3, 1), date(2024, 4, 1)]
    assert list(ActualDayCountConvention.get_day_count(s, e[0])) == [60, 29]
    assert list(ActualDayCountConvention.get_day_count(s, e)) == [60, 60]


def test_day_count_rejects_lists_of_different_lengths():
    with pytest.raises(ValueError):
        ActualDayCountConvention.get_day_count([date(2024, 1, 1)], [date(2024, 2, 1), date(2024, 3, 1)])


def test_day_count_rejects_unmapped_input_types():
    with pytest.raises(TypeError):
        ActualDayCountConvention.get_day_count((date(2024, 1, 1),), date(2024, 2, 1))


@pytest.mark.parametrize(
    "convention, name",
    [
        (ActualDayCountConvention, "act"),
        (Days30ADayCountConvention, "30a"),
        (Days30UDayCountConvention, "30u"),
        (Days30EDayCountConvention, "30e"),
        (Days30EISDADayCountConvention, "30eISDA"),
    ],
)
def test_conventions_have_short_names(convention, name):
    assert convention.__str__() == name
