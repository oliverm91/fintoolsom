from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.dates import (
    ActualDayCountConvention,
    Days30ADayCountConvention,
    Days30EDayCountConvention,
    Days30UDayCountConvention,
    Calendar,
    ScheduleGenerator,
)
from fintoolsom.dates.adjustments import (
    FollowingConvention,
    PrecedingConvention,
    ModifiedFollowingConvention,
    ModifiedPrecedingConvention,
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


def test_calendar_is_holiday_for_custom_holiday_and_weekend():
    cal = Calendar(custom_holidays=[date(2024, 7, 4)])  # Thursday
    assert cal.is_holiday(date(2024, 7, 4)) is True
    assert cal.is_holiday(date(2024, 7, 6)) is True  # Saturday
    assert cal.is_holiday(date(2024, 7, 5)) is False  # Friday, not a holiday


def test_calendar_add_business_days_skips_weekend_and_holiday():
    cal = Calendar(custom_holidays=[date(2024, 7, 4)])  # Thursday
    assert cal.add_business_days(date(2024, 7, 4), 1) == date(2024, 7, 5)
    assert cal.add_business_days(date(2024, 7, 4), -1) == date(2024, 7, 3)


def test_calendar_combine_merges_holidays_from_both_calendars():
    cal_a = Calendar(custom_holidays=[date(2024, 3, 1)])
    cal_b = Calendar(custom_holidays=[date(2024, 4, 1)])
    combined = cal_a.combine(cal_b)
    assert combined.is_holiday(date(2024, 3, 1)) is True
    assert combined.is_holiday(date(2024, 4, 1)) is True
    assert combined.is_holiday(date(2024, 3, 4)) is False  # Monday, not a holiday


def test_following_and_preceding_convention_adjust_on_holiday():
    cal = Calendar(custom_holidays=[date(2024, 7, 4)])  # Thursday
    following = FollowingConvention(cal)
    preceding = PrecedingConvention(cal)

    assert following.adjust(date(2024, 7, 4)) == date(2024, 7, 5)
    assert following.adjust(date(2024, 7, 5)) == date(2024, 7, 5)  # not a holiday
    assert preceding.adjust(date(2024, 7, 4)) == date(2024, 7, 3)
    assert preceding.adjust(date(2024, 7, 5)) == date(2024, 7, 5)  # not a holiday


def test_modified_conventions_correct_to_stay_within_same_month():
    # Last business day of March is a holiday: following would roll into April,
    # so modified following must fall back to preceding instead.
    cal_month_end = Calendar(custom_holidays=[date(2024, 3, 29)])
    modified_following = ModifiedFollowingConvention(cal_month_end)
    assert modified_following.adjust(date(2024, 3, 29)) == date(2024, 3, 28)

    # First business day of April is a holiday: preceding would roll into March,
    # so modified preceding must fall forward instead.
    cal_month_start = Calendar(custom_holidays=[date(2024, 4, 1)])
    modified_preceding = ModifiedPrecedingConvention(cal_month_start)
    assert modified_preceding.adjust(date(2024, 4, 1)) == date(2024, 4, 2)


def test_schedule_generator_semiannual_schedule_over_one_year():
    adj_conv = ModifiedFollowingConvention(Calendar())
    schedule = ScheduleGenerator.generate_schedule(
        date(2024, 1, 1), "1y", "6m", adj_conv
    )
    assert schedule == [date(2024, 1, 1), date(2024, 7, 1), date(2025, 1, 1)]
