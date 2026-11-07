from datetime import date
from fintoolsom.dates import (
    Calendar,
)



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


def test_calendar_custom_holidays_none_means_no_extra_holidays():
    cal = Calendar(custom_holidays=None)
    assert cal.custom_holidays == []
    assert not cal.is_holiday(date(2024, 1, 3))


def test_calendar_adds_custom_holidays_after_construction():
    cal = Calendar()
    cal.add_custom_holidays([date(2024, 1, 3), date(2024, 1, 4)])
    assert cal.is_holiday(date(2024, 1, 3)) and cal.is_holiday(date(2024, 1, 4))


def test_calendars_add_with_plus():
    combined = Calendar(custom_holidays=[date(2024, 1, 3)]) + Calendar(custom_holidays=[date(2024, 1, 4)])
    assert combined.is_holiday(date(2024, 1, 3)) and combined.is_holiday(date(2024, 1, 4))
