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
    combined = Calendar(custom_holidays=[date(2024, 1, 3)]) + Calendar(
        custom_holidays=[date(2024, 1, 4)]
    )
    assert combined.is_holiday(date(2024, 1, 3)) and combined.is_holiday(
        date(2024, 1, 4)
    )


# ── Shared per-country calendars ────────────────────────────────────────────


def test_get_calendar_returns_one_shared_instance_per_country():
    from fintoolsom.dates import get_calendar

    assert get_calendar("CL") is get_calendar("CL")
    assert get_calendar("CL") is not get_calendar("US")
    assert get_calendar() is get_calendar(None)
    assert get_calendar("CL").is_holiday(date(2025, 9, 18))  # Fiestas Patrias


def test_adjustment_conventions_resolve_a_country_code_to_the_shared_calendar():
    from fintoolsom.dates import (
        FollowingConvention,
        ModifiedFollowingConvention,
        ModifiedPrecedingConvention,
        PrecedingConvention,
        get_calendar,
    )

    for convention in (
        FollowingConvention,
        PrecedingConvention,
        ModifiedFollowingConvention,
        ModifiedPrecedingConvention,
    ):
        assert convention("CL").calendar is get_calendar("CL")
    assert ModifiedFollowingConvention("CL")._fc.calendar is get_calendar("CL")
    assert ModifiedFollowingConvention().calendar is get_calendar()
    own = Calendar(country="CL")
    assert ModifiedFollowingConvention(own).calendar is own


def test_a_holiday_added_to_a_shared_calendar_reaches_every_consumer():
    from fintoolsom.dates import ModifiedFollowingConvention, get_calendar

    convention = ModifiedFollowingConvention("CL")
    extra = date(2031, 3, 4)  # an ordinary Tuesday
    assert convention.adjust(extra) == extra
    assert not Calendar(country="CL").is_holiday(extra)
    get_calendar("CL").add_custom_holiday(extra)
    try:
        assert convention.adjust(extra) == date(2031, 3, 5)
        assert not Calendar(country="CL").is_holiday(
            extra
        )  # an own calendar is independent
    finally:
        get_calendar("CL")._calendar.pop(extra)


def test_repr_is_short():
    from datetime import date
    from fintoolsom.dates import Calendar

    assert repr(Calendar()) == "Calendar(weekends only)"
    cal = Calendar(country="CL", subdiv=None, custom_holidays=[date(2025, 1, 2)])
    assert repr(cal) == "Calendar(CL, 1 custom)"
    assert str(cal) == repr(cal)
