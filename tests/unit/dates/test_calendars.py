from datetime import date
from fintoolsom.dates import (
    Calendar,
    get_calendar,
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


def test_added_custom_holidays_are_recorded_once_and_survive_combine():
    cal = Calendar()
    cal.add_custom_holiday(date(2024, 1, 3))
    cal.add_custom_holiday(date(2024, 1, 3))
    assert cal.custom_holidays == [date(2024, 1, 3)]
    assert repr(cal) == "Calendar(weekends only, 1 custom)"
    assert (cal + Calendar()).custom_holidays == [date(2024, 1, 3)]


def test_calendars_add_with_plus():
    combined = Calendar(custom_holidays=[date(2024, 1, 3)]) + Calendar(
        custom_holidays=[date(2024, 1, 4)]
    )
    assert combined.is_holiday(date(2024, 1, 3)) and combined.is_holiday(
        date(2024, 1, 4)
    )


# ── Combining country calendars ─────────────────────────────────────────────

US_INDEPENDENCE = date(2024, 7, 4)  # Thursday
CL_FIESTAS = [date(2024, 9, 18), date(2024, 9, 19), date(2024, 9, 20)]  # Wed-Fri
CL_BANK_HOLIDAY = date(2024, 12, 31)  # Tuesday: the "bank" category of CL


def test_combine_merges_country_calendars_either_way():
    us, cl = Calendar(country="US"), Calendar(country="CL")
    for combined in (us.combine(cl), cl.combine(us), us + cl):
        for d in (US_INDEPENDENCE, *CL_FIESTAS, CL_BANK_HOLIDAY):
            assert combined.is_holiday(d), d
        assert not combined.is_holiday(date(2024, 3, 4))  # an ordinary Monday
        assert combined.is_holiday(date(2024, 3, 2))  # Saturday


def test_combine_covers_years_neither_calendar_has_loaded_yet():
    combined = Calendar(country="US") + Calendar(country="CL")
    assert combined.is_holiday(date(2040, 7, 4))
    assert combined.is_holiday(date(2040, 9, 18))
    assert combined.is_holiday(date(2040, 12, 31))  # CL bank holiday


def test_combine_keeps_a_subdivision():
    combined = Calendar(country="US", subdiv="NY") + Calendar(country="CL")
    assert combined.is_holiday(date(2024, 2, 12))  # Lincoln's Birthday, NY only
    assert not (Calendar(country="US") + Calendar(country="CL")).is_holiday(
        date(2024, 2, 12)
    )


def test_business_days_on_a_combined_calendar_skip_both_calendars_holidays():
    us, cl = Calendar(country="US"), Calendar(country="CL")
    combined = us + cl
    assert combined.add_business_days(date(2024, 7, 3), 1) == date(2024, 7, 5)
    assert combined.add_business_days(date(2024, 7, 5), -1) == date(2024, 7, 3)
    # Fiestas Patrias then the weekend: Tuesday 17 -> Monday 23.
    assert combined.add_business_days(date(2024, 9, 17), 1) == date(2024, 9, 23)
    assert us.add_business_days(date(2024, 9, 17), 1) == date(2024, 9, 18)
    assert cl.add_business_days(date(2024, 7, 3), 1) == date(2024, 7, 4)


def test_combine_chains_and_takes_a_weekends_only_calendar():
    custom = date(2024, 3, 8)  # Friday
    combined = (
        Calendar(country="US")
        + Calendar(country="CL")
        + Calendar(custom_holidays=[custom])
    )
    assert all(combined.is_holiday(d) for d in (US_INDEPENDENCE, CL_FIESTAS[0], custom))
    weekends_only = Calendar() + Calendar(country="US")
    assert weekends_only.is_holiday(US_INDEPENDENCE)
    assert not (Calendar() + Calendar()).is_holiday(date(2024, 3, 4))


def test_combined_shared_calendars_load_any_year_in_any_order():
    # The parts loading years on their own used to make the combined calendar raise
    # RuntimeError (dictionary changed size during iteration) on its next new year.
    from fintoolsom.dates import get_calendar

    cl, us = get_calendar("CL"), get_calendar("US")
    joint = cl + us
    cl.is_holiday(date(2045, 6, 15))
    us.is_holiday(date(2025, 6, 15))
    assert not joint.is_holiday(date(2024, 6, 14))  # an ordinary Friday
    assert joint.is_holiday(date(2024, 7, 4)) and joint.is_holiday(date(2024, 9, 18))


def test_a_holiday_added_to_a_part_reaches_the_combined_calendar():
    us, cl = Calendar(country="US"), Calendar(country="CL")
    combined = us + cl
    extra = date(2024, 3, 5)  # an ordinary Tuesday
    assert combined.add_business_days(date(2024, 3, 4), 1) == extra
    cl.add_custom_holiday(extra)
    assert combined.is_holiday(extra)
    assert combined.add_business_days(date(2024, 3, 4), 1) == date(2024, 3, 6)
    cl.remove_custom_holiday(extra)
    assert not combined.is_holiday(extra) and not cl.is_holiday(extra)


def test_remove_custom_holiday_keeps_the_country_holidays():
    cl = Calendar(country="CL")
    cl.add_custom_holiday(date(2024, 9, 18))  # also Fiestas Patrias
    cl.remove_custom_holiday(date(2024, 9, 18))
    assert cl.custom_holidays == [] and cl.is_holiday(date(2024, 9, 18))
    cl.remove_custom_holiday(date(2024, 3, 5))  # not a custom holiday: no-op
    assert not cl.is_holiday(date(2024, 3, 5))


def test_add_business_days_crosses_years_and_counts_from_a_holiday():
    cl = Calendar(country="CL")
    # 31 Dec (bank holiday) and 1 Jan closed: Monday 30 Dec 2024 -> Thursday 2 Jan 2025.
    assert cl.add_business_days(date(2024, 12, 30), 1) == date(2025, 1, 2)
    assert cl.add_business_days(date(2025, 1, 2), -1) == date(2024, 12, 30)
    assert cl.add_business_days(date(2024, 12, 31), 0) == date(2025, 1, 2)
    assert cl.add_business_days(date(2024, 12, 30), 0) == date(2024, 12, 30)
    start = date(2024, 6, 3)
    assert cl.add_business_days(cl.add_business_days(start, 600), -600) == start


def test_combine_leaves_its_calendars_unchanged():
    us, cl = Calendar(country="US"), Calendar(country="CL")
    combined = us + cl
    assert not us.is_holiday(CL_FIESTAS[0]) and not cl.is_holiday(US_INDEPENDENCE)
    extra = date(2024, 3, 5)  # an ordinary Tuesday
    combined.add_custom_holiday(extra)
    assert combined.is_holiday(extra)
    assert not us.is_holiday(extra) and not cl.is_holiday(extra)


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
        get_calendar("CL").remove_custom_holiday(extra)
    assert convention.adjust(extra) == extra


def test_repr_is_short():
    from datetime import date
    from fintoolsom.dates import Calendar

    assert repr(Calendar()) == "Calendar(weekends only)"
    cal = Calendar(country="CL", subdiv=None, custom_holidays=[date(2025, 1, 2)])
    assert repr(cal) == "Calendar(CL, 1 custom)"
    assert str(cal) == repr(cal)
    assert repr(Calendar(country="US", subdiv="NY")) == "Calendar(US-NY)"


def test_a_combined_calendar_repr_names_every_calendar_in_order():
    cl, us = Calendar(country="CL"), Calendar(country="US")
    assert repr(cl + us) == "Calendar(CL + US)"
    assert repr(us + cl) == "Calendar(US + CL)"
    ny = Calendar(country="US", subdiv="NY")
    assert repr(cl + ny + Calendar(country="JP")) == "Calendar(CL + US-NY + JP)"
    assert repr(cl + us + cl) == "Calendar(CL + US)"  # each once
    assert repr(cl + Calendar()) == "Calendar(CL)"  # weekends add no label
    assert repr(Calendar() + Calendar()) == "Calendar(weekends only)"
    custom = Calendar(custom_holidays=[date(2025, 1, 2), date(2025, 1, 3)])
    assert repr(cl + custom) == "Calendar(CL, 2 custom)"
    assert (cl + us)._countries == ["CL", "US"]
    assert cl._countries == ["CL"] and us._countries == ["US"]  # inputs unchanged


# ── EUR: the TARGET2 calendar (V5) ──────────────────────────────────────────


def test_eur_indices_use_the_target2_calendar():
    from fintoolsom.market import Currency, OvernightRateIndex

    estr = OvernightRateIndex("ESTR", currency=Currency.EUR)
    assert estr.calendar is get_calendar("XECB")
    # German public holidays, TARGET2 business days.
    for d in (date(2025, 5, 29), date(2025, 6, 9), date(2025, 10, 3)):
        assert not estr.calendar.is_holiday(d), d
    # TARGET2 closing days.
    for d in (
        date(2025, 4, 18),
        date(2025, 4, 21),
        date(2025, 5, 1),
        date(2025, 12, 26),
    ):
        assert estr.calendar.is_holiday(d), d
    assert estr.calendar.add_business_days(date(2025, 4, 17), 2) == date(2025, 4, 23)


def test_a_financial_market_calendar_combines_with_a_country_calendar():
    joint = get_calendar("CL") + get_calendar("XECB")
    assert repr(joint) == "Calendar(CL + XECB)"
    assert joint.is_holiday(date(2025, 9, 18))  # CL only
    assert joint.is_holiday(date(2025, 12, 26))  # TARGET2 only
    assert not joint.is_holiday(date(2025, 10, 3))
