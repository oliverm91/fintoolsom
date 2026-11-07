"""Holiday rules, checked against published holiday dates."""
from copy import copy
from datetime import date

import pytest

from fintoolsom.dates.holidays import (
    ConsecutiveHolidaySandwichRule,
    FridayEasterRule,
    LastWeekWeekdayRule,
    MondayEasterRule,
    MonthDayRule,
    OrdinalWeekWeekdayRule,
    _easter_sunday,
)

MON, THU = 0, 3


@pytest.mark.parametrize(
    "year, easter",
    [(2000, date(2000, 4, 23)), (2019, date(2019, 4, 21)), (2024, date(2024, 3, 31)), (2025, date(2025, 4, 20))],
)
def test_easter_sunday(year, easter):
    assert _easter_sunday(year) == easter


def test_good_friday_and_easter_monday():
    assert FridayEasterRule().get_date(2024) == date(2024, 3, 29)
    assert MondayEasterRule().get_date(2024) == date(2024, 4, 1)


def test_ordinal_weekday_rule_us_thanksgiving_is_fourth_thursday_of_november():
    rule = OrdinalWeekWeekdayRule(ordinal=4, weekday=THU, month=11)
    assert rule.get_date(2024) == date(2024, 11, 28)
    assert rule.get_date(2023) == date(2023, 11, 23)


def test_last_weekday_rule_us_memorial_day_is_last_monday_of_may():
    rule = LastWeekWeekdayRule(weekday=MON, month=5)
    assert rule.get_date(2024) == date(2024, 5, 27)
    assert rule.get_date(2021) == date(2021, 5, 31)  # the last day itself


def test_month_day_rule_is_fixed_by_default():
    assert MonthDayRule(9, 18).get_date(2024) == date(2024, 9, 18)


@pytest.mark.parametrize(
    "year, observed",
    [
        (2023, date(2023, 6, 26)),  # Thu 29 Jun -> Monday of that week
        (2022, date(2022, 6, 27)),  # Wed 29 Jun -> Monday of that week
        (2021, date(2021, 6, 28)),  # Tue 29 Jun -> Monday of that week
        (2018, date(2018, 7, 2)),  # Fri 29 Jun -> following Monday
        (2024, date(2024, 6, 29)),  # Saturday: not moved
        (2025, date(2025, 6, 29)),  # Sunday: not moved
        (2020, date(2020, 6, 29)),  # Monday: not moved
    ],
)
def test_month_day_rule_moves_to_monday_like_chilean_san_pedro_y_san_pablo(year, observed):
    # Chilean Law 19.668; observed dates as published in the Chilean holiday calendar.
    assert MonthDayRule(6, 29, monday_adjustable=True).get_date(year) == observed


def test_sandwich_rule_bridges_chilean_independence_days():
    rule = ConsecutiveHolidaySandwichRule((MonthDayRule(9, 18), MonthDayRule(9, 19)))
    assert rule.get_date(2018) == date(2018, 9, 17)  # 18 Sep on Tuesday -> Monday 17
    assert rule.get_date(2019) == date(2019, 9, 20)  # 19 Sep on Thursday -> Friday 20
    assert rule.get_date(2024) == date(2024, 9, 20)  # 19 Sep on Thursday -> Friday 20
    assert rule.get_date(2025) is None  # Thursday 18 and Friday 19: no bridge needed


def test_sandwich_rule_needs_two_rules():
    with pytest.raises(ValueError):
        ConsecutiveHolidaySandwichRule((MonthDayRule(9, 18),))


@pytest.mark.parametrize(
    "rule",
    [
        OrdinalWeekWeekdayRule(4, THU, 11),
        LastWeekWeekdayRule(MON, 5),
        MonthDayRule(6, 29, monday_adjustable=True),
        ConsecutiveHolidaySandwichRule((MonthDayRule(9, 18), MonthDayRule(9, 19))),
        MondayEasterRule(),
        FridayEasterRule(),
    ],
)
def test_copies_are_new_objects_giving_the_same_dates(rule):
    duplicate = copy(rule)
    assert duplicate is not rule and type(duplicate) is type(rule)
    assert [duplicate.get_date(y) for y in range(2018, 2026)] == [rule.get_date(y) for y in range(2018, 2026)]
