from datetime import date
from fintoolsom.dates import (
    Calendar,
)
from fintoolsom.dates.adjustments import (
    FollowingConvention,
    PrecedingConvention,
    ModifiedFollowingConvention,
    ModifiedPrecedingConvention,
)



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


def test_modified_preceding_rolls_forward_when_preceding_leaves_the_month():
    from fintoolsom.dates.adjustments import ModifiedPrecedingConvention

    # Saturday 1 June 2024: preceding would be Friday 31 May, so roll to Monday 3 June.
    assert ModifiedPrecedingConvention(Calendar()).adjust(date(2024, 6, 1)) == date(2024, 6, 3)
