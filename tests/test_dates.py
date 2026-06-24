from datetime import date, timedelta

from fintoolsom.dates import ActualDayCountConvention, Days30EDayCountConvention


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
