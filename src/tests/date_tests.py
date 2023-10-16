from fintoolsom.dates import get_day_count, DayCountConvention
from datetime import date, timedelta

def run_tests():
    t1 = date.today()
    days_between = 7
    t2 = t1 + timedelta(days=days_between)
    test_value = get_day_count(t1, t2, DayCountConvention.Actual)
    assert test_value == days_between, f"Days between {t1} and {t2} do not match the expected value: {days_between}. Obtained value: {test_value}"
    print('Date test: Ok')