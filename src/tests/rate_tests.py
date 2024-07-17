from datetime import date, timedelta
from fintoolsom.rates import Rate, RateConvention, InterestConvention
from fintoolsom.dates import DayCountConvention

def run_tests():
    days = 70
    df = 0.99
    t_start = date(2023, 10, 17)
    t_end = t_start + timedelta(days=days)

    base = 360
    rc = RateConvention(InterestConvention.Linear, DayCountConvention.Actual, base)
    test_val = Rate.get_rate_from_df(df, t_start, t_end, rc).rate_value
    expected_val = (1 / df - 1)*(base/days)
    assert test_val == expected_val, f'Rate test (get rate from df) failed. Rate value obtained: {test_val}. Expected {expected_val}'

    rate = 3
    base = 360
    rc = RateConvention(InterestConvention.Linear, DayCountConvention.Actual, base)
    t_start = date(2023,10,17)
    days=25
    t_end = t_start + timedelta(days=days)
    r = Rate(rc, rate/100)
    base_2 = 365
    r.convert_rate_conventions(RateConvention(InterestConvention.Compounded, DayCountConvention.Actual, base_2), t_start, t_end)
    test_val = r.rate_value
    expected_val = (1 + rate/100 * days / 360)**(base_2/days) - 1
    assert test_val == expected_val, f'Rate test (convert rate convention) failed. Rate value obtained: {test_val}. Expected {expected_val}'

    
    print('Rate tests: Ok')