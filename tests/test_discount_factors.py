import pytest

from fintoolsom.rates import LinearInterestConvention, CompoundedInterestConvention, ExponentialInterestConvention


# --- Discount factor from rate ---

def test_get_df_from_rate_linear_positive_rate():
    rate, t = 0.05, 0.5
    df = LinearInterestConvention.get_df_from_rate(rate, t)
    assert df == pytest.approx(1 / (1 + rate * t))


def test_get_df_from_rate_compounded_positive_rate():
    rate, t = 0.04, 2
    df = CompoundedInterestConvention.get_df_from_rate(rate, t)
    assert df == pytest.approx(1 / (1 + rate) ** t)


def test_get_df_from_rate_exponential_positive_rate():
    import math
    rate, t = 0.03, 1.5
    df = ExponentialInterestConvention.get_df_from_rate(rate, t)
    assert df == pytest.approx(math.exp(-rate * t))


def test_get_df_from_rate_linear_negative_rate():
    rate, t = -0.01, 1
    df = LinearInterestConvention.get_df_from_rate(rate, t)
    assert df == pytest.approx(1 / (1 + rate * t))
    assert df > 1


def test_get_df_from_rate_compounded_zero_rate_equals_one():
    rate, t = 0, 3
    df = CompoundedInterestConvention.get_df_from_rate(rate, t)
    assert df == pytest.approx(1)


def test_get_df_from_rate_exponential_large_time_fraction():
    import math
    rate, t = 0.02, 10
    df = ExponentialInterestConvention.get_df_from_rate(rate, t)
    assert df == pytest.approx(math.exp(-rate * t))
    assert 0 < df < 1


# --- Rate from discount factor ---

def test_get_rate_from_df_linear_short_term():
    df, t = 0.995, 0.25
    rate = LinearInterestConvention.get_rate_from_df(df, t)
    assert rate == pytest.approx((1 / df - 1) / t)


def test_get_rate_from_df_compounded_long_term():
    df, t = 0.8, 5
    rate = CompoundedInterestConvention.get_rate_from_df(df, t)
    assert rate == pytest.approx((1 / df) ** (1 / t) - 1)


def test_get_rate_from_df_exponential_mid_term():
    import math
    df, t = 0.97, 0.75
    rate = ExponentialInterestConvention.get_rate_from_df(df, t)
    assert rate == pytest.approx(-math.log(df) / t)


def test_get_rate_from_df_linear_df_greater_than_one_gives_negative_rate():
    df, t = 1.01, 2
    rate = LinearInterestConvention.get_rate_from_df(df, t)
    assert rate == pytest.approx((1 / df - 1) / t)
    assert rate < 0


def test_get_rate_from_df_compounded_df_equals_one_gives_zero_rate():
    df, t = 1.0, 4
    rate = CompoundedInterestConvention.get_rate_from_df(df, t)
    assert rate == pytest.approx(0)


def test_get_rate_from_df_exponential_df_greater_than_one_gives_negative_rate():
    import math
    df, t = 1.02, 3
    rate = ExponentialInterestConvention.get_rate_from_df(df, t)
    assert rate == pytest.approx(-math.log(df) / t)
    assert rate < 0


# --- Round-trip consistency ---

@pytest.mark.parametrize("convention", [LinearInterestConvention, CompoundedInterestConvention, ExponentialInterestConvention])
def test_df_from_rate_and_rate_from_df_are_inverses(convention):
    rate, t = 0.035, 1.25
    df = convention.get_df_from_rate(rate, t)
    recovered_rate = convention.get_rate_from_df(df, t)
    assert recovered_rate == pytest.approx(rate)
