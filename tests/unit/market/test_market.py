from datetime import date
import pytest
from fintoolsom.market import Market
from tests.factories.swap_factory import (
    _rate,
)


def test_get_rate_falls_back_to_the_latest_past_fixing():
    d1, d2, d3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    market = Market(t=d3, interest_rates={"x": {d1: _rate(0.01), d3: _rate(0.03)}})
    assert market.get_rate(d2, "X", use_closest_past_rate=True).rate == 0.01
    with pytest.raises(ValueError):
        market.get_rate(d2, "X")
    with pytest.raises(ValueError):
        market.get_rate(date(2023, 1, 1), "X", use_closest_past_rate=True)


def test_get_rate_falls_back_across_more_than_a_month_without_fixings():
    old, recent = date(2023, 5, 10), date(2023, 12, 20)
    market = Market(
        t=date(2024, 3, 1),
        interest_rates={"X": {old: _rate(0.01), recent: _rate(0.02)}},
    )
    assert (
        market.get_rate(date(2023, 12, 21), "X", use_closest_past_rate=True).rate
        == 0.02
    )
    assert (
        market.get_rate(date(2024, 2, 28), "X", use_closest_past_rate=True).rate == 0.02
    )
    assert (
        market.get_rate(date(2023, 12, 19), "X", use_closest_past_rate=True).rate
        == 0.01
    )
