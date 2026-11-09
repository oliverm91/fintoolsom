"""Market rate lookups and fixing-based accruals."""

from datetime import date, timedelta

import pytest

from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.market.currencies import CurrencyPair, FX_Rate, FX_RateData
from fintoolsom.rates import LinearInterestConvention, Rate, RateConvention

LIN360 = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)
N = 1_000_000.0


def _lin(r: float) -> Rate:
    return Rate(LIN360.copy(), r)


def _flat_series(r: float, start: date, days: int) -> dict[date, Rate]:
    return {start + timedelta(days=i): _lin(r) for i in range(days)}


def test_business_day_accrual_compounds_daily_with_weekend_steps():
    # Mon 1 Jan -> Mon 8 Jan 2024 (no holidays): four 1-day steps and one 3-day step.
    r = 0.05
    market = Market(
        t=date(2024, 1, 8),
        interest_rates={"ON": _flat_series(r, date(2023, 12, 25), 20)},
    )
    accrued = market.accrue_rates_reset_business_days(
        N, "ON", date(2024, 1, 1), date(2024, 1, 8)
    )
    expected = N * ((1 + r / 360) ** 4 * (1 + 3 * r / 360) - 1)
    assert accrued == pytest.approx(expected, rel=1e-12)


def test_business_day_accrual_reads_fixings_lagged_and_falls_back_to_past_ones():
    rates = {date(2023, 12, 29): _lin(0.04)}  # only a fixing before the period
    market = Market(t=date(2024, 1, 3), interest_rates={"ON": rates})
    accrued = market.accrue_rates_reset_business_days(
        N,
        "ON",
        date(2024, 1, 1),
        date(2024, 1, 3),
        fixing_lag=1,
        use_closest_past_rate_for_fixing=True,
    )
    assert accrued == pytest.approx(N * ((1 + 0.04 / 360) ** 2 - 1), rel=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixing_lag": -1},
        {"start_date": date(2024, 1, 9)},
    ],
)
def test_business_day_accrual_rejects_bad_inputs(kwargs):
    market = Market(
        t=date(2024, 1, 8),
        interest_rates={"ON": _flat_series(0.05, date(2024, 1, 1), 10)},
    )
    args = {
        "notional": N,
        "rate_name": "ON",
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 1, 8),
    }
    with pytest.raises(ValueError):
        market.accrue_rates_reset_business_days(**{**args, **kwargs})


def test_custom_reset_accrual_compounds_each_reset_period():
    r1, r2 = 0.05, 0.07
    market = Market(
        t=date(2024, 2, 1),
        interest_rates={
            "TERM": {date(2024, 1, 1): _lin(r1), date(2024, 1, 15): _lin(r2)}
        },
    )
    accrued = market.accrue_rates_custom_reset_days(
        N,
        "TERM",
        date(2024, 1, 1),
        date(2024, 2, 1),
        [date(2024, 1, 15), date(2024, 1, 1)],
    )
    # 14 days at r1, then 17 days at r2 on the grown notional.
    assert accrued == pytest.approx(
        N * ((1 + r1 * 14 / 360) * (1 + r2 * 17 / 360) - 1), rel=1e-12
    )


def test_custom_reset_accrual_stops_at_the_end_date():
    market = Market(
        t=date(2024, 3, 1),
        interest_rates={
            "TERM": {date(2024, 1, 1): _lin(0.05), date(2024, 3, 1): _lin(0.09)}
        },
    )
    accrued = market.accrue_rates_custom_reset_days(
        N,
        "TERM",
        date(2024, 1, 1),
        date(2024, 1, 31),
        [date(2024, 1, 1), date(2024, 3, 1)],
    )
    assert accrued == pytest.approx(N * 0.05 * 30 / 360, rel=1e-12)


def test_custom_reset_accrual_rejects_bad_inputs():
    market = Market(
        t=date(2024, 3, 1), interest_rates={"TERM": {date(2024, 1, 1): _lin(0.05)}}
    )
    with pytest.raises(ValueError):  # first reset after the start
        market.accrue_rates_custom_reset_days(
            N, "TERM", date(2024, 1, 1), date(2024, 2, 1), [date(2024, 1, 2)]
        )
    with pytest.raises(ValueError):  # start after end
        market.accrue_rates_custom_reset_days(
            N, "TERM", date(2024, 3, 1), date(2024, 2, 1), [date(2024, 1, 1)]
        )


def test_single_reset_accrual_is_simple_interest_at_the_fixing():
    market = Market(
        t=date(2024, 3, 1), interest_rates={"TERM": {date(2024, 1, 1): _lin(0.06)}}
    )
    accrued = market.accrue_rates_single_reset_day(
        N, "TERM", date(2024, 1, 1), date(2024, 4, 1), date(2024, 1, 1)
    )
    assert accrued == pytest.approx(N * 0.06 * 91 / 360, rel=1e-12)
    with pytest.raises(ValueError):
        market.accrue_rates_single_reset_day(
            N, "TERM", date(2024, 2, 1), date(2024, 1, 1), date(2024, 1, 1)
        )
    with pytest.raises(ValueError):
        market.accrue_rates_single_reset_day(
            N, "TERM", date(2024, 1, 1), date(2024, 4, 1), date(2024, 1, 2)
        )


def test_unknown_rate_series_raises():
    with pytest.raises(ValueError):
        Market(t=date(2024, 1, 1)).get_rate(date(2024, 1, 1), "NOPE")


# ── Construction and lookups ────────────────────────────────────────────────


def test_risk_free_index_needs_a_currency():
    with pytest.raises(ValueError):
        Market(t=date(2024, 1, 1), risk_free_index=OvernightRateIndex("X"))


def test_fx_history_keys_must_match_their_data():
    usdclp = CurrencyPair(Currency.USD, Currency.CLP)
    eurclp = CurrencyPair(Currency.EUR, Currency.CLP)
    data = FX_RateData(usdclp, {date(2024, 1, 1): FX_Rate(usdclp, 900.0)})
    with pytest.raises(ValueError):
        Market(t=date(2024, 1, 1), fx_history={eurclp: data})


def test_missing_lookups_raise_key_errors():
    market = Market(t=date(2024, 1, 1))
    index = OvernightRateIndex("SOFR", currency=Currency.USD)
    with pytest.raises(KeyError, match="No index curve for 'SOFR'"):
        market.get_index_curve(index)
    with pytest.raises(ValueError, match="no risk_free_index"):
        market.get_discount_curve(Currency.USD)
    with pytest.raises(KeyError, match="No discount curve 'CLP_SOFR'"):
        market.get_discount_curve(Currency.CLP, risk_free_index=index)
    with pytest.raises(KeyError):
        market.get_volatility_surface(CurrencyPair(Currency.USD, Currency.CLP))
    with pytest.raises(KeyError):
        market.get_fx_rate(date(2024, 1, 1), CurrencyPair(Currency.USD, Currency.CLP))


def test_fx_rate_of_a_currency_against_itself_is_one():
    pair = CurrencyPair(Currency.CLP, Currency.CLP)
    assert Market(t=date(2024, 1, 1)).get_fx_rate(date(2024, 1, 1), pair).value == 1.0
