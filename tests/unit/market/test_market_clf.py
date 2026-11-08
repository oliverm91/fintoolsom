import math
from datetime import date, timedelta
import pytest
from fintoolsom.market import Currency, Market, OvernightRateIndex, uf
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from fintoolsom.rates import DiscountCurve, IndexCurve



CLF, CLP, USD = Currency.CLF, Currency.CLP, Currency.USD


CLFCLP = CurrencyPair(CLF, CLP)


RISKLESS = OvernightRateIndex("OIS", currency=CLP)


# Before the 8th (CPI release): the UF is published up to the 9th of this month.
T = date(2024, 8, 4)


KNOWN_UNTIL = date(2024, 8, 9)  # T + 5


R_CLP, R_CLF = 0.06, 0.02


UF_T = 37_500.0


def uf_level(d: date) -> float:
    return UF_T * 1.0001 ** (d - T).days


def _flat_curve(rate: float, currency: Currency | None = None) -> IndexCurve | DiscountCurve:
    """Flat continuous zero rate: the risk-free index curve, or `currency`'s discount curve."""
    if currency is None:
        return IndexCurve(T, RISKLESS, [T, T + timedelta(days=1825)], [rate])
    pillars = [T + timedelta(days=d) for d in (1, 7, 30, 91, 365, 730, 1825)]
    return DiscountCurve(T, currency, RISKLESS, pillars, [math.exp(-rate * (d - T).days / 365) for d in pillars])


def _market() -> Market:
    # Load only what is published as of T: up to KNOWN_UNTIL.
    assert uf.last_published_date(T) == KNOWN_UNTIL
    levels = {T + timedelta(days=i): uf_level(T + timedelta(days=i)) for i in range(-60, 6)}
    return Market(
        t=T,
        risk_free_index=RISKLESS,
        curves=[_flat_curve(R_CLP), _flat_curve(R_CLF, CLF)],
        fx_history={CLFCLP: uf.uf_rate_data(levels)},
    )


def test_market_reads_clf_rates_including_published_future_levels():
    market = _market()
    assert market.get_fx_rate(T, CLFCLP).value == UF_T
    assert market.get_fx_rate(KNOWN_UNTIL, CLFCLP).value == uf_level(KNOWN_UNTIL)
    assert market.get_fx_rate(T, CLFCLP.invert()).value == pytest.approx(1 / UF_T)
    with pytest.raises(KeyError):
        market.get_fx_rate(KNOWN_UNTIL + timedelta(days=1), CLFCLP)


def test_clf_rates_can_be_added_like_any_fx_rate():
    market = _market()
    market.add_fx_rate(KNOWN_UNTIL + timedelta(days=1), FX_Rate(CLFCLP, 40_000.0))
    assert market.get_fx_rate(KNOWN_UNTIL + timedelta(days=1), CLFCLP).value == 40_000.0
    assert market.get_fx_known_until(CLFCLP) == KNOWN_UNTIL + timedelta(days=1)


def test_fx_known_until_is_the_latest_loaded_date_not_before_today():
    market = _market()
    assert market.get_fx_known_until(CLFCLP) == KNOWN_UNTIL
    assert market.get_fx_known_until(CLFCLP.invert()) == KNOWN_UNTIL
    usdclp = CurrencyPair(USD, CLP)
    assert market.get_fx_known_until(usdclp) == T  # nothing loaded
    market.add_fx_rate(T - timedelta(days=3), FX_Rate(usdclp, 940.0))
    assert market.get_fx_known_until(usdclp) == T  # only past data
