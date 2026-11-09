from datetime import date
from typing import Any
import pytest
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.coupons import XCCYCoupon
from fintoolsom.derivatives.swaps.legs import OvernightLeg, XCCYFloatingLeg
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency
from tests.factories.swap_factory import (
    FIXED_RATES,
    NOTIONALS,
    PAYMENT_FREQUENCY,
    SOFR,
    SPREAD,
    START_DATES,
    SWAP_TERM_YEARS,
    Ccy,
    SwapStatus,
    _rate,
    adj_convention,
)


USD, CLP = Currency.USD, Currency.CLP


def _common(ccy: Ccy, status: SwapStatus) -> dict:
    return dict(
        notional=NOTIONALS[ccy],
        start_date=START_DATES[status],
        term=Term(SWAP_TERM_YEARS, TermUnit.Y, adj_convention(ccy)),
        frequency=PAYMENT_FREQUENCY,
        adj_convention=adj_convention(ccy),
    )


def _xccy_usd(
    status: SwapStatus = SwapStatus.NOT_STARTED, fx_fixing_lag: int = 0
) -> Any:
    return builders.xccy_floating_leg(
        index=SOFR,
        spread=SPREAD,
        fx_fixing_lag=fx_fixing_lag,
        **_common(Ccy.USD, status),
    )


def _fixed(
    ccy: Ccy, status: SwapStatus = SwapStatus.NOT_STARTED, fx_fixing_lag: int = 0
) -> Any:
    return builders.fixed_leg(
        rate=_rate(FIXED_RATES[ccy]),
        currency=USD if ccy is Ccy.USD else CLP,
        fx_fixing_lag=fx_fixing_lag,
        **_common(ccy, status),
    )


def test_xccy_leg_is_a_daily_compounded_leg_of_xccy_coupons():
    leg = _xccy_usd()
    assert isinstance(leg, XCCYFloatingLeg) and isinstance(leg, OvernightLeg)
    assert all(isinstance(c, XCCYCoupon) for c in leg.coupons)
    assert leg.currency == USD
    assert leg.n_coupons == SWAP_TERM_YEARS * 4


def test_only_the_final_notional_is_exchanged():
    leg = _xccy_usd()
    assert list(leg.amortizations[:-1]) == [0.0] * (leg.n_coupons - 1)
    assert leg.amortizations[-1] == NOTIONALS[Ccy.USD]
    assert min(leg.payment_dates) > leg.start_dates[0]  # no flow at the start date


def test_fx_fixing_dates_default_to_payment_dates():
    for leg in (_xccy_usd(), _fixed(Ccy.USD)):
        assert leg.fx_fixing_dates == leg.payment_dates
        assert [c.fx_fixing_date for c in leg.coupons] == leg.payment_dates


def test_fx_fixing_lag_counts_business_days_back_from_payment():
    calendar = adj_convention(Ccy.USD).calendar
    leg = _xccy_usd(fx_fixing_lag=2)
    assert leg.fx_fixing_dates == [
        calendar.add_business_days(p, -2) for p in leg.payment_dates
    ]
    assert all(f < p for f, p in zip(leg.fx_fixing_dates, leg.payment_dates))


def test_invalid_fx_fixing_dates_raise():
    with pytest.raises(ValueError):
        _xccy_usd(fx_fixing_lag=-1)
    with pytest.raises(ValueError, match="fx_fixing_date"):
        d = date(2025, 1, 10)
        XCCYCoupon(
            1.0,
            0.0,
            date(2024, 10, 10),
            d,
            d,
            fx_fixing_date=date(2025, 1, 13),
            spread=SPREAD,
        )
