from typing import Any

import pytest

from fintoolsom.derivatives.swaps import builders
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

USD = Currency.USD


def _common(ccy: Ccy = Ccy.USD) -> dict:
    return dict(
        notional=NOTIONALS[ccy],
        start_date=START_DATES[SwapStatus.NOT_STARTED],
        term=Term(SWAP_TERM_YEARS, TermUnit.Y, adj_convention(ccy)),
        frequency=PAYMENT_FREQUENCY,
        adj_convention=adj_convention(ccy),
    )


def _fixed(**kw: Any):
    return builders.fixed_leg(
        rate=_rate(FIXED_RATES[Ccy.USD]), currency=USD, **_common(), **kw
    )


def _overnight(**kw: Any):
    return builders.overnight_leg(index=SOFR, spread=SPREAD, **_common(), **kw)


def _xccy(**kw: Any):
    return builders.xccy_floating_leg(index=SOFR, spread=SPREAD, **_common(), **kw)


def test_default_payment_lag_pays_on_period_end():
    leg = _fixed()
    assert leg.payment_dates == leg.end_dates


@pytest.mark.parametrize("make", [_fixed, _overnight, _xccy])
@pytest.mark.parametrize("lag", [1, 2])
def test_payment_date_is_lag_business_days_after_period_end(make, lag):
    base, lagged = make(), make(payment_lag=lag)
    cal = adj_convention(Ccy.USD).calendar
    assert lagged.end_dates == base.end_dates
    assert lagged.start_dates == base.start_dates
    assert lagged.payment_dates == [
        cal.add_business_days(d, lag) for d in base.end_dates
    ]
    assert all(p > e for p, e in zip(lagged.payment_dates, lagged.end_dates))


def test_floating_leg_dispatch_passes_payment_lag():
    leg = builders.floating_leg(index=SOFR, spread=SPREAD, payment_lag=2, **_common())
    assert leg.payment_dates == _overnight(payment_lag=2).payment_dates


def test_fx_fixing_is_measured_back_from_the_lagged_payment_date():
    leg = _xccy(payment_lag=2, fx_fixing_lag=2)
    cal = adj_convention(Ccy.USD).calendar
    assert leg.fx_fixing_dates == [
        cal.add_business_days(p, -2) for p in leg.payment_dates
    ]


def test_negative_payment_lag_raises():
    with pytest.raises(ValueError, match="payment_lag"):
        _fixed(payment_lag=-1)


def test_payment_lag_changes_the_payment_ordinals_only():
    base, lagged = _overnight(), _overnight(payment_lag=2)
    assert (lagged._payment_ordinals > base._payment_ordinals).all()
    assert (lagged._end_ordinals == base._end_ordinals).all()
