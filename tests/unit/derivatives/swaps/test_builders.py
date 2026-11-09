from datetime import date
import pytest
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.legs import OvernightLeg, TermRateLeg, XCCYFloatingLeg
from tests.factories.swap_factory import (
    CAMARA,
    ICP,
    SPREAD,
    TAB_3M,
    Ccy,
    adj_convention,
)


ADJ_CLP = adj_convention(Ccy.CLP)


def _leg(builder, index, start=date(2024, 7, 15), years=2, **kw):
    return builder(
        notional=1_000.0,
        start_date=start,
        term=Term(years, TermUnit.Y, ADJ_CLP),
        frequency="QUARTERLY",
        adj_convention=ADJ_CLP,
        index=index,
        **kw,
    )


def test_spread_defaults_to_a_fresh_zero_rate_per_leg():
    a, b = _leg(builders.overnight_leg, CAMARA), _leg(builders.term_rate_leg, TAB_3M)
    assert a.spread.rate == 0.0 and b.spread.rate == 0.0
    assert a.spread is not b.spread  # bumping one leg's spread must not move another's
    assert list(a.spreads_values) == [0.0] * a.n_coupons


def test_term_leg_fixes_spot_lag_business_days_before_each_start():
    leg = _leg(builders.term_rate_leg, TAB_3M)
    calendar = ADJ_CLP.calendar
    assert TAB_3M.spot_lag == 2
    assert leg.fixing_dates == [
        calendar.add_business_days(s, -2) for s in leg.start_dates
    ]
    assert (
        _leg(builders.term_rate_leg, TAB_3M, fixing_lag=0).fixing_dates
        == leg.start_dates
    )


@pytest.mark.parametrize(
    "index, cross_currency, expected",
    [
        (CAMARA, False, OvernightLeg),
        (ICP, False, OvernightLeg),
        (CAMARA, True, XCCYFloatingLeg),
        (TAB_3M, False, TermRateLeg),
        (TAB_3M, True, TermRateLeg),  # a term index never compounds daily
    ],
)
def test_floating_leg_builds_the_kind_the_index_calls_for(
    index, cross_currency, expected
):
    leg = _leg(
        builders.floating_leg, index, cross_currency=cross_currency, spread=SPREAD
    )
    assert type(leg) is expected
