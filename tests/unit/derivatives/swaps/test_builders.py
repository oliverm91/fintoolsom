from datetime import date
import pytest
from fintoolsom.dates import ActualDayCountConvention, Days30EDayCountConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import _swap_pv
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.legs import (
    OvernightLeg,
    TermRateLeg,
    XCCYFloatingLeg,
    _zero_spread,
)
from fintoolsom.market import Currency, OvernightRateIndex
from fintoolsom.market.index import TermRateIndex
from fintoolsom.rates import (
    CompoundedInterestConvention,
    LinearInterestConvention,
    Rate,
    RateConvention,
)
from tests.factories.swap_factory import (
    CAMARA,
    ICP,
    SPREAD,
    TAB_3M,
    Ccy,
    adj_convention,
    build_market,
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


# ── The spread of a leg built without one ───────────────────────────────────

LIN_ACT_365 = RateConvention(LinearInterestConvention, ActualDayCountConvention, 365)
SONIA = OvernightRateIndex("SONIA", currency=Currency.GBP, rate_convention=LIN_ACT_365)
TERM_30E = TermRateIndex(  # a term rate quoted compounded 30E/360
    "TERM_30E",
    currency=Currency.CLP,
    term=Term(3, TermUnit.M, ADJ_CLP),
    rate_convention=RateConvention(
        CompoundedInterestConvention, Days30EDayCountConvention, 360
    ),
)
DEFAULT_SPREAD_CASES = [
    pytest.param(builders.overnight_leg, CAMARA, id="overnight-rate"),
    pytest.param(builders.overnight_leg, ICP, id="interest-price"),
    pytest.param(builders.term_rate_leg, TAB_3M, id="term-rate"),
    pytest.param(builders.xccy_floating_leg, CAMARA, id="xccy"),
    pytest.param(builders.floating_leg, ICP, id="floating"),
    pytest.param(builders.overnight_leg, SONIA, id="act-365-index"),
    pytest.param(builders.term_rate_leg, TERM_30E, id="compounded-30e-index"),
]


@pytest.mark.parametrize("builder, index", DEFAULT_SPREAD_CASES)
def test_default_spread_is_zero_and_linear_in_the_index_basis(builder, index):
    spread = _leg(builder, index).spread
    own = index.rate_convention
    assert spread.rate == 0.0
    # Linear whatever the index's interest convention: a spread is added, not compounded.
    assert spread.rate_convention == RateConvention(
        LinearInterestConvention, own.day_count_convention, own.time_fraction_base
    )
    assert spread.rate_convention is not own  # the index's convention is not shared


def test_every_default_spread_is_a_fresh_rate():
    a, b = _leg(builders.overnight_leg, CAMARA), _leg(builders.overnight_leg, CAMARA)
    assert a.spread == b.spread
    assert a.spread is not b.spread  # bumping one leg's spread must not move another's
    assert a.spread.rate_convention is not b.spread.rate_convention
    assert list(a.spreads_values) == [0.0] * a.n_coupons


def test_an_index_without_a_rate_convention_gets_a_linear_act_360_spread():
    spread = _zero_spread(object())
    assert spread.rate == 0.0
    assert spread.rate_convention == RateConvention(
        LinearInterestConvention, ActualDayCountConvention, 360
    )


@pytest.mark.parametrize(
    "builder, index",
    [(builders.overnight_leg, CAMARA), (builders.term_rate_leg, TAB_3M)],
    ids=["overnight", "term"],
)
def test_the_default_spread_values_the_leg_as_any_zero_spread(builder, index):
    market = build_market()
    default = _leg(builder, index)
    compounded = _leg(builder, index, spread=Rate(RateConvention(), 0.0))
    assert not default._has_spread
    assert list(default.fixed_flows) == list(compounded.fixed_flows)
    assert _swap_pv._leg_pv(default, index, market) == _swap_pv._leg_pv(
        compounded, index, market
    )


@pytest.mark.parametrize("index, base", [(CAMARA, 360), (SONIA, 365)], ids=["360", "365"])
def test_a_bump_of_the_default_spread_accrues_simple_interest_in_the_index_basis(
    index, base
):
    # What moves a spread (the curve builder's 1bp off-par report) keeps its convention.
    leg = _leg(builders.overnight_leg, index)
    bumped = Rate(leg.spread.rate_convention, 1e-4)
    start, end = leg.start_dates[0], leg.end_dates[0]
    assert bumped.get_wealth_factor(start, end) == pytest.approx(
        1 + 1e-4 * (end - start).days / base, rel=1e-15
    )


@pytest.mark.parametrize(
    "builder, index",
    [(builders.overnight_leg, CAMARA), (builders.term_rate_leg, TAB_3M)],
)
def test_fixed_flows_are_spread_interest_plus_amortization(builder, index):
    leg = _leg(builder, index, spread=SPREAD)
    assert leg.spreads_values.any()
    assert list(leg.fixed_flows) == pytest.approx(
        list(leg.spreads_values + leg.amortizations)
    )
    assert leg.fixed_flows[-1] == pytest.approx(leg.spreads_values[-1] + 1_000.0)


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
