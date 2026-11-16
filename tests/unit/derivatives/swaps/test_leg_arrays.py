"""Legs store per-coupon arrays: derived amortizations and flows, read-only arrays,
default dates, validation, coupons built on request and legs built from coupons."""

import pickle
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.derivatives.swaps import (
    FixedCoupon,
    FixedLeg,
    OvernightCoupon,
    OvernightLeg,
    TermRateCoupon,
    TermRateLeg,
    XCCYCoupon,
    XCCYFloatingLeg,
)
from fintoolsom.market import Currency, CurrencyName
from fintoolsom.market.index import OvernightRateIndex, TermRateIndex
from fintoolsom.dates import ModifiedFollowingConvention, Term, TermUnit
from fintoolsom.rates import Rate, RateConvention

USD = Currency(CurrencyName.USD)
SOFR = OvernightRateIndex("SOFR", currency=USD)
TSOFR = TermRateIndex(
    "TSOFR3M", currency=USD, term=Term(3, TermUnit.M, ModifiedFollowingConvention("US"))
)
RATE = Rate(RateConvention(), 0.05)
SPREAD = Rate(RateConvention(), 0.001)
STARTS = [date(2026, 1, 5), date(2026, 4, 6), date(2026, 7, 6)]
ENDS = [date(2026, 4, 6), date(2026, 7, 6), date(2026, 10, 5)]
PAYS = [date(2026, 4, 8), date(2026, 7, 8), date(2026, 10, 7)]
BACKENDS = pytest.mark.parametrize("backend", ["python", "cpp"])


def _fixed(**kwargs) -> FixedLeg:
    fields = dict(
        start_dates=STARTS, end_dates=ENDS, residuals=[100.0, 60.0, 80.0], rate=RATE
    )
    return FixedLeg(**{**fields, **kwargs}, currency=USD)


@BACKENDS
def test_amortizations_repay_the_notional_and_may_be_negative(backend):
    with settings.use_backend(backend):
        leg = _fixed()
    # 100 -> 60 repays 40; 60 -> 80 draws 20 (negative); the last repays the rest.
    np.testing.assert_array_equal(leg.amortizations, [40.0, -20.0, 80.0])
    assert leg.amortizations.sum() == leg.residuals[0]


@BACKENDS
def test_fixed_flows_are_the_interest_plus_the_amortizations(backend):
    with settings.use_backend(backend):
        leg = _fixed()
    interest = [
        RATE.get_accrued_interest(r, s, e)
        for r, s, e in zip(leg.residuals, STARTS, ENDS)
    ]
    np.testing.assert_allclose(leg.fixed_interest, interest, rtol=1e-15)
    np.testing.assert_array_equal(
        leg.fixed_flows, leg.fixed_interest + leg.amortizations
    )
    assert leg.flows is leg.fixed_flows


def test_floating_fixed_interest_is_the_spread_interest():
    leg = OvernightLeg(
        start_dates=STARTS,
        end_dates=ENDS,
        residuals=[1e6] * 3,
        index=SOFR,
        spread=SPREAD,
    )
    expected = 1e6 * (SPREAD.get_wealth_factor(STARTS, ENDS) - 1)
    np.testing.assert_allclose(leg.fixed_interest, expected, rtol=1e-15)
    assert leg.spreads_values is leg.fixed_interest
    no_spread = OvernightLeg(
        start_dates=STARTS, end_dates=ENDS, residuals=[1e6] * 3, index=SOFR
    )
    assert not no_spread.fixed_interest.any() and not no_spread._has_spread


def test_every_array_is_read_only():
    leg = TermRateLeg(
        start_dates=STARTS,
        end_dates=ENDS,
        residuals=[1.0] * 3,
        index=TSOFR,
        spread=SPREAD,
    )
    arrays = [
        leg.residuals,
        leg.amortizations,
        leg.fixed_interest,
        leg.fixed_flows,
        leg._start_ordinals,
        leg._end_ordinals,
        leg._payment_ordinals,
        leg._fx_fixing_ordinals,
        leg._fixing_ordinals,
    ]
    for a in arrays:
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 0
    for a in pickle.loads(pickle.dumps(leg)).__dict__.values():
        if isinstance(a, np.ndarray):
            assert not a.flags.writeable


def test_the_leg_copies_its_inputs():
    residuals = np.array([1.0, 1.0, 1.0])
    starts = list(STARTS)
    leg = _fixed(residuals=residuals, start_dates=starts)
    residuals[0], starts[0] = 5.0, date(2025, 1, 1)
    assert leg.residuals[0] == 1.0 and leg.start_dates[0] == STARTS[0]


def test_payment_and_fx_fixing_dates_default_to_the_end_dates():
    leg = _fixed()
    assert leg.payment_dates == ENDS and leg.fx_fixing_dates == ENDS
    # Shared, not converted again.
    assert leg._payment_ordinals is leg._end_ordinals
    assert leg._fx_fixing_ordinals is leg._payment_ordinals
    lagged = _fixed(payment_dates=PAYS)
    assert lagged.fx_fixing_dates == PAYS
    assert lagged._fx_fixing_ordinals is lagged._payment_ordinals


def test_term_rate_fixings_default_to_the_start_dates_and_cannot_be_late():
    leg = TermRateLeg(
        start_dates=STARTS, end_dates=ENDS, residuals=[1.0] * 3, index=TSOFR
    )
    assert leg.fixing_dates == STARTS
    with pytest.raises(ValueError, match="fixing date is after the start date"):
        TermRateLeg(
            start_dates=STARTS,
            end_dates=ENDS,
            residuals=[1.0] * 3,
            index=TSOFR,
            fixing_dates=[STARTS[0], ENDS[1], STARTS[2]],
        )
    with pytest.raises(ValueError, match="one fixing date per coupon"):
        TermRateLeg(
            start_dates=STARTS,
            end_dates=ENDS,
            residuals=[1.0] * 3,
            index=TSOFR,
            fixing_dates=STARTS[:2],
        )


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"residuals": [1.0, 1.0]}, "one value per coupon"),
        ({"end_dates": ENDS[:2]}, "one value per coupon"),
        ({"end_dates": [STARTS[0], *ENDS[1:]]}, "Coupon 0: a start date is not before"),
        (
            {"payment_dates": [ENDS[0], date(2026, 7, 1), PAYS[2]]},
            "Coupon 1: an end date is after its payment date",
        ),
        (
            {"fx_fixing_dates": [ENDS[0], ENDS[1], date(2026, 10, 9)]},
            "Coupon 2: an FX fixing date is after its payment date",
        ),
    ],
)
def test_inconsistent_dates_or_lengths_are_rejected(fields, match):
    with pytest.raises(ValueError, match=match):
        _fixed(**fields)


@pytest.mark.parametrize(
    "leg, coupon_cls",
    [
        (_fixed(payment_dates=PAYS), FixedCoupon),
        (
            OvernightLeg(
                start_dates=STARTS, end_dates=ENDS, residuals=[2.0] * 3, index=SOFR
            ),
            OvernightCoupon,
        ),
        (
            XCCYFloatingLeg(
                start_dates=STARTS, end_dates=ENDS, residuals=[2.0] * 3, index=SOFR
            ),
            XCCYCoupon,
        ),
        (
            TermRateLeg(
                start_dates=STARTS,
                end_dates=ENDS,
                residuals=[2.0] * 3,
                index=TSOFR,
                fixing_dates=[date(2026, 1, 2), date(2026, 4, 2), date(2026, 7, 2)],
            ),
            TermRateCoupon,
        ),
    ],
    ids=["fixed", "overnight", "xccy", "term"],
)
def test_coupons_are_built_on_request_and_round_trip(leg, coupon_cls):
    coupons = leg.get_coupons()
    assert len(coupons) == leg.n_coupons == 3
    assert all(type(c) is coupon_cls for c in coupons)
    assert [c.payment_date for c in coupons] == leg.payment_dates
    assert [c.residual for c in coupons] == leg.residuals.tolist()
    assert [c.amortization for c in coupons] == leg.amortizations.tolist()
    assert leg.get_coupon(-1) == coupons[-1]
    assert leg.get_coupons() is not coupons  # not stored
    with pytest.raises(IndexError):
        leg.get_coupon(3)
    if coupon_cls is FixedCoupon:
        np.testing.assert_allclose([c.flow for c in coupons], leg.flows, rtol=1e-15)
        kwargs = {"currency": leg.currency}
    else:
        kwargs = {"index": leg.index, "spread": leg.spread}
    if coupon_cls is TermRateCoupon:
        assert [c.fixing_date for c in coupons] == leg.fixing_dates
    rebuilt = type(leg).from_coupons(coupons, **kwargs)
    for name in ("residuals", "amortizations", "fixed_flows", "_payment_ordinals"):
        np.testing.assert_array_equal(getattr(rebuilt, name), getattr(leg, name))
    if coupon_cls is TermRateCoupon:
        assert rebuilt.fixing_dates == leg.fixing_dates


def test_from_coupons_takes_one_rate_from_the_coupons():
    coupons = _fixed().get_coupons()
    assert FixedLeg.from_coupons(coupons, currency=USD).rate is RATE
    coupons[1] = replace(coupons[1], rate=Rate(RateConvention(), 0.06))
    with pytest.raises(ValueError, match="one rate"):
        FixedLeg.from_coupons(coupons, currency=USD)


def test_from_coupons_rejects_amortizations_not_repaying_the_notional():
    coupons = _fixed().get_coupons()
    coupons[-1] = replace(coupons[-1], amortization=0.0)
    with pytest.raises(ValueError, match="repay the notional"):
        FixedLeg.from_coupons(coupons, currency=USD)


def test_replace_recomputes_the_derived_arrays():
    leg = _fixed()
    bullet = replace(leg, residuals=[100.0] * 3)
    np.testing.assert_array_equal(bullet.amortizations, [0.0, 0.0, 100.0])
    assert bullet._core is None


def test_an_empty_leg():
    leg = FixedLeg(start_dates=[], end_dates=[], residuals=[], rate=RATE, currency=USD)
    assert leg.n_coupons == 0 and leg.get_coupons() == [] and leg.flows.size == 0
    assert "(no coupons)" in str(leg)
