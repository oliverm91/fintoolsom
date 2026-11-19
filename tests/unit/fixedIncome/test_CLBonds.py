"""Coupon, Coupons and CLBond on synthetic coupons: validation, copies, accessors,
valuation and round trips. Real bonds on the real market: test_CLBonds_real_market.py.
"""

from copy import copy
from datetime import date

import pytest

from fintoolsom.fixedIncome import CLBond, Coupon, Coupons
from tests.factories import clbond_factory, curve_factory
from tests.factories.rate_factory import (
    comp_act365,
    exp_act365,
    lin_30e360,
    lin_act360,
    rate,
)

S, E = date(2024, 1, 1), date(2024, 7, 1)


# ── Coupon ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "args, error",
    [
        (("0", 2, 100, S, E), TypeError),  # amortization
        ((0, 2, "100", S, E), TypeError),  # residual
        ((0, 2, 0, S, E), ValueError),  # non-positive residual
        ((150, 2, 100, S, E), ValueError),  # amortization above residual
        ((0, "2", 100, S, E), TypeError),  # interest
        ((0, 2, 100, "2024-01-01", E), TypeError),
        ((0, 2, 100, S, "2024-07-01"), TypeError),
        ((0, 2, 100, E, S), ValueError),  # start after end
    ],
)
def test_coupon_validates_its_inputs(args, error):
    with pytest.raises(error):
        Coupon(*args, lin_30e360())


def test_coupon_needs_a_rate_convention():
    with pytest.raises(TypeError):
        Coupon(0, 2, 100, S, E, "lin30/360")


def test_coupon_implied_accrual_rate_reproduces_its_interest():
    coupon = Coupon(0, 2, 100, S, E, lin_30e360())
    assert coupon.flow == 2.0 and coupon.wf == pytest.approx(1.02)
    assert coupon.accrue_rate.rate == pytest.approx(0.04)  # 2 over 180/360 days
    assert coupon.get_accrued_interest(date(2024, 4, 1)) == pytest.approx(
        100 * 0.04 * 90 / 360
    )
    assert coupon.get_accrued_interest(S) == 0 and coupon.get_accrued_interest(E) == 0


def test_coupon_accrues_at_a_given_rate():
    coupon = Coupon(0, 2, 100, S, E, lin_30e360())
    assert coupon.get_accrued_interest(
        date(2024, 4, 1), rate(0.08, lin_30e360())
    ) == pytest.approx(2.0)


def test_coupon_copy_is_equal_and_independent():
    coupon = Coupon(0, 2, 100, S, E, lin_30e360())
    duplicate = copy(coupon)
    assert duplicate is not coupon and duplicate.flow == coupon.flow
    duplicate.amortization = 1.0
    assert coupon.amortization == 0.0


# ── Coupons ─────────────────────────────────────────────────────────────────


def test_coupons_must_be_a_list_of_coupon():
    with pytest.raises(TypeError):
        Coupons(tuple(clbond_factory.two_coupon_coupons().coupons))
    with pytest.raises(TypeError):
        Coupons(["coupon"])


def test_coupons_are_sorted_by_start_date():
    first, second = clbond_factory.two_coupon_coupons().coupons
    assert Coupons([second, first]).coupons == [first, second]


def test_coupons_reject_residuals_inconsistent_with_amortizations():
    conv = lin_30e360()
    with pytest.raises(ValueError, match="Residual of coupon"):
        Coupons(
            [
                Coupon(0, 2, 100, S, E, conv),
                Coupon(50, 2, 100, E, date(2025, 1, 1), conv),
            ],  # repays 50 of 100
            check_residuals=True,
        )


def test_coupons_accessors():
    coupons = clbond_factory.two_coupon_coupons()
    assert list(coupons.get_flows_between_dates(S, E)) == [2.0]
    assert list(coupons.get_flows_maturities(S)) == [182, 366]
    assert coupons.get_current_coupon(date(2025, 6, 1)) is None
    assert coupons.get_residual_amount(date(2024, 3, 1)) == 100
    assert coupons.get_accrued_interest(date(2024, 4, 1)) == pytest.approx(
        100 * 0.04 * 90 / 360
    )
    assert coupons.get_accrue_rate().rate == pytest.approx(0.04)
    assert list(coupons.get_remaining_flows(E)) == [
        102
    ]  # the coupon ending on E is gone
    boundary = coupons.get_current_coupon(E)  # a boundary belongs to the next coupon
    assert (boundary.start_date, boundary.end_date) == (E, date(2025, 1, 1))


def test_coupons_copy_is_independent():
    coupons = clbond_factory.two_coupon_coupons()
    duplicate = copy(coupons)
    duplicate.adjust_to_notional(1_000)
    assert (
        coupons.coupons[-1].amortization == 100
        and duplicate.coupons[-1].amortization == 1_000
    )


def test_adjust_to_notional_scales_every_amount():
    coupons = clbond_factory.two_coupon_coupons()

    def amounts():
        return [
            x
            for c in coupons.coupons
            for x in (c.amortization, c.interest, c.residual, c.flow)
        ]

    before = amounts()
    coupons.adjust_to_notional(50_000_000)
    assert amounts() == pytest.approx([x * 500_000 for x in before], rel=1e-14)
    assert list(coupons.flows) == [c.flow for c in coupons.coupons]


def test_adjust_to_notional_rejects_amortizations_that_do_not_repay_it():
    conv = lin_30e360()
    coupons = Coupons([Coupon(50, 2, 100, S, E, conv)], check_residuals=False)
    coupons.check_residuals = True
    with pytest.raises(ValueError, match="Could not adjust"):
        coupons.adjust_to_notional(1_000)


def test_amortizing_coupons_factory_repays_the_notional():
    coupons = clbond_factory.amortizing_coupons(S, 4, 0.06)
    assert sum(c.amortization for c in coupons.coupons) == pytest.approx(100)
    assert [c.residual for c in coupons.coupons] == pytest.approx([100, 75, 50, 25])


# ── CLBond ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(coupons="not coupons"), TypeError),
        (dict(currency=123), TypeError),
        (dict(notional=0), ValueError),
    ],
    ids=["coupons", "currency", "notional"],
)
def test_bond_validates_its_inputs(kwargs, error):
    args = dict(
        coupons=clbond_factory.two_coupon_coupons(), currency="clp", notional=100
    )
    with pytest.raises(error):
        CLBond(**(args | kwargs))


def test_bond_copy_keeps_terms_and_tera():
    bond = clbond_factory.btp0470930()
    duplicate = copy(bond)
    assert duplicate is not bond and duplicate.tera.rate == bond.tera.rate
    assert (
        duplicate.get_maturity_date() == date(2030, 9, 1)
        and duplicate.notional == bond.notional
    )


def test_bond_accrued_interest_is_the_current_coupon_accrual():
    bond = clbond_factory.two_coupon_bond()
    assert bond.get_accrued_interest(date(2024, 4, 1)) == pytest.approx(
        100 * 0.04 * 90 / 360
    )


def test_amortizing_bond_tera_at_issue_is_its_coupon_rate():
    bond = CLBond(
        clbond_factory.amortizing_coupons(S, 4, 0.06),
        currency="clp",
        notional=1_000_000,
    )
    assert bond.get_present_value(S, bond.tera) == pytest.approx(100, abs=1e-4)
    assert bond.get_par_value(S) == pytest.approx(100)


def test_irr_with_curve_reprices_the_curve_value():
    bond = clbond_factory.bullet_clbond(
        start=date(2024, 3, 1), n_coupons=10, coupon_amount=2.5
    )
    curve = curve_factory.sample_curve()
    t = curve.curve_date
    irr = bond.get_irr_with_zcc(t, curve, comp_act365())
    assert bond.get_present_value(t, irr) == pytest.approx(
        bond.get_present_value_zc(t, curve), rel=1e-9
    )


def test_z_spread_reprices_the_irr_value_on_the_bumped_curve(
    sample_bond, sample_curve, curve_date
):
    irr = rate(0.06)
    z_spread = sample_bond.get_z_spread(curve_date, irr, sample_curve)
    bumped = sample_curve.parallel_bump_bps(z_spread)
    assert sample_bond.get_present_value_zc(curve_date, bumped) == pytest.approx(
        sample_bond.get_present_value(curve_date, irr), abs=1e-4
    )


def test_duration_is_below_maturity_and_dv01_uses_the_modified_duration():
    bond = clbond_factory.btp0470930()
    t, irr = date(2026, 6, 24), rate(0.05)
    duration = bond.get_duration_from_rate(t, irr)  # Macaulay
    assert 0 < duration < (bond.get_maturity_date() - t).days / 365
    pv = bond.get_present_value(t, irr)
    expected = -pv * duration / 1.05 / 10_000 * bond.notional / 100
    assert bond.get_dv01(t, irr) == pytest.approx(expected, rel=1e-12) and expected < 0


@pytest.mark.parametrize("convention", [comp_act365, exp_act365, lin_act360])
def test_dv01_is_the_value_change_for_one_basis_point(convention):
    bond = clbond_factory.btp0470930()
    t, y, h = date(2026, 6, 24), 0.05, 1e-6

    def value(rate_value):
        return bond.get_present_value(t, rate(rate_value, convention()))

    slope = (value(y + h) - value(y - h)) / (2 * h)  # dPV/dy, base 100
    expected = slope / 10_000 * bond.notional / 100
    assert bond.get_dv01(t, rate(y, convention())) == pytest.approx(expected, rel=1e-7)


def test_a_bond_on_coupons_in_another_base_is_the_same_bond():
    """CLBond rescales its coupons to base 100: interests and flows too (B1)."""
    base_100 = clbond_factory.btp0470930()
    k = 10_000.0  # coupons of a 1,000,000 face value
    scaled = Coupons(
        [
            Coupon(
                c.amortization * k,
                c.interest * k,
                c.residual * k,
                c.start_date,
                c.end_date,
                c.accrue_rate_convention.copy(),
            )
            for c in base_100.coupons.coupons
        ],
        check_residuals=True,
    )
    bond = CLBond(scaled, base_100.currency, base_100.notional)
    t, irr = date(2026, 6, 24), rate(0.05)
    assert bond.tera.rate == base_100.tera.rate
    assert list(bond.flows_amount) == pytest.approx(
        list(base_100.flows_amount), rel=1e-14
    )
    assert list(bond.coupons.flows) == pytest.approx(
        list(base_100.coupons.flows), rel=1e-14
    )
    assert bond.get_present_value(t, irr) == pytest.approx(
        base_100.get_present_value(t, irr), rel=1e-14
    )
    assert bond.get_price(t, irr) == base_100.get_price(t, irr)
    assert bond.get_dv01(t, irr) == pytest.approx(base_100.get_dv01(t, irr), rel=1e-14)
    # The caller's coupons keep their base.
    assert scaled.coupons[0].residual == base_100.coupons.coupons[0].residual * k
    assert scaled.coupons[0].interest == base_100.coupons.coupons[0].interest * k


def test_irr_from_amount_round_trips_in_the_default_or_a_given_convention():
    # A notional other than 100: the initial guess scales the TERA by notional/100,
    # or the Newton solve diverges.
    bond = clbond_factory.btp0470930()
    assert bond.notional != 100
    t = date(2026, 6, 24)
    amount = bond.get_amount_value(t, rate(0.0512))
    irr = bond.get_irr_from_amount(t, amount)
    assert irr.rate_convention == bond.irr_default_convention
    assert bond.get_amount_value(t, irr) == amount
    # The IRR comes back rounded to 6 decimals: start from a rate quoted that way.
    linear = bond.get_amount_value(t, rate(0.054576, lin_act360()))
    given = bond.get_irr_from_amount(t, linear, irr_rate_convention=lin_act360())
    assert given.rate_convention == lin_act360() and given.rate == 0.054576
    assert bond.get_amount_value(t, given) == linear


# ── Duration on a curve ─────────────────────────────────────────────────────


def _flat_continuous_curve(t: date, r: float):
    from fintoolsom.rates import ModelCurve

    class _Flat:
        def get_df(self, years):
            import numpy as np

            return np.exp(-r * np.asarray(years, dtype=float))

    return ModelCurve(t, _Flat(), date(t.year + 40, t.month, t.day))


def test_duration_on_a_flat_curve_is_the_duration_at_that_rate():
    bond = clbond_factory.btp0470930()
    t = date(2026, 6, 24)
    on_curve = bond.get_duration_from_curve(t, _flat_continuous_curve(t, 0.05))
    at_rate = bond.get_duration_from_rate(t, rate(0.05, exp_act365()))
    assert on_curve == pytest.approx(at_rate, rel=1e-12)
    assert 0 < on_curve < (bond.get_maturity_date() - t).days / 365


def test_duration_on_a_curve_is_the_df_weighted_average_time():
    bond = clbond_factory.btp0470930()
    t = date(2026, 6, 24)
    curve = curve_factory.sample_curve()  # curve date 2024-07-08, before t
    dates = bond.coupons.get_remaining_end_dates(t)
    pvs = bond.coupons.get_remaining_flows(t) * curve.get_dfs(dates)
    times = [(d - t).days / 365 for d in dates]
    expected = sum(p * x for p, x in zip(pvs, times)) / sum(pvs)
    assert bond.get_duration_from_curve(t, curve) == pytest.approx(expected, rel=1e-12)


def test_a_zero_coupon_duration_on_any_curve_is_its_maturity():
    t = date(2024, 7, 8)
    bond = clbond_factory.bullet_clbond(start=t, n_coupons=1, coupon_amount=0.0)
    maturity = (bond.get_maturity_date() - t).days / 365
    assert bond.get_duration_from_curve(
        t, curve_factory.sample_curve()
    ) == pytest.approx(maturity, rel=1e-12)


def test_a_matured_bond_has_no_duration_on_a_curve():
    bond = clbond_factory.two_coupon_bond()
    with pytest.raises(ZeroDivisionError):
        bond.get_duration_from_curve(date(2026, 1, 1), curve_factory.sample_curve())
