"""Coupon, Coupons and CLBond: validation, copies, accessors and round trips."""

from copy import copy
from datetime import date

import pytest

from fintoolsom.fixedIncome import CLBond, Coupon, Coupons
from tests.factories import clbond_factory, curve_factory
from tests.factories.rate_factory import comp_act365, lin_30e360, rate

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


def test_coupons_copy_is_independent():
    coupons = clbond_factory.two_coupon_coupons()
    duplicate = copy(coupons)
    duplicate.adjust_to_notional(1_000)
    assert (
        coupons.coupons[-1].amortization == 100
        and duplicate.coupons[-1].amortization == 1_000
    )


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
    curve = curve_factory.sample_zero_coupon_curve()
    t = curve.curve_date
    irr = bond.get_irr_with_zcc(t, curve, comp_act365())
    assert bond.get_present_value(t, irr) == pytest.approx(
        bond.get_present_value_zc(t, curve), rel=1e-9
    )


def test_duration_is_below_maturity_and_dv01_matches_its_formula():
    bond = clbond_factory.btp0470930()
    t, irr = date(2026, 6, 24), rate(0.05)
    duration = bond.get_duration(t, irr)
    assert 0 < duration < (bond.get_maturity_date() - t).days / 365
    expected = -bond.get_present_value(t, irr) * duration / 10_000 * bond.notional / 100
    assert bond.get_dv01(t, irr) == pytest.approx(expected)


def test_dv01_approximates_the_value_change_for_one_basis_point():
    bond = clbond_factory.btp0470930()
    t = date(2026, 6, 24)
    up = bond.get_present_value(t, rate(0.0501)) - bond.get_present_value(t, rate(0.05))
    # Duration is the Macaulay one, so the bp move is ~ dv01 / (1 + y).
    assert up * bond.notional / 100 == pytest.approx(
        bond.get_dv01(t, rate(0.05)) / 1.05, rel=1e-3
    )


def test_irr_from_amount_uses_the_default_irr_convention():
    bond = clbond_factory.btp0470930()
    t = date(2026, 6, 24)
    amount = bond.get_amount_value(t, rate(0.0512))
    irr = bond.get_irr_from_amount(t, amount)
    assert irr.rate_convention == bond.irr_default_convention
    assert bond.get_amount_value(t, irr) == amount
