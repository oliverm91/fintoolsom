"""Unit tests for build_curve_from_instruments (CLBond and Deposit bootstrapping)."""

import warnings
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.curve_builder import (
    InsufficientQuotesError,
    build_curve_from_instruments,
)
from fintoolsom.fixedIncome import Deposit
from fintoolsom.models import NelsonSiegelSvensson
from fintoolsom.rates import InterpolationMethod, ModelCurve, PillarCurve, Rate
from tests.factories import clbond_factory
from tests.factories.rate_factory import comp_act365, lin_act360

TODAY = date(2024, 7, 8)
TRUE_NSS = NelsonSiegelSvensson(0.050, -0.012, 0.010, 0.008, 1.2, 9.0)


def _reference_curve(curve_date: date = TODAY) -> ModelCurve:
    return ModelCurve(curve_date, TRUE_NSS, curve_date + timedelta(days=11500))


def _make_deposits(
    curve_date: date = TODAY,
    days_list: tuple[int, ...] = (30, 91, 182),
    curve=None,
) -> list[tuple[Deposit, Rate]]:
    curve = curve or _reference_curve(curve_date)
    out = []
    for days in days_list:
        pay_date = curve_date + timedelta(days=days)
        df = curve.get_df(pay_date)
        rate_val = (1.0 / df - 1.0) / (days / 360.0)
        out.append(
            (
                Deposit(f"D{days}", "clp", pay_date, 1_000_000.0),
                Rate(lin_act360(), rate_val),
            )
        )
    return out


def _make_bonds(
    curve_date: date = TODAY,
    years_list: tuple[int, ...] = (1, 2, 3, 5, 10),
    curve=None,
) -> list[tuple]:
    curve = curve or _reference_curve(curve_date)
    out = []
    for years in years_list:
        bond = clbond_factory.bullet_clbond(
            start=curve_date - timedelta(days=60),
            n_coupons=2 * years,
            coupon_amount=2.5,
        )
        pv = bond.get_present_value_zc(curve_date, curve)
        irr = bond.get_irr_from_present_value(curve_date, pv, comp_act365())
        out.append((bond, irr))
    return out


def _assert_reprices(
    curve_date: date,
    curve: PillarCurve,
    instruments: list[tuple],
    tol: float = 1e-8,
) -> None:
    for inst, irr in instruments:
        if isinstance(inst, Deposit):
            expected_df = float(irr.get_discount_factor(curve_date, inst.payment_date))
            assert curve.get_df(inst.payment_date) == pytest.approx(
                expected_df, abs=tol
            )
        else:
            expected_pv = inst.get_present_value(curve_date, irr)
            actual_pv = inst.get_present_value_zc(curve_date, curve)
            assert actual_pv == pytest.approx(expected_pv, abs=tol)


def test_deposits_only_exact_dfs_and_precalculated():
    deposits = _make_deposits()
    curve = build_curve_from_instruments(TODAY, deposits)

    assert isinstance(curve, PillarCurve)
    assert curve.interpolation_method is InterpolationMethod.PiecewiseConstant
    assert curve.curve_date == TODAY
    assert curve.knot_dates == [TODAY, *(d.payment_date for d, _ in deposits)]
    assert curve.precalculate is True
    assert curve._grid is not None
    _assert_reprices(TODAY, curve, deposits, tol=1e-12)


def test_bonds_only_recovers_piecewise_constant_pillar_curve():
    bonds = [
        clbond_factory.bullet_clbond(
            start=TODAY - timedelta(days=60), n_coupons=2 * y, coupon_amount=2.5
        )
        for y in (1, 2, 3, 5, 10)
    ]
    knots = [TODAY, *(b.get_maturity_date() for b in bonds)]
    true_forwards = np.array([0.042, 0.046, 0.051, 0.055, 0.058])
    true_curve = PillarCurve(
        TODAY, knots, true_forwards, InterpolationMethod.PiecewiseConstant
    )

    quoted = [
        (
            b,
            b.get_irr_from_present_value(
                TODAY, b.get_present_value_zc(TODAY, true_curve), comp_act365()
            ),
        )
        for b in bonds
    ]
    built = build_curve_from_instruments(TODAY, quoted)

    assert built.knot_dates == knots
    assert built.forward_rates == pytest.approx(true_forwards, abs=1e-10)
    _assert_reprices(TODAY, built, quoted, tol=1e-10)


def test_mixed_deposits_and_bonds_reprice_and_order_independent():
    instruments = _make_deposits() + _make_bonds()
    curve = build_curve_from_instruments(TODAY, instruments)
    _assert_reprices(TODAY, curve, instruments, tol=1e-10)

    shuffled = list(reversed(instruments))
    curve_shuffled = build_curve_from_instruments(TODAY, shuffled)
    assert curve_shuffled.knot_dates == curve.knot_dates
    assert curve_shuffled.forward_rates == pytest.approx(curve.forward_rates, abs=1e-12)


@pytest.mark.parametrize(
    "method",
    [
        InterpolationMethod.PiecewiseConstant,
        InterpolationMethod.PiecewiseLinear,
        InterpolationMethod.CubicSpline,
        InterpolationMethod.HermiteCubicSpline,
        InterpolationMethod.ConvexMonotone,
    ],
)
def test_all_interpolation_methods_reprice(method):
    instruments = _make_deposits() + _make_bonds()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        curve = build_curve_from_instruments(TODAY, instruments, interp_method=method)

    assert curve.interpolation_method is method
    assert curve.precalculate is True
    assert curve._grid is not None
    _assert_reprices(TODAY, curve, instruments, tol=1e-8)


def test_insufficient_instruments_raises():
    deposits = _make_deposits(days_list=(30,))
    with pytest.raises(InsufficientQuotesError):
        build_curve_from_instruments(TODAY, [])
    with pytest.raises(InsufficientQuotesError):
        build_curve_from_instruments(TODAY, deposits)


def test_duplicate_maturities_raises():
    d1 = Deposit("D1", "clp", TODAY + timedelta(days=90), 100.0)
    d2 = Deposit("D2", "clp", TODAY + timedelta(days=90), 200.0)
    r = Rate(lin_act360(), 0.05)
    with pytest.raises(ValueError, match="distinct maturity dates"):
        build_curve_from_instruments(TODAY, [(d1, r), (d2, r)])


def test_matured_instruments_raise():
    d_ok = Deposit("D90", "clp", TODAY + timedelta(days=90), 100.0)
    d_past = Deposit("D0", "clp", TODAY, 100.0)
    r = Rate(lin_act360(), 0.05)
    with pytest.raises(ValueError, match="on or before"):
        build_curve_from_instruments(TODAY, [(d_past, r), (d_ok, r)])

    matured_bond = clbond_factory.bullet_clbond(
        start=date(2020, 1, 1), n_coupons=4, coupon_amount=2.5
    )
    with pytest.raises(ValueError, match="no flows after"):
        build_curve_from_instruments(
            TODAY, [(matured_bond, Rate(comp_act365(), 0.05)), (d_ok, r)]
        )


def test_unsupported_instrument_type_raises():
    d_ok = Deposit("D90", "clp", TODAY + timedelta(days=90), 100.0)
    r = Rate(lin_act360(), 0.05)
    with pytest.raises(TypeError, match="Expected CLBond or Deposit"):
        build_curve_from_instruments(TODAY, [("not_an_instrument", r), (d_ok, r)])  # type: ignore[list-item]


@pytest.mark.parametrize("bad_tol", [0.0, -1e-6, float("nan"), float("inf")])
def test_invalid_reprice_tol_raises(bad_tol):
    deposits = _make_deposits(days_list=(30, 91))
    with pytest.raises(ValueError, match="reprice_tol"):
        build_curve_from_instruments(TODAY, deposits, reprice_tol=bad_tol)


def test_inconsistent_quotes_negative_remaining_pv_raises():
    # Two-coupon bond whose first coupon is at 6M and maturity at 1Y, paired with a
    # 6M deposit at 0% and a bond priced absurdly low (remaining PV <= 0).
    bond = clbond_factory.bullet_clbond(start=TODAY, n_coupons=2, coupon_amount=50.0)
    first_coupon_date = bond.coupons.get_remaining_end_dates(TODAY)[0]
    dep = Deposit("D1", "clp", first_coupon_date, 100.0)
    with pytest.raises(ValueError, match="non-positive"):
        build_curve_from_instruments(
            TODAY,
            [
                (dep, Rate(lin_act360(), 0.0)),
                (bond, Rate(comp_act365(), 10.0)),
            ],
        )
