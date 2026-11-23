"""calibrate_from_instruments (bonds and deposits in one stack of flows) and the
`duration_weighted` fit of calibrate_from_bonds, with the previous fits frozen."""

from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.fixedIncome import Deposit
from fintoolsom.models import NelsonSiegel, NelsonSiegelSvensson
from fintoolsom.rates import ModelCurve, Rate
from tests.factories import clbond_factory
from tests.factories.rate_factory import comp_act365, lin_act360
from tests.unit.models import test_NelsonSiegelSvensson as nss_tests
from tests.unit.models.test_calibrate_many import TODAY, _bond_sets, _curves

TRUE_PARAMS = [0.050, -0.012, 0.010, 0.008, 1.2, 9.0]
DEPOSIT_DAYS = [30, 91, 182]


def _deposits(cal, noise_bps=0.0, payment=1_000_000, seed=11):
    """Deposits on the true NSS curve, quoted linear Act/360 (plus noise)."""
    rng = np.random.default_rng(seed)
    true = NelsonSiegelSvensson(*TRUE_PARAMS)
    out = []
    for days in DEPOSIT_DAYS:
        df = float(true.get_df(days / 365))
        rate = (1 / df - 1) / (days / 360) + rng.normal(0, noise_bps * 1e-4)
        out.append(
            (
                Deposit(f"D{days}", "clp", cal + timedelta(days=days), payment),
                Rate(lin_act360(), rate),
            )
        )
    return out


def _exact_bonds(cal):
    """Bonds priced on TRUE_PARAMS curve, quoted with no noise."""
    curve = ModelCurve(
        cal, NelsonSiegelSvensson(*TRUE_PARAMS), cal + timedelta(days=11500)
    )
    out = []
    for years in (1, 2, 3, 5, 7, 10, 15, 20):
        bond = clbond_factory.bullet_clbond(
            start=cal - timedelta(days=60), n_coupons=2 * years, coupon_amount=2.5
        )
        pv = bond.get_present_value_zc(cal, curve)
        irr = bond.get_irr_from_present_value(cal, pv, comp_act365())
        out.append((bond, irr))
    return out


# ── Nothing changes with duration_weighted=False: frozen results of the PV fit ──


def test_pv_fit_of_zero_coupon_bonds_is_unchanged():
    cal = date(2024, 7, 8)
    bonds = nss_tests._bonds_irr_list(cal)

    ns = NelsonSiegel()
    ns.calibrate_from_bonds(cal, bonds, duration_weighted=False)
    assert ns.get_params() == pytest.approx(
        [0.056455991826236805, -0.004511281523937836, -0.017966073310038695]
        + [1.5701242983407468],
        rel=1e-9,
    )
    nss = NelsonSiegelSvensson()
    nss.calibrate_from_bonds(cal, bonds, duration_weighted=False)
    assert nss.get_params() == pytest.approx(
        [0.056442308847306874, -0.004834684900272151, -0.023367873895444542]
        + [0.006481121431155725, 1.7032728465051705, 2.005533774202487],
        rel=1e-9,
    )


def test_pv_fit_of_a_coupon_bond_set_is_unchanged():
    cal, bonds = _bond_sets(6)[0]

    ns = NelsonSiegel()
    errors = ns.calibrate_from_bonds(cal, bonds, duration_weighted=False)
    assert ns.get_params() == pytest.approx(
        [0.050231025031479345, -0.005632101427648325, 0.013400705663684108]
        + [5.0841561371167066],
        rel=1e-9,
    )
    assert errors[0]["error_bps"] == pytest.approx(17.78816287, abs=1e-6)
    assert errors[-1]["error_bps"] == pytest.approx(0.25922858, abs=1e-6)

    nss = NelsonSiegelSvensson()
    errors = nss.calibrate_from_bonds(cal, bonds, duration_weighted=False)
    assert nss.get_params() == pytest.approx(
        [0.049846566357753516, -0.10623839565583557, 0.08077036938407403]
        + [0.008564233031889155, 0.25000000000000006, 7.48978374602868],
        rel=1e-9,
    )
    assert errors[0]["error_bps"] == pytest.approx(-0.27319652, abs=1e-6)
    assert errors[-1]["error_bps"] == pytest.approx(0.03965573, abs=1e-6)


def test_curve_fits_are_unchanged():
    curve = _curves(30, drop=3)[0]
    ns = NelsonSiegel()
    ns.calibrate_from_curve(curve)
    assert ns.get_params() == pytest.approx(
        [0.052604811708161, -0.014562869758362337, 0.013870848528693854]
        + [1.5909858033978304],
        rel=1e-9,
    )
    nss = NelsonSiegelSvensson()
    nss.calibrate_from_curve(curve)
    assert nss.get_params() == pytest.approx(
        [0.05100530153170468, -0.013345418268045695, 0.00947903111343018]
        + [0.007422349386784433, 1.138986863906447, 8.165924562743733],
        rel=1e-9,
    )


def test_many_curves_fits_are_unchanged():
    curves = _curves(30, drop=3)
    ns_models, _ = NelsonSiegel.calibrate_many_from_curves(curves)
    assert ns_models[0].get_params() == pytest.approx(
        [0.05260481185662811, -0.014562870122902681, 0.013870847009015835]
        + [1.5909856167177867],
        rel=1e-7,
    )
    assert ns_models[17].get_params() == pytest.approx(
        [0.05560494150017647, -0.014841164350820482, 0.008977102980176106]
        + [1.0711742929854386],
        rel=1e-7,
    )
    nss_models, _ = NelsonSiegelSvensson.calibrate_many_from_curves(curves)
    assert nss_models[0].get_params() == pytest.approx(
        [0.05100530216326932, -0.013345419078002298, 0.00947902692454519]
        + [0.007422348487606786, 1.1389866135956754, 8.165921997939538],
        rel=1e-7,
    )
    assert nss_models[17].get_params() == pytest.approx(
        [0.05474698666105921, -0.014014942577548485, -0.005048023602565079]
        + [0.004158623722796134, 0.4291710497318099, 3.731818431844358],
        rel=1e-7,
    )


# ── duration_weighted ──


def test_bonds_are_duration_weighted_by_default():
    cal, bonds = _bond_sets(6)[0]
    default, explicit, pv_fit = NelsonSiegel(), NelsonSiegel(), NelsonSiegel()
    default.calibrate_from_bonds(cal, bonds)
    explicit.calibrate_from_bonds(cal, bonds, duration_weighted=True)
    pv_fit.calibrate_from_bonds(cal, bonds, duration_weighted=False)
    assert default.get_params() == explicit.get_params()
    assert default.get_params() != pytest.approx(pv_fit.get_params(), rel=1e-3)


def test_duration_weighting_fits_irrs_better_than_pvs():
    # A one-hump model cannot follow the 10 bonds: the PV fit lets the short end
    # absorb the error in bp (17.8bp on the first bond), the IRR fit spreads it.
    cal, bonds = _bond_sets(6)[0]
    rms = {}
    for weighted in (True, False):
        errors = NelsonSiegel().calibrate_from_bonds(
            cal, bonds, duration_weighted=weighted
        )
        rms[weighted] = float(np.sqrt(np.mean([e["error_bps"] ** 2 for e in errors])))
    assert rms[True] < rms[False]


def test_calibrate_from_bonds_is_calibrate_from_instruments_with_bonds_only():
    cal, bonds = _bond_sets(6)[0]
    for weighted in (True, False):
        a, b = NelsonSiegelSvensson(), NelsonSiegelSvensson()
        errors_a = a.calibrate_from_bonds(cal, bonds, duration_weighted=weighted)
        errors_b = b.calibrate_from_instruments(
            cal, bonds_irr_list=bonds, duration_weighted=weighted
        )
        assert a.get_params() == b.get_params()
        assert errors_a == errors_b


# ── calibrate_from_instruments ──


@pytest.mark.parametrize("weighted", [True, False])
def test_bonds_and_deposits_recover_the_curve_they_were_priced_on(weighted):
    cal = TODAY
    model = NelsonSiegelSvensson()
    errors = model.calibrate_from_instruments(
        cal,
        bonds_irr_list=_exact_bonds(cal),
        deposits_irr_list=_deposits(cal),
        duration_weighted=weighted,
    )
    assert len(errors) == 8 + 3
    assert max(abs(e["error_bps"]) for e in errors) < 0.05
    ts = np.array([0.1, 0.5, 1, 3, 7, 12, 20])
    assert model.get_rate(ts) == pytest.approx(
        NelsonSiegelSvensson(*TRUE_PARAMS).get_rate(ts), abs=2e-4
    )


def test_errors_are_bonds_first_then_deposits_and_a_deposit_term_is_its_years():
    cal = TODAY
    bonds, deposits = _exact_bonds(cal), _deposits(cal)
    errors = NelsonSiegelSvensson().calibrate_from_instruments(
        cal, bonds_irr_list=bonds, deposits_irr_list=deposits
    )
    for error, days in zip(errors[len(bonds) :], DEPOSIT_DAYS):
        assert error["term"] == pytest.approx(days / 365)
    assert 0.5 < errors[0]["term"] < 1  # the 1y bond comes first


def test_a_deposit_payment_is_only_scale():
    cal = TODAY
    bonds = _exact_bonds(cal)
    fits = []
    for payment in (100, 1_000_000_000):
        model = NelsonSiegelSvensson()
        model.calibrate_from_instruments(
            cal,
            bonds_irr_list=bonds,
            deposits_irr_list=_deposits(cal, noise_bps=1.0, payment=payment),
        )
        fits.append(model.get_params())
    assert fits[0] == pytest.approx(fits[1], rel=1e-12)


def test_deposits_anchor_the_short_end_where_bonds_do_not_reach():
    cal = TODAY
    horizon = cal + timedelta(days=700)
    bonds = [
        pair for pair in _exact_bonds(cal) if pair[0].get_maturity_date() > horizon
    ]
    short_ts = np.array(DEPOSIT_DAYS) / 365
    truth = NelsonSiegelSvensson(*TRUE_PARAMS).get_rate(short_ts)
    only_bonds, with_deposits = NelsonSiegel(), NelsonSiegel()
    only_bonds.calibrate_from_instruments(cal, bonds_irr_list=bonds)
    with_deposits.calibrate_from_instruments(
        cal, bonds_irr_list=bonds, deposits_irr_list=_deposits(cal)
    )
    # Six NSS parameters cannot be fitted to five bonds, so this is a one-hump fit.
    err_bonds = np.abs(only_bonds.get_rate(short_ts) - truth).max()
    err_deposits = np.abs(with_deposits.get_rate(short_ts) - truth).max()
    assert err_deposits < err_bonds
    assert err_deposits < 5e-4


def test_deposits_only_calibration_fits_them():
    cal = TODAY
    errors = NelsonSiegel().calibrate_from_instruments(
        cal, deposits_irr_list=_deposits(cal)
    )
    assert max(abs(e["error_bps"]) for e in errors) < 0.5


def test_a_deposit_is_priced_in_the_convention_of_its_rate():
    # Linear Act/360 and compounded Act/365 quotes of the same discount factor
    # give the same fit.
    cal = TODAY
    bonds = _exact_bonds(cal)
    days = 91
    df = float(NelsonSiegelSvensson(*TRUE_PARAMS).get_df(days / 365))
    deposit = Deposit("D", "clp", cal + timedelta(days=days), 100)
    lin = Rate(lin_act360(), (1 / df - 1) / (days / 360))
    comp = Rate(comp_act365(), (1 / df) ** (365 / days) - 1)
    fits = []
    for quote in (lin, comp):
        model = NelsonSiegelSvensson()
        model.calibrate_from_instruments(
            cal, bonds_irr_list=bonds, deposits_irr_list=[(deposit, quote)]
        )
        fits.append(model.get_params())
    assert fits[0] == pytest.approx(fits[1], rel=1e-6)


# ── Validation ──


def test_a_matured_bond_raises():
    cal, bonds = _bond_sets(6)[0]
    bond, irr = bonds[0]
    late = bond.get_maturity_date() + timedelta(days=1)
    with pytest.raises(ValueError, match="no flows after"):
        NelsonSiegel().calibrate_from_bonds(late, [(bond, irr)])


@pytest.mark.parametrize("days", [0, -5])
def test_a_deposit_not_after_the_calibration_date_raises(days):
    cal = TODAY
    deposit = Deposit("OLD", "clp", cal + timedelta(days=days), 100)
    with pytest.raises(ValueError, match="OLD"):
        NelsonSiegel().calibrate_from_instruments(
            cal,
            bonds_irr_list=_exact_bonds(cal),
            deposits_irr_list=[(deposit, Rate(lin_act360(), 0.05))],
        )


def test_no_instruments_raises():
    with pytest.raises(ValueError, match="Nothing to calibrate"):
        NelsonSiegel().calibrate_from_instruments(TODAY)
    with pytest.raises(ValueError, match="Nothing to calibrate"):
        NelsonSiegel().calibrate_from_bonds(TODAY, [])
