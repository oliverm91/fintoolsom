"""calibrate_many_from_curves: many curves fitted in batched numpy."""

import sys
from datetime import date, timedelta

import numpy as np
import pytest

import fintoolsom.models.NelsonSiegelSvensson  # noqa: F401  (the module, for monkeypatch)
from fintoolsom.models import NelsonSiegel, NelsonSiegelSvensson
from fintoolsom.rates import InterpolationMethod, ModelCurve, PillarCurve, Rate
from tests.factories import clbond_factory
from tests.factories.rate_factory import comp_act365, exp_act365, lin_act360

TODAY = date(2026, 11, 19)
TENOR_DAYS = [30, 91, 182, 365, 547, 730, 1095, 1826, 2557, 3652, 5479, 7305, 10957]


def _curves(n: int, seed: int = 3, noise_bps: float = 1.0, drop: int = 0):
    """`n` daily curves ending today: standard tenors each moved by -3..+3 days,
    NSS rates drifting day to day plus noise; every other curve loses its last
    `drop` pillars, so curves differ in length too."""
    rng = np.random.default_rng(seed)
    params = np.array([0.050, -0.012, 0.010, 0.008, 1.2, 9.0])
    curves = []
    for k in range(n):
        curve_date = TODAY - timedelta(days=n - 1 - k)
        params = params + rng.normal(0, [4e-4, 4e-4, 6e-4, 6e-4, 0.03, 0.2])
        params[4], params[5] = np.clip(params[4], 0.4, 2.8), np.clip(params[5], 3, 18)
        days = np.array(TENOR_DAYS) + rng.integers(-3, 4, len(TENOR_DAYS))
        if drop and k % 2:
            days = days[:-drop]
        t = days / 365
        rates = NelsonSiegelSvensson(*params).get_rate(t)
        rates = rates + rng.normal(0, noise_bps * 1e-4, t.size)
        dates = [curve_date + timedelta(days=int(d)) for d in days]
        curves.append(
            PillarCurve.from_dfs(
                curve_date,
                dates,
                list(np.exp(-rates * t)),
                InterpolationMethod.PiecewiseConstant,
            )
        )
    return curves


def _rms(errors) -> float:
    return float(np.sqrt(np.mean([e["error_bps"] ** 2 for e in errors])))


def _sse(model, curve) -> float:
    t = np.array([(d - curve.curve_date).days for d in curve.knot_dates[1:]]) / 365
    rates = -np.log(curve.get_dfs(curve.knot_dates[1:])) / t
    return float(np.sum((model.get_rate(t) - rates) ** 2))


@pytest.mark.parametrize("cls", [NelsonSiegel, NelsonSiegelSvensson])
def test_own_humps_fit_each_curve_as_the_one_curve_calibration(cls):
    curves = _curves(30, drop=3)
    models, errors = cls.calibrate_many_from_curves(curves)
    assert len(models) == len(errors) == len(curves)
    for curve, model, errs in zip(curves, models, errors):
        single = cls()
        single_errors = single.calibrate_from_curve(curve)
        assert [e["term"] for e in errs] == [e["term"] for e in single_errors]
        # Never a worse fit; the same one where the single fit finds the minimum.
        assert _rms(errs) <= _rms(single_errors) + 1e-6
        assert np.allclose(
            [e["error_bps"] for e in errs],
            [e["error_bps"] for e in single_errors],
            atol=1e-3,
        )


@pytest.mark.parametrize("cls", [NelsonSiegel, NelsonSiegelSvensson])
def test_curves_of_different_lengths_and_pillars_are_padded_not_aligned(cls):
    curves = _curves(6, drop=4)
    assert len({len(c.knot_dates) for c in curves}) == 2
    models, errors = cls.calibrate_many_from_curves(curves)
    for curve, model, errs in zip(curves, models, errors):
        assert len(errs) == len(curve.knot_dates) - 1
        assert _rms(errs) == pytest.approx(
            float(np.sqrt(_sse(model, curve) / len(errs))) * 1e4, rel=1e-9
        )


def test_noise_free_curves_recover_their_models():
    curves = _curves(5, noise_bps=0.0)
    models, errors = NelsonSiegelSvensson.calibrate_many_from_curves(curves)
    assert max(_rms(e) for e in errors) < 1e-6


@pytest.mark.parametrize("cls", [NelsonSiegel, NelsonSiegelSvensson])
def test_global_humps_are_shared_and_minimise_the_total_error(cls):
    curves = _curves(20)
    models, errors = cls.calibrate_many_from_curves(curves, global_humps=True)
    n_betas = cls()._n_betas
    humps = {tuple(m.get_params()[n_betas:]) for m in models}
    assert len(humps) == 1
    total = sum(_sse(m, c) for m, c in zip(models, curves))
    # Any other shared humps fit worse in total: try each curve's own.
    own, _ = cls.calibrate_many_from_curves(curves)
    for other in own[:5]:
        shared, _ = cls.calibrate_many_from_curves(
            curves, global_humps=True, initial_guess=other.get_params()[n_betas:]
        )
        assert sum(_sse(m, c) for m, c in zip(shared, curves)) >= total * (1 - 1e-9)
    # Each curve fits worse (or equal) than with its own humps.
    assert sum(_sse(m, c) for m, c in zip(own, curves)) <= total


def test_one_curve_with_global_humps_is_the_one_curve_fit():
    curve = _curves(1)[0]
    (model,), (errors,) = NelsonSiegel.calibrate_many_from_curves(
        [curve], global_humps=True
    )
    single = NelsonSiegel()
    single_errors = single.calibrate_from_curve(curve)
    assert model.lambda_ == pytest.approx(single.lambda_, rel=1e-6)
    assert [e["error_bps"] for e in errors] == pytest.approx(
        [e["error_bps"] for e in single_errors], abs=1e-6
    )


def test_a_warm_start_reaches_the_same_fits():
    curves = _curves(10)
    cold, _ = NelsonSiegel.calibrate_many_from_curves(curves)
    warm, _ = NelsonSiegel.calibrate_many_from_curves(
        curves, initial_guess=[cold[0].lambda_]
    )
    for a, b in zip(cold, warm):
        assert a.lambda_ == pytest.approx(b.lambda_, rel=1e-5)


def test_no_curves_give_no_models():
    assert NelsonSiegelSvensson.calibrate_many_from_curves([]) == ([], [])


def test_the_models_are_independent_copies():
    models, _ = NelsonSiegel.calibrate_many_from_curves(_curves(3))
    models[0].b0 = 99.0
    assert models[1].b0 != 99.0


# ── Bonds ──

_MODULE = sys.modules["fintoolsom.models.NelsonSiegelSvensson"]
YEARS = (1, 2, 3, 5, 7, 10, 12, 15, 20, 30)


def _bond_sets(n, seed=5, noise_bps=1.0, convention=comp_act365, sizes=None):
    """`n` daily sets of bullet bonds (1-30y, semiannual) ending today, priced on a
    drifting NSS and quoted at their IRRs plus noise; set k keeps sizes[k] bonds."""
    rng = np.random.default_rng(seed)
    params = np.array([0.050, -0.012, 0.010, 0.008, 1.2, 9.0])
    sets = []
    for k in range(n):
        cal = TODAY - timedelta(days=n - 1 - k)
        params = params + rng.normal(0, [4e-4, 4e-4, 6e-4, 6e-4, 0.03, 0.2])
        params[4], params[5] = np.clip(params[4], 0.4, 2.8), np.clip(params[5], 3, 18)
        curve = ModelCurve(
            cal, NelsonSiegelSvensson(*params), cal + timedelta(days=11500)
        )
        bonds = []
        for years in YEARS[: (sizes[k] if sizes else len(YEARS))]:
            bond = clbond_factory.bullet_clbond(
                start=cal - timedelta(days=int(rng.integers(0, 180))),
                n_coupons=2 * years,
                coupon_amount=2.5,
            )
            pv = bond.get_present_value_zc(cal, curve)
            irr = bond.get_irr_from_present_value(cal, pv, convention())
            quote = irr.rate + rng.normal(0, noise_bps * 1e-4)
            bonds.append((bond, Rate(irr.rate_convention, quote)))
        sets.append((cal, bonds))
    return sets


def _single(cls, sets):
    out = []
    for cal, bonds in sets:
        model = cls()
        out.append(
            (model, model.calibrate_from_bonds(cal, bonds, duration_weighted=False))
        )
    return out


@pytest.mark.parametrize("convention", [comp_act365, lin_act360, exp_act365])
def test_bond_sets_fit_as_the_one_set_calibration_in_any_quote_convention(
    convention,
):
    sets = _bond_sets(8, convention=convention)
    models, errors = NelsonSiegel.calibrate_many_from_bonds(sets)
    for (single, single_errors), model, errs in zip(
        _single(NelsonSiegel, sets), models, errors
    ):
        assert model.get_params() == pytest.approx(single.get_params(), rel=1e-5)
        assert [e["term"] for e in errs] == pytest.approx(
            [e["term"] for e in single_errors], rel=1e-7
        )
        assert [e["error_bps"] for e in errs] == pytest.approx(
            [e["error_bps"] for e in single_errors], abs=1e-3
        )


def test_noise_free_bond_sets_recover_their_model():
    sets = _bond_sets(4, noise_bps=0.0)
    _, errors = NelsonSiegelSvensson.calibrate_many_from_bonds(sets)
    assert max(_rms(e) for e in errors) < 1e-4


def test_nss_bond_sets_fit_on_average_at_least_as_well_as_the_loop():
    sets = _bond_sets(20)
    _, errors = NelsonSiegelSvensson.calibrate_many_from_bonds(sets)
    single = [_rms(e) for _, e in _single(NelsonSiegelSvensson, sets)]
    assert np.mean([_rms(e) for e in errors]) <= np.mean(single) + 1e-6


def test_bond_sets_may_differ_in_size_and_date():
    sets = _bond_sets(4, sizes=[10, 7, 10, 5])
    models, errors = NelsonSiegel.calibrate_many_from_bonds(sets)
    assert [len(e) for e in errors] == [10, 7, 10, 5]
    for (single, single_errors), errs in zip(_single(NelsonSiegel, sets), errors):
        assert [e["error_bps"] for e in errs] == pytest.approx(
            [e["error_bps"] for e in single_errors], abs=1e-3
        )


def test_a_set_not_converged_is_refitted_alone(monkeypatch):
    sets = _bond_sets(3)
    monkeypatch.setattr(_MODULE, "_MAX_BOND_ITERATIONS", 1)
    models, _ = NelsonSiegel.calibrate_many_from_bonds(sets)
    monkeypatch.undo()
    for (single, _), model in zip(_single(NelsonSiegel, sets), models):
        assert model.get_params() == pytest.approx(single.get_params(), rel=1e-5)


def test_a_warm_start_reaches_the_same_bond_fits():
    sets = _bond_sets(5)
    cold, _ = NelsonSiegel.calibrate_many_from_bonds(sets)
    warm, _ = NelsonSiegel.calibrate_many_from_bonds(
        sets, initial_guess=cold[0].get_params()
    )
    for a, b in zip(cold, warm):
        assert a.get_params() == pytest.approx(b.get_params(), rel=1e-5)


def test_no_bond_sets_give_no_models():
    assert NelsonSiegel.calibrate_many_from_bonds([]) == ([], [])
