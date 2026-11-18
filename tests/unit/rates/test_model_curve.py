"""ModelCurve: a Curve backed by a term-structure model."""

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.market import Currency, Market, OvernightRateIndex
from fintoolsom.models import NelsonSiegel, NelsonSiegelSvensson
from fintoolsom.rates import (
    Curve,
    ExponentialInterestConvention,
    ModelCurve,
    RateConvention,
    TermStructureModel,
)
from tests.factories import clbond_factory

T = date(2024, 7, 8)
HORIZON = T + timedelta(days=365 * 30)
DATES = [T + timedelta(days=n) for n in (0, 1, 30, 365, 1830, 3650, 7300, 12000)]
YEARS = np.array([(d - T).days / 365 for d in DATES])
CONTINUOUS = RateConvention(
    ExponentialInterestConvention, ActualDayCountConvention, 365
)


def _nss() -> NelsonSiegelSvensson:
    return NelsonSiegelSvensson(0.05, -0.01, 0.02, 0.01, 1.5, 12.0)


@dataclass
class _Flat:
    """The simplest model: a flat continuous rate, not a Nelson-Siegel one."""

    rate: float

    def get_df(self, t):
        return np.exp(-self.rate * np.asarray(t, dtype=float))


def test_model_curve_is_a_concrete_curve():
    assert issubclass(ModelCurve, Curve) and not ModelCurve.__abstractmethods__


def test_any_object_with_get_df_is_a_term_structure_model():
    assert isinstance(NelsonSiegel(), TermStructureModel)
    assert isinstance(NelsonSiegelSvensson(), TermStructureModel)
    assert isinstance(_Flat(0.05), TermStructureModel)
    with pytest.raises(TypeError, match="must have get_df"):
        ModelCurve(T, object(), HORIZON)  # type: ignore[arg-type]


def test_dfs_and_zero_rates_are_the_models():
    nss = _nss()
    curve = ModelCurve(T, nss, HORIZON)
    np.testing.assert_allclose(curve.get_dfs(DATES), nss.get_df(YEARS), rtol=1e-15)
    assert curve.get_df(T) == 1.0
    for d, t in zip(DATES[1:], YEARS[1:]):
        assert curve.get_zero_rate(d, CONTINUOUS).rate == pytest.approx(
            nss.get_rate(t),
            abs=1e-12,  # a 1-day rate amplifies df rounding 365x
        )


def test_dfs_at_ordinals_match_get_dfs():
    curve = ModelCurve(T, _nss(), HORIZON, spread=0.001)
    ordinals = np.array([d.toordinal() for d in DATES], dtype=np.int64)
    np.testing.assert_allclose(
        curve._dfs_at_ordinals(ordinals), curve.get_dfs(DATES), rtol=1e-15
    )


def test_the_curve_keeps_its_own_copy_of_the_model():
    nss = _nss()
    curve = ModelCurve(T, nss, HORIZON)
    before = curve.get_dfs(DATES)
    nss.b0 = 0.09  # e.g. a later recalibration
    np.testing.assert_array_equal(curve.get_dfs(DATES), before)
    assert curve.model is not nss and curve.model == _nss()
    assert curve.copy().model is not curve.model


def test_last_date_is_the_horizon_and_the_model_extrapolates_past_it():
    curve = ModelCurve(T, _nss(), HORIZON)
    assert curve.last_date == HORIZON
    beyond = HORIZON + timedelta(days=3650)
    assert curve.get_df(beyond) == pytest.approx(
        float(_nss().get_df((beyond - T).days / 365)), rel=1e-15
    )


def test_a_parallel_bump_shifts_every_zero_rate_and_leaves_the_original():
    curve = ModelCurve(T, _nss(), HORIZON)
    bumped = curve.parallel_bump_bps(1).parallel_bump_bps(2)
    assert bumped.spread == pytest.approx(3e-4, abs=1e-18) and curve.spread == 0.0
    for d in DATES[1:]:
        shift = (
            bumped.get_zero_rate(d, CONTINUOUS).rate
            - curve.get_zero_rate(d, CONTINUOUS).rate
        )
        assert shift == pytest.approx(3e-4, abs=1e-13)


@pytest.mark.parametrize("spread", [0.0, 0.0025])
def test_aging_divides_by_the_df_at_the_new_date(spread):
    curve = ModelCurve(T, _nss(), HORIZON, spread=spread)
    aged_on = T + timedelta(days=400)
    aged = curve.get_aged_curve(aged_on)
    assert (aged.curve_date, aged.model_date, aged.last_date) == (aged_on, T, HORIZON)
    later = [d for d in DATES if d > aged_on]
    np.testing.assert_allclose(
        aged.get_dfs(later),
        curve.get_dfs(later) / curve.get_df(aged_on),
        rtol=1e-13,
    )
    assert aged.get_df(aged_on) == pytest.approx(1.0, abs=1e-15)
    # Aging and bumping commute: the spread runs from the curve date.
    np.testing.assert_allclose(
        aged.parallel_bump_bps(5).get_dfs(later),
        curve.parallel_bump_bps(5).get_aged_curve(aged_on).get_dfs(later),
        rtol=1e-13,
    )


def test_aging_outside_the_curve_raises():
    curve = ModelCurve(T, _nss(), HORIZON)
    with pytest.raises(ValueError, match="posterior to curve date"):
        curve.get_aged_curve(T)
    with pytest.raises(ValueError, match="previous to the curve's last date"):
        curve.get_aged_curve(HORIZON)


def test_invalid_dates_are_rejected():
    with pytest.raises(ValueError, match="horizon"):
        ModelCurve(T, _nss(), T)
    with pytest.raises(ValueError, match="model_date"):
        ModelCurve(T, _nss(), HORIZON, model_date=T + timedelta(days=1))


def test_a_flat_model_curve_in_a_market_discounts_and_projects():
    icp = OvernightRateIndex("ICP", currency=Currency.CLP)
    market = Market(
        T,
        risk_free_index=icp,
        curves={
            icp: ModelCurve(T, _Flat(0.05), HORIZON),
            Currency.USD: ModelCurve(T, NelsonSiegel(0.04, -0.01, 0.0, 2.0), HORIZON),
        },
    )
    start, end = DATES[2], DATES[3]
    assert market.get_df(Currency.CLP, end) == pytest.approx(
        math.exp(-0.05 * YEARS[3]), rel=1e-15
    )
    assert market.get_accrual(icp, start, end) == pytest.approx(
        math.exp(0.05 * (end - start).days / 365) - 1, rel=1e-13
    )
    assert market.get_dfs(Currency.USD, DATES[1:4]) == pytest.approx(
        NelsonSiegel(0.04, -0.01, 0.0, 2.0).get_df(YEARS[1:4]), rel=1e-15
    )


def test_a_bond_values_and_finds_its_z_spread_on_a_model_curve():
    bond = clbond_factory.two_coupon_bond()
    curve = ModelCurve(T, _nss(), HORIZON)
    flows = bond.coupons.get_remaining_flows(T)
    ts = np.array([(d - T).days / 365 for d in bond.coupons.get_remaining_end_dates(T)])
    assert bond.get_present_value_zc(T, curve) == pytest.approx(
        float(np.sum(flows * _nss().get_df(ts))), rel=1e-14
    )
    irr = bond.get_irr_with_zcc(T, curve.parallel_bump_bps(25), CONTINUOUS)
    assert bond.get_z_spread(T, irr, curve) == pytest.approx(25, abs=1e-6)
