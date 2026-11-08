import inspect
import math
from copy import copy
from datetime import date, timedelta
import numpy as np
import pytest
from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.rates import (
    CompoundedInterestConvention,
    ExponentialInterestConvention,
    InterpolationMethod,
    LinearInterestConvention,
    LegacyCurve,
    Rate,
    RateConvention,
    ZeroCouponCurve,
)



T0 = date(2024, 7, 8)



COMPOUNDED = RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)


LINEAR_360 = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)


CONTINUOUS = RateConvention(ExponentialInterestConvention, ActualDayCountConvention, 365)


D = [T0 + timedelta(days=n) for n in (10, 95, 400, 1100, 2500)]  # inside the pillar range


STARTS, ENDS = D[:-1], D[1:]


def _zcc(method: InterpolationMethod) -> ZeroCouponCurve:
    pillars = [T0 + timedelta(days=n) for n in (1, 30, 182, 365, 730, 1825, 3650)]
    return ZeroCouponCurve(
        T0,
        date_dfs=[(p, math.exp(-0.05 * (p - T0).days / 365)) for p in pillars],
        df_interpolation_method=method,
    )


BACKENDS = {
    "zcc_loglinear": lambda: _zcc(InterpolationMethod.LogLinear),
    "zcc_hermite": lambda: _zcc(InterpolationMethod.HermiteCubicSpline),
}


@pytest.fixture(params=list(BACKENDS), ids=list(BACKENDS))
def curve(request) -> LegacyCurve:
    return BACKENDS[request.param]()


EXPECTED_INTERFACE = {
    "get_df", "get_dfs", "get_df_fwd", "get_dfs_fwds",
    "get_wf", "get_wfs", "get_wf_fwd", "get_wfs_fwds",
    "get_zero_rate", "get_forward_rates", "get_forward_rates_values",
    "get_accrued_interest", "get_accrued_interests",
    "parallel_bump_rates_bps", "get_aged_curve", "copy", "__copy__",
}


def test_every_curve_query_is_abstract():
    assert LegacyCurve.__abstractmethods__ == EXPECTED_INTERFACE


def test_backend_missing_a_method_cannot_be_instantiated():
    namespace = {name: (lambda self, *a, **k: None) for name in EXPECTED_INTERFACE - {"get_aged_curve"}}
    Incomplete = type("Incomplete", (LegacyCurve,), namespace)
    with pytest.raises(TypeError, match="get_aged_curve"):
        Incomplete()


def test_zero_coupon_curve_implements_the_whole_interface():
    assert issubclass(ZeroCouponCurve, LegacyCurve) and not inspect.isabstract(ZeroCouponCurve)


def test_scalar_and_vector_queries_agree(curve):
    dfs = curve.get_dfs(D)
    assert [curve.get_df(d) for d in D] == pytest.approx(list(dfs), abs=1e-15)
    assert curve.get_wfs(D) == pytest.approx(1 / dfs, abs=1e-15)
    assert curve.get_wf(D[2]) == pytest.approx(1 / dfs[2], abs=1e-15)
    assert np.all(np.diff(dfs) < 0)  # positive rates: dfs decrease


def test_forward_factors_are_ratios_of_spot_factors(curve):
    ratio = curve.get_dfs(ENDS) / curve.get_dfs(STARTS)
    assert curve.get_dfs_fwds(STARTS, ENDS) == pytest.approx(ratio, abs=1e-15)
    assert curve.get_wfs_fwds(STARTS, ENDS) == pytest.approx(1 / ratio, abs=1e-12)
    assert curve.get_df_fwd(STARTS[0], ENDS[0]) == pytest.approx(ratio[0], abs=1e-15)
    assert curve.get_wf_fwd(STARTS[0], ENDS[0]) == pytest.approx(1 / ratio[0], abs=1e-12)
    with pytest.raises(ValueError):
        curve.get_dfs_fwds(STARTS, ENDS[:-1])


def test_accrued_interest_is_notional_times_forward_accrual(curve):
    wf = curve.get_wfs_fwds(STARTS, ENDS)
    assert curve.get_accrued_interests(np.full(len(STARTS), 100.0), STARTS, ENDS) == pytest.approx(100 * (wf - 1))
    assert curve.get_accrued_interest(100.0, STARTS[0], ENDS[0]) == pytest.approx(100 * (wf[0] - 1))


@pytest.mark.parametrize("convention", [COMPOUNDED, LINEAR_360, CONTINUOUS], ids=["comp", "lin360", "cont"])
def test_forward_rates_reproduce_forward_wealth_factors(curve, convention):
    rates = curve.get_forward_rates(STARTS, ENDS, convention)
    assert isinstance(rates, list) and all(isinstance(r, Rate) for r in rates)
    for r, s, e, wf in zip(rates, STARTS, ENDS, curve.get_wfs_fwds(STARTS, ENDS)):
        assert r.get_wealth_factor(s, e) == pytest.approx(wf, abs=1e-12)

    single = curve.get_forward_rates(STARTS[0], ENDS[0], convention)
    assert isinstance(single, Rate)
    assert single.rate == pytest.approx(rates[0].rate, abs=1e-14)
    values = curve.get_forward_rates_values(STARTS, ENDS, convention)
    assert values == pytest.approx([r.rate for r in rates], abs=1e-14)


@pytest.mark.parametrize("convention", [COMPOUNDED, LINEAR_360, CONTINUOUS], ids=["comp", "lin360", "cont"])
def test_zero_rate_reproduces_discount_factor(curve, convention):
    for d in D:
        rate = curve.get_zero_rate(d, convention)
        assert rate.get_discount_factor(T0, d) == pytest.approx(curve.get_df(d), abs=1e-12)


def test_aged_curve_discounts_with_forward_factors(curve):
    age = D[1]
    aged = curve.get_aged_curve(age)
    assert aged.curve_date == age
    later = D[2:]
    assert aged.get_dfs(later) == pytest.approx(curve.get_dfs_fwds([age] * len(later), later), abs=1e-10)
    assert curve.curve_date == T0  # original untouched
    with pytest.raises(ValueError):
        curve.get_aged_curve(T0)


def test_copy_is_equal_and_independent(curve):
    before = curve.get_dfs(D)
    clone = copy(curve)
    assert type(clone) is type(curve)
    assert clone.get_dfs(D) == pytest.approx(before, abs=1e-15)  # same backend settings
    clone.parallel_bump_rates_bps(50)
    assert curve.get_dfs(D) == pytest.approx(before, abs=1e-15)
    assert np.all(clone.get_dfs(D) < before)


def test_parallel_bump_raises_rates_and_lowers_dfs(curve):
    before = curve.get_dfs(D)
    curve.parallel_bump_rates_bps(25)
    assert np.all(curve.get_dfs(D) < before)
    curve.parallel_bump_rates_bps(-25)
    assert curve.get_dfs(D) == pytest.approx(before, abs=1e-12)
