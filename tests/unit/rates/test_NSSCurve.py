import math
from datetime import date, timedelta
import numpy as np
import pytest
from fintoolsom.models import NelsonSiegelSvensson
from fintoolsom.rates import (
    Curve,
    InterpolationMethod,
    NSSCurve,
    ZeroCouponCurve,
)



T0 = date(2024, 7, 8)


NSS_PARAMS = (0.05, -0.01, 0.01, 0.005, 1.5, 0.3)


D = [T0 + timedelta(days=n) for n in (10, 95, 400, 1100, 2500)]  # inside the pillar range


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
    "nss": lambda: NSSCurve(T0, *NSS_PARAMS),
}


@pytest.fixture(params=list(BACKENDS), ids=list(BACKENDS))
def curve(request) -> Curve:
    return BACKENDS[request.param]()


def test_nss_df_is_one_at_curve_date():
    assert NSSCurve(T0, *NSS_PARAMS).get_df(T0) == 1.0


def test_nss_curve_matches_model_closed_form_exactly():
    curve = NSSCurve(T0, *NSS_PARAMS)
    t = np.array([(d - T0).days for d in D]) / 365
    assert curve.get_dfs(D) == pytest.approx(NelsonSiegelSvensson._get_df(t, *NSS_PARAMS), abs=1e-15)


def test_nss_model_and_curve_share_one_zero_rate_formula():
    model = NelsonSiegelSvensson()
    model.b0, model.b1, model.b2, model.b3, model.lambda_, model.mu_ = NSS_PARAMS
    curve = NSSCurve(T0, *NSS_PARAMS)
    t = np.array([0.0] + [(d - T0).days / 365 for d in D])
    # Same zero rates (curve rates derived from its dfs, model rates from the formula).
    curve_rates = [curve.get_zero_rate(d).rate for d in D]
    assert curve_rates == pytest.approx(model.get_rate(t[1:]), abs=1e-14)
    # Rate is linear in the betas through the factor loadings used by calibration.
    assert model.get_rate(t) == pytest.approx(NelsonSiegelSvensson._get_design_matrix(t, *NSS_PARAMS[4:]) @ NSS_PARAMS[:4], abs=1e-15)
    # t=0 limits: short rate b0 + b1, df = 1 (no 0/0 NaN in the model either).
    assert model.get_rate(0.0) == pytest.approx(NSS_PARAMS[0] + NSS_PARAMS[1], abs=1e-15)
    assert model.get_df(0.0) == 1.0 and curve.get_df(T0) == 1.0


def test_nss_parallel_bump_is_exact_in_continuous_rates_even_when_aged():
    curve = NSSCurve(T0, *NSS_PARAMS).get_aged_curve(D[0])
    before = [curve.get_zero_rate(d).rate for d in D[1:]]
    curve.parallel_bump_rates_bps(50)
    after = [curve.get_zero_rate(d).rate for d in D[1:]]
    assert np.array(after) - np.array(before) == pytest.approx(np.full(len(before), 0.005), abs=1e-12)
