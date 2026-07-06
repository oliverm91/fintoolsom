"""Unit tests for the ProjectionCurve forward-rate curve and its discount-view alias.

Focused on the Phase-1/2 seam: piecewise-constant projection must be numerically
identical to a LogLinear ZeroCouponCurve built from the same pseudo-DFs, the
scalar/vector accrual API, the equivalent-forward inversion, input validation, and
Market.get_projection's fallback to a DiscountProjectionView.
"""
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.rates import (
    ProjectionCurve,
    ProjectionInterpolationMethod,
    DiscountProjectionView,
    ZeroCouponCurve,
    InterpolationMethod,
    RateConvention,
    ExponentialInterestConvention,
)
from fintoolsom.market import Market
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.market.currencies import Currency, CurrencyName


T = date(2026, 6, 29)
KNOTS = [T, T + timedelta(days=90), T + timedelta(days=180), T + timedelta(days=365), T + timedelta(days=730)]
FWDS = np.array([0.055, 0.052, 0.050, 0.045])  # continuously compounded, act/365


def _pc() -> ProjectionCurve:
    return ProjectionCurve(T, KNOTS, FWDS, ProjectionInterpolationMethod.PiecewiseConstant)


def _equivalent_loglinear_zcc(pc: ProjectionCurve) -> ZeroCouponCurve:
    date_pdfs = [(k, pc.pseudo_df(k)) for k in KNOTS[1:]]
    return ZeroCouponCurve(T, date_dfs=date_pdfs, df_interpolation_method=InterpolationMethod.LogLinear)


def test_pseudo_df_matches_loglinear_zcc_at_knots():
    pc = _pc()
    zcc = _equivalent_loglinear_zcc(pc)
    for k in KNOTS[1:]:
        assert pc.pseudo_df(k) == pytest.approx(zcc.get_df(k), abs=1e-12)


def test_pseudo_df_anchor_is_one():
    assert _pc().pseudo_df(KNOTS[0]) == pytest.approx(1.0, abs=1e-15)


def test_accrual_matches_loglinear_zcc_between_knots():
    pc = _pc()
    zcc = _equivalent_loglinear_zcc(pc)
    for d0 in (30, 100, 200, 400):
        for d1 in (d0 + 45, d0 + 120, d0 + 250):
            s, e = T + timedelta(days=d0), T + timedelta(days=d1)
            wf_z = zcc.get_wf_fwd(s, e)
            assert pc.get_wealth_factor(s, e) == pytest.approx(wf_z, abs=1e-12)
            assert pc.get_accrual(s, e) == pytest.approx(wf_z - 1, abs=1e-12)


def test_accrual_vectorized_matches_scalar_and_zcc():
    pc = _pc()
    zcc = _equivalent_loglinear_zcc(pc)
    starts = [T + timedelta(days=d) for d in (10, 100, 200)]
    ends = [T + timedelta(days=d) for d in (100, 200, 400)]
    vec = pc.get_accrual(starts, ends)
    assert isinstance(vec, np.ndarray)
    z = zcc.get_wfs_fwds(starts, ends) - 1
    assert np.max(np.abs(vec - z)) == pytest.approx(0.0, abs=1e-12)
    # vector == scalar element-wise
    for i, (s, e) in enumerate(zip(starts, ends)):
        assert vec[i] == pytest.approx(pc.get_accrual(s, e), abs=1e-15)


def test_equivalent_forward_rate_recovers_segment_forward():
    # A single piecewise-constant segment, queried under exp/act365, returns the stored forward.
    pc = _pc()
    rc = RateConvention(interest_convention=ExponentialInterestConvention, time_fraction_base=365)
    r = pc.get_equivalent_forward_rate(KNOTS[1], KNOTS[2], rc)
    assert r.value == pytest.approx(0.052, abs=1e-12)


def test_flat_extrapolation_beyond_last_knot():
    # Beyond the last knot the last segment forward is held flat.
    pc = _pc()
    e1 = KNOTS[-1] + timedelta(days=100)
    e2 = KNOTS[-1] + timedelta(days=200)
    rc = RateConvention(interest_convention=ExponentialInterestConvention, time_fraction_base=365)
    r = pc.get_equivalent_forward_rate(e1, e2, rc)
    assert r.value == pytest.approx(FWDS[-1], abs=1e-12)


def test_validation_errors():
    with pytest.raises(ValueError):  # too few knots
        ProjectionCurve(T, [T], np.array([]))
    with pytest.raises(ValueError):  # wrong number of forwards
        ProjectionCurve(T, KNOTS, np.array([0.05, 0.05]))
    with pytest.raises(ValueError):  # non-increasing knots
        ProjectionCurve(T, [T, T, T + timedelta(days=10)], np.array([0.05, 0.05]))
    with pytest.raises(NotImplementedError):
        ProjectionCurve(
            T, KNOTS, FWDS, interpolation_method="not-a-method",  # type: ignore[arg-type]
        )


def test_discount_projection_view_delegates_to_curve():
    pc = _pc()
    zcc = _equivalent_loglinear_zcc(pc)
    view = DiscountProjectionView(zcc)
    s, e = T + timedelta(days=50), T + timedelta(days=300)
    assert view.pseudo_df(e) == pytest.approx(zcc.get_df(e), abs=1e-15)
    assert view.get_wealth_factor(s, e) == pytest.approx(zcc.get_wf_fwd(s, e), abs=1e-15)
    assert view.get_accrual(s, e) == pytest.approx(zcc.get_wf_fwd(s, e) - 1, abs=1e-15)
    starts = [T + timedelta(days=d) for d in (10, 100)]
    ends = [T + timedelta(days=d) for d in (100, 300)]
    assert np.max(np.abs(view.get_accrual(starts, ends) - (zcc.get_wfs_fwds(starts, ends) - 1))) == pytest.approx(0.0, abs=1e-15)


def test_market_get_projection_falls_back_to_discount_view():
    usd = Currency(CurrencyName.USD)
    sofr = OvernightRateIndex("SOFR", currency=usd)
    pc = _pc()
    zcc = _equivalent_loglinear_zcc(pc)
    market = Market(T)
    market.curves[(sofr, usd)] = zcc

    # No projection curve registered -> fallback view over the (index, index.currency) discount curve.
    proj = market.get_projection(sofr)
    assert isinstance(proj, DiscountProjectionView)
    assert proj.discount_curve is zcc

    # After set_projection, the stored curve is returned as-built.
    market.set_projection(sofr, pc)
    assert market.get_projection(sofr) is pc
