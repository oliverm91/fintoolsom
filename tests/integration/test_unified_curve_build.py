"""One scheduler for discount and projection curves, bootstrapped date by date.

The real 2026-06-29 quotes plus synthetic USD term-index quotes (all SOFR
collateralised): fixed vs TERM3M IRS pin the TERM3M projection curve, and TERM6M vs
TERM3M + 8bp basis swaps then pin TERM6M on top of it. Projection-only indexes are
solved by the same scheduler as the discount curves.
"""

from datetime import timedelta

import numpy as np
import pytest

from fintoolsom.curve_builder import _solver, build_curves
from fintoolsom.curve_builder import InsufficientQuotesError
from fintoolsom.curve_builder.builder import _CurveBuilder
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.market import CrossCurrencyFloatFloatQuote, Currency, ForwardPointsQuote
from fintoolsom.rates import IndexCurve, InterpolationMethod
from tests.factories.quote_factory import (
    QUOTE_DATE,
    TERM3M,
    TERM6M,
    load_quotes,
    term_basis_quotes,
    term_irs_quotes,
    term_market,
)

CLP = Currency.CLP
GRID = [
    QUOTE_DATE + timedelta(days=d) for d in (1, 91, 365, 730, 1826, 3652, 7305, 10957)
]
# 1M-1Y XCCY quotes are redundant with the forward points (see test_curve_build_and_valuation).
REDUNDANT_XCCY_TERMS = {"1M", "3M", "6M", "9M", "1Y"}
REPRICE_TOL = 1e-6  # CLP, on quote notionals of 100 USD (90,000 CLP)


@pytest.fixture(scope="module")
def real():
    return load_quotes()


@pytest.fixture(scope="module")
def term_irs(real):
    return term_irs_quotes(real)


@pytest.fixture(scope="module")
def basis(term_irs):
    return term_basis_quotes(term_irs)


@pytest.fixture(scope="module")
def full(real, term_irs, basis):
    market = term_market()
    build_curves(real + term_irs + basis, market)
    return market


def _build(quotes, **kwargs):
    market = term_market()
    build_curves(quotes, market, **kwargs)
    return market


def _mtm(quote, market) -> float:
    fx_dependent = isinstance(quote, (CrossCurrencyFloatFloatQuote, ForwardPointsQuote))
    instrument = (
        quote.get_instrument(market) if fx_dependent else quote.get_instrument()
    )
    return Calculator.valuate(instrument, market, CLP)


def _exact(quotes):
    return [
        q
        for q in quotes
        if not (
            isinstance(q, CrossCurrencyFloatFloatQuote)
            and str(q.term) in REDUNDANT_XCCY_TERMS
        )
    ]


def _dfs(market) -> dict:
    return {key: market.curves[key].get_dfs(GRID) for key in market.curves}


def _accruals(market) -> dict:
    index_curves = {k: c for k, c in market.curves.items() if isinstance(c, IndexCurve)}
    return {k: c.get_accrual(GRID[:-1], GRID[1:]) for k, c in index_curves.items()}


def _assert_same_curves(a, b, rel):
    assert set(a.curves) == set(b.curves)
    for key, dfs in _dfs(a).items():
        np.testing.assert_allclose(dfs, _dfs(b)[key], rtol=rel)
    for index, accruals in _accruals(a).items():
        np.testing.assert_allclose(accruals, _accruals(b)[index], rtol=rel)


def _spy_block_sizes(monkeypatch) -> list[int]:
    """Number of unknowns of every root solve the builder runs."""
    sizes: list[int] = []
    solve = _solver._solve_or_rollback

    def spy(residual, y0, restore, what):
        sizes.append(len(y0))
        return solve(residual, y0, restore, what)

    monkeypatch.setattr(_solver, "_solve_or_rollback", spy)
    return sizes


# ── Projection curves in the shared scheduler ───────────────────────────────


def test_term_index_gets_a_projection_curve_that_reprices_its_swaps(real, term_irs):
    market = _build(real + term_irs)
    curve = market.get_index_curve(TERM3M)
    assert isinstance(curve, IndexCurve) and curve.spot_rate is None
    assert len(curve.forward_rates) == 1 + len(
        term_irs
    )  # the spot anchor + one segment per swap
    assert max(abs(_mtm(q, market)) for q in term_irs) < REPRICE_TOL


def test_basis_swaps_pin_a_second_projection_curve_on_top_of_the_first(
    full, term_irs, basis
):
    assert {TERM3M.name, TERM6M.name} <= set(full.curves)
    assert max(abs(_mtm(q, full)) for q in term_irs + basis) < REPRICE_TOL
    # Receiving 6M against 3M + 8bp at par: 6M forwards sit ~8bp above the 3M ones.
    for a, b in ((365, 730), (730, 1826), (1826, 3652), (3652, 7305)):
        start, end = QUOTE_DATE + timedelta(days=a), QUOTE_DATE + timedelta(days=b)
        f3 = full.get_index_curve(TERM3M).get_forward_rate(start, end).rate
        f6 = full.get_index_curve(TERM6M).get_forward_rate(start, end).rate
        assert (f6 - f3) * 1e4 == pytest.approx(8.0, abs=0.5)


def test_basis_swaps_do_not_move_the_curves_their_irs_pin(real, term_irs, full):
    alone = _build(real + term_irs)
    np.testing.assert_allclose(
        _accruals(full)[TERM3M.name], _accruals(alone)[TERM3M.name], rtol=1e-9
    )
    for key, dfs in _dfs(alone).items():
        np.testing.assert_allclose(_dfs(full)[key], dfs, rtol=1e-12)


def test_basis_swaps_alone_cannot_pin_two_projection_curves(real, basis):
    # 13 basis swaps for 26 unknown segments (TERM3M and TERM6M): under-determined.
    with pytest.raises(InsufficientQuotesError):
        _build(real + basis)


def test_quote_order_does_not_change_the_curves(real, term_irs, basis, full):
    # Term quotes first: their groups must wait for the SOFR curve they are
    # collateralised in, although they give it no pillar.
    _assert_same_curves(_build(basis + term_irs + real), full, rel=1e-9)


def test_existing_discount_curves_keep_the_real_quotes_repriced(real, full):
    assert max(abs(_mtm(q, full)) for q in _exact(real)) < REPRICE_TOL


# ── Date-by-date bootstrap ──────────────────────────────────────────────────


def test_date_by_date_bootstrap_matches_the_joint_solve(
    real, term_irs, basis, full, monkeypatch
):
    monkeypatch.setattr(_CurveBuilder, "date_by_date", False)
    _assert_same_curves(_build(real + term_irs + basis), full, rel=1e-8)


def test_local_interpolation_solves_one_pillar_date_at_a_time(real, monkeypatch):
    sizes = _spy_block_sizes(monkeypatch)
    _build(real)
    # One small solve per pillar date (at most one unknown per curve of the group).
    assert len(sizes) > 50 and max(sizes) <= 2


def test_joint_solve_runs_one_root_per_group(real, monkeypatch):
    monkeypatch.setattr(_CurveBuilder, "date_by_date", False)
    sizes = _spy_block_sizes(monkeypatch)
    _build(real)
    assert len(sizes) == 4 and sum(sizes) > 50  # ICP, SOFR, and the two USD groups


def test_non_local_interpolation_skips_the_sweep_and_still_reprices(real, monkeypatch):
    sizes = _spy_block_sizes(monkeypatch)
    market = _build(real, discount_interp_method=InterpolationMethod.HermiteCubicSpline)
    # A spline pillar moves the whole discount curve: the two USD groups solve jointly,
    # while the index curves (piecewise-constant forwards) still sweep date by date.
    big = [n for n in sizes if n > 2]
    assert len(big) == 2 and len(sizes) > len(big)
    assert max(abs(_mtm(q, market)) for q in _exact(real)) < REPRICE_TOL
