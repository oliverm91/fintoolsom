"""Par valuation of floating legs discounted on their own index curve.

The shortcut (``_par_leg_pv``) must give the coupon-by-coupon PV (``_flows_pv``), and
be taken only when it applies. The legs are built on the Python backend, whose
functions these tests spy on; test_cpp_swap_parity checks the C++ one against it.
"""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.calculator import _simplified_swap_pv, _swap_pv
from fintoolsom.derivatives.swaps import (
    FixedLeg,
    FloatingLeg,
    NonStartedSimplifiedSwap,
    SimplifiedSwap,
    builders,
)
from tests.factories.swap_factory import (
    CAMARA,
    SOFR,
    SPREAD,
    VAL_DATE,
    Ccy,
    SwapStatus,
    adj_convention,
    build_market,
    build_swap,
    swap_cases,
)

ADJ = adj_convention(Ccy.CLP)
N = 950_000_000.0


@pytest.fixture(scope="module")
def market():
    return build_market()


@pytest.fixture
def par_calls(monkeypatch):
    """Count the calls to the par shortcut, wherever it is reached from."""
    calls = []
    par = _swap_pv._par_leg_pv

    def counting(leg, market):
        calls.append(leg)
        return par(leg, market)

    monkeypatch.setattr(_swap_pv, "_par_leg_pv", counting)
    monkeypatch.setattr(_simplified_swap_pv, "_par_leg_pv", counting)
    return calls


def _python():
    return settings.use_backend("python", "derivatives")


def _leg(start=date(2024, 7, 15), index=CAMARA, **kwargs):
    with _python():
        return _build_leg(start, index, **kwargs)


def _build_leg(start, index, **kwargs):
    return builders.overnight_leg(
        notional=N,
        start_date=start,
        term=Term(2, TermUnit.Y, ADJ),
        frequency="QUARTERLY",
        adj_convention=ADJ,
        index=index,
        **kwargs,
    )


def _amortizing(leg, amortizations):
    """`leg` with its notional repaid by `amortizations` (summing to it)."""
    residuals = N - np.concatenate(([0.0], np.cumsum(amortizations)[:-1]))
    coupons = [
        replace(c, residual=float(r), amortization=float(a))
        for c, r, a in zip(leg.coupons, residuals, amortizations)
    ]
    with _python():
        return replace(leg, coupons=coupons)


# ── Eligibility, decided at construction ───────────────────────────────────


def test_a_bullet_leg_without_lag_is_par_eligible():
    assert _leg()._par_eligible
    assert _leg(spread=SPREAD)._par_eligible


def test_a_payment_lag_makes_a_leg_not_par_eligible():
    assert not _leg(payment_lag=2)._par_eligible


def test_an_amortizing_leg_repaying_its_notional_is_par_eligible():
    leg = _leg()
    assert _amortizing(leg, np.full(leg.n_coupons, N / leg.n_coupons))._par_eligible


def test_a_leg_not_repaying_its_notional_is_not_par_eligible():
    leg = _leg()
    coupons = [*leg.coupons[:-1], replace(leg.coupons[-1], amortization=N / 2)]
    with _python():
        assert not replace(leg, coupons=coupons)._par_eligible


def test_a_gap_in_the_schedule_makes_a_leg_not_par_eligible():
    leg = _leg()
    gap = replace(leg.coupons[1], start_date=leg.coupons[0].end_date.replace(day=1))
    coupons = [leg.coupons[0], gap, *leg.coupons[2:]]
    with _python():
        assert not replace(leg, coupons=coupons)._par_eligible


def test_the_spread_flag():
    assert not _leg()._has_spread
    assert _leg(spread=SPREAD)._has_spread


# ── Regular swaps ──────────────────────────────────────────────────────────


def _own_currency_legs():
    """Every leg of the factory swaps settled in its own currency."""
    out = []
    for case in swap_cases():
        with _python():
            swap = build_swap(case)
        for side, leg in (("receive", swap.receive_leg), ("pay", swap.pay_leg)):
            if swap.leg_settlement_currency(leg) == leg.currency:
                out.append(
                    pytest.param(leg, swap.collateral_index, id=f"{case.id}-{side}")
                )
    return out


@pytest.mark.parametrize("leg, collateral", _own_currency_legs())
def test_leg_pv_matches_the_flows_pv(leg, collateral, market):
    expected = _swap_pv._flows_pv(leg, collateral, market)
    assert _swap_pv._leg_pv(leg, collateral, market) == pytest.approx(
        expected, rel=1e-12, abs=1e-6
    )


def _par_expected(leg, collateral, market) -> bool:
    return (
        isinstance(leg, FloatingLeg)
        and leg._par_eligible
        and _swap_pv._nothing_realised(leg, market.t.toordinal())
        and _swap_pv._discounted_on_index_curve(leg, collateral, market)
    )


def test_the_factory_swaps_reach_the_par_shortcut(market, par_calls):
    legs = [p.values for p in _own_currency_legs()]
    expected = [leg for leg, coll in legs if _par_expected(leg, coll, market)]
    for leg, coll in legs:
        _swap_pv._leg_pv(leg, coll, market)
    assert expected and par_calls == expected


@pytest.mark.parametrize(
    "collateral, index, expected",
    [
        (CAMARA, CAMARA, True),  # CSA on the leg index
        (SOFR, CAMARA, False),  # CSA on another index
        (None, CAMARA, False),  # uncollateralised: ICP is the CLP funding curve
    ],
)
def test_par_needs_the_discount_curve_to_be_the_index_curve(
    collateral, index, expected, market, par_calls
):
    leg = _leg(index=index, spread=SPREAD)
    pv = _swap_pv._leg_pv(leg, collateral, market)
    assert bool(par_calls) is expected
    assert pv == pytest.approx(
        _swap_pv._flows_pv(leg, collateral, market), rel=1e-12, abs=1e-6
    )


def test_a_started_leg_is_valued_coupon_by_coupon(market, par_calls):
    leg = _leg(start=date(2024, 1, 10))
    assert leg._start_ordinals[0] <= VAL_DATE.toordinal()
    _swap_pv._leg_pv(leg, CAMARA, market)
    assert not par_calls


def test_a_lagged_leg_is_valued_coupon_by_coupon(market, par_calls):
    _swap_pv._leg_pv(_leg(payment_lag=2), CAMARA, market)
    assert not par_calls


def test_an_amortizing_leg_at_par(market, par_calls):
    leg = _leg(spread=SPREAD)
    leg = _amortizing(leg, np.full(leg.n_coupons, N / leg.n_coupons))
    pv = _swap_pv._leg_pv(leg, CAMARA, market)
    assert par_calls
    assert pv == pytest.approx(
        _swap_pv._flows_pv(leg, CAMARA, market), rel=1e-12, abs=1e-6
    )


def test_a_zero_spread_leg_is_worth_its_notional_discounted_from_its_start(market):
    leg = _leg()
    df = market.get_index_curve(CAMARA).get_df(leg.start_dates[0])
    assert _swap_pv._leg_pv(leg, CAMARA, market) == pytest.approx(N * df, rel=1e-15)


# ── Simplified swaps: eligibility assumed, not checked ─────────────────────


def _simplified_legs(status=None):
    out = []
    for case in swap_cases():
        if case.uses_uf or (status is not None and case.status is not status):
            continue
        with _python():
            swap = build_swap(case)
            s = SimplifiedSwap(
                receive_leg=swap.receive_leg,
                pay_leg=swap.pay_leg,
                collateral_index=swap.collateral_index,
            )
        for side, leg in (("receive", s.receive_leg), ("pay", s.pay_leg)):
            out.append(pytest.param(leg, s.collateral_index, id=f"{case.id}-{side}"))
    return out


@pytest.mark.parametrize("leg, collateral", _simplified_legs())
def test_simplified_leg_pv_matches_the_flows_pv(leg, collateral, market):
    expected = _swap_pv._flows_pv(leg, collateral, market)
    assert _simplified_swap_pv._simplified_leg_pv(
        leg, collateral, market
    ) == pytest.approx(expected, rel=1e-12, abs=1e-6)


@pytest.mark.parametrize("leg, collateral", _simplified_legs(SwapStatus.NOT_STARTED))
def test_non_started_leg_pv_matches_the_flows_pv(leg, collateral, market):
    expected = _swap_pv._flows_pv(leg, collateral, market)
    assert _simplified_swap_pv._non_started_leg_pv(
        leg, collateral, market
    ) == pytest.approx(expected, rel=1e-12, abs=1e-6)


def test_simplified_valuation_does_not_read_the_eligibility_flag(market, par_calls):
    # A lagged leg is not eligible, but a SimplifiedSwap moves its payments to the
    # end dates and is valued at par without asking the leg.
    lagged = _leg(payment_lag=2)
    with _python():
        fixed = _fixed_leg(lagged)
    assert isinstance(fixed, FixedLeg)
    for cls in (SimplifiedSwap, NonStartedSimplifiedSwap):
        with _python():
            swap = cls(receive_leg=lagged, pay_leg=fixed, collateral_index=CAMARA)
        leg = swap.receive_leg
        object.__setattr__(leg, "_par_eligible", False)
        par_calls.clear()
        if cls is SimplifiedSwap:
            pv = _simplified_swap_pv._simplified_leg_pv(leg, CAMARA, market)
        else:
            pv = _simplified_swap_pv._non_started_leg_pv(leg, CAMARA, market)
        assert par_calls == [leg]
        assert pv == pytest.approx(
            _swap_pv._flows_pv(leg, CAMARA, market), rel=1e-12, abs=1e-6
        )


def _fixed_leg(like):
    return builders.fixed_leg(
        notional=N,
        start_date=date(2024, 7, 15),
        term=Term(2, TermUnit.Y, ADJ),
        frequency="QUARTERLY",
        adj_convention=ADJ,
        rate=SPREAD,
        currency=like.currency,
    )
