"""``fintoolsom._native.bullet_leg``: a bullet leg's schedule, dates and arrays in one
C++ call (``SwapLeg._bullet_parts``), bit for bit the legs the Python path builds."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.derivatives.swaps import builders, legs
from fintoolsom.market import Currency
from tests.factories.swap_factory import (
    FIXED_RATES,
    NOTIONALS,
    PAYMENT_FREQUENCY,
    SOFR,
    SPREAD,
    START_DATES,
    SWAP_TERM_YEARS,
    TERM_SOFR_3M,
    Ccy,
    SwapStatus,
    _rate,
    adj_convention,
)

_ADJ = adj_convention(Ccy.USD)


def _common(status: SwapStatus) -> dict:
    return dict(
        notional=NOTIONALS[Ccy.USD],
        start_date=START_DATES[status],
        term=Term(SWAP_TERM_YEARS, TermUnit.Y, _ADJ),
        frequency=PAYMENT_FREQUENCY,
        adj_convention=_ADJ,
    )


def _fixed(**kw: Any):
    return builders.fixed_leg(
        rate=_rate(FIXED_RATES[Ccy.USD]), currency=Currency.USD, **kw
    )


def _overnight(**kw: Any):
    return builders.overnight_leg(index=SOFR, spread=SPREAD, **kw)


def _overnight_no_spread(**kw: Any):
    return builders.overnight_leg(index=SOFR, **kw)


def _xccy(**kw: Any):
    return builders.xccy_floating_leg(index=SOFR, spread=SPREAD, **kw)


def _term(**kw: Any):
    return builders.term_rate_leg(index=TERM_SOFR_3M, **kw)


MAKERS = [_fixed, _overnight, _overnight_no_spread, _xccy, _term]
ARRAYS = (
    "residuals",
    "amortizations",
    "fixed_interest",
    "fixed_flows",
    "_start_ordinals",
    "_end_ordinals",
    "_payment_ordinals",
    "_fx_fixing_ordinals",
)


def _same_leg(a, b) -> None:
    assert type(a) is type(b)
    for name in vars(b):
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(y, np.ndarray):
            assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), name
            assert x.flags.writeable == y.flags.writeable, name
        elif name in ("spread", "rate"):
            assert x.rate == y.rate and str(x.rate_convention) == str(y.rate_convention)
        else:
            assert type(x) is type(y) and x == y, name
    for x, y in ((a, b),):  # which date lists and ordinals are one object
        assert (x.payment_dates is x.end_dates) == (y.payment_dates is y.end_dates)
        assert (x.fx_fixing_dates is x.payment_dates) == (
            y.fx_fixing_dates is y.payment_dates
        )
        assert (x._payment_ordinals is x._end_ordinals) == (
            y._payment_ordinals is y._end_ordinals
        )


def _python_leg(monkeypatch, make, kw):
    with monkeypatch.context() as m:
        m.setattr(legs, "_native_schedule_args", lambda *a: None)
        return make(**kw)


def _spy(monkeypatch) -> list:
    calls = []
    native = settings.native()
    real = native.bullet_leg

    def spy(*args):
        out = real(*args)
        calls.append(out is not None and out[5][6] is not None)  # interest in C++
        return out

    monkeypatch.setattr(native, "bullet_leg", spy)
    return calls


@pytest.mark.parametrize("make", MAKERS)
@pytest.mark.parametrize("status", [SwapStatus.NOT_STARTED, SwapStatus.STARTED])
@pytest.mark.parametrize("lags", [(0, 0), (2, 0), (0, 1), (2, 1)])
@pytest.mark.parametrize("rates", ["cpp", "python"])
def test_a_cpp_bullet_leg_is_the_python_one(monkeypatch, make, status, lags, rates):
    payment_lag, fx_fixing_lag = lags
    kw = dict(_common(status), payment_lag=payment_lag, fx_fixing_lag=fx_fixing_lag)
    with settings.use_backend("cpp"), settings.use_backend(rates, "rates"):
        calls = _spy(monkeypatch)
        leg = make(**kw)
        assert calls == [rates == "cpp"]  # interest left to Python on Python rates
        _same_leg(leg, _python_leg(monkeypatch, make, kw))


@pytest.mark.parametrize("make", MAKERS)
def test_replace_rebuilds_a_cpp_leg(monkeypatch, make):
    kw = _common(SwapStatus.NOT_STARTED)
    with settings.use_backend("cpp"):
        leg = make(**kw)
        halved = replace(leg, residuals=leg.residuals / 2)
        expected = replace(
            _python_leg(monkeypatch, make, kw), residuals=leg.residuals / 2
        )
    _same_leg(halved, expected)
    assert halved.fixed_flows[-1] == pytest.approx(leg.fixed_flows[-1] / 2)


@pytest.mark.parametrize("backend", ["dates", "derivatives"])
def test_the_python_path_runs_without_both_cpp_backends(monkeypatch, backend):
    with settings.use_backend("cpp"), settings.use_backend("python", backend):
        calls = _spy(monkeypatch)
        _fixed(**_common(SwapStatus.NOT_STARTED))
    assert calls == []


def test_a_negative_lag_still_raises_the_python_error():
    with settings.use_backend("cpp"), pytest.raises(ValueError, match="payment_lag"):
        _fixed(**_common(SwapStatus.NOT_STARTED), payment_lag=-1)
