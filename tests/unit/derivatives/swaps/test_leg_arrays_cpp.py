"""``fintoolsom._native.leg_arrays``: a C++-backend leg's arrays equal the Python
backend's, and inputs it does not take fall back to the Python code."""

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.derivatives.swaps import FixedLeg, OvernightLeg
from fintoolsom.market import Currency, CurrencyName
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.rates import Rate, RateConvention

USD = Currency(CurrencyName.USD)
SOFR = OvernightRateIndex("SOFR", currency=USD)
RATE = Rate(RateConvention(), 0.04)
ARRAYS = (
    "_start_ordinals",
    "_end_ordinals",
    "_payment_ordinals",
    "_fx_fixing_ordinals",
    "residuals",
    "amortizations",
    "fixed_interest",
    "fixed_flows",
    "_par_schedule",
)


def _legs(**fields):
    out = {}
    for backend in ("python", "cpp"):
        with settings.use_backend(backend, "derivatives"):
            out[backend] = OvernightLeg(**fields, index=SOFR, spread=RATE)
    return out["python"], out["cpp"]


def _fields(rng, n, lag=0, fx_lag=0, gap=False):
    starts = [
        date(2026, 1, 5) + timedelta(int(d))
        for d in np.cumsum(rng.integers(20, 200, n))
    ]
    ends = [s + timedelta(int(rng.integers(1, 40))) for s in starts[1:]]
    ends.append(starts[-1] + timedelta(30))
    if not gap:
        ends = [*starts[1:], starts[-1] + timedelta(30)]
    fields = {
        "start_dates": starts,
        "end_dates": ends,
        "residuals": rng.normal(1e6, 3e5, n),
    }
    if lag:
        fields["payment_dates"] = [e + timedelta(lag) for e in ends]
    if fx_lag:
        pay = fields.get("payment_dates", ends)
        fields["fx_fixing_dates"] = [p - timedelta(fx_lag) for p in pay]
    return fields


@pytest.mark.parametrize("n", [1, 2, 7, 40])
@pytest.mark.parametrize(
    "lag, fx_lag, gap", [(0, 0, False), (2, 0, False), (0, 1, False), (2, 3, True)]
)
def test_cpp_arrays_equal_python(n, lag, fx_lag, gap):
    py, cpp = _legs(**_fields(np.random.default_rng(n + 10 * lag), n, lag, fx_lag, gap))
    assert py._native is False and cpp._native is True
    for name in ARRAYS:
        np.testing.assert_array_equal(
            getattr(cpp, name), getattr(py, name), err_msg=name
        )
    assert cpp._par_eligible == py._par_eligible == (not lag and not gap)
    for name in ARRAYS[:-1]:
        assert not getattr(cpp, name).flags.writeable
    # Defaults share the end (payment) ordinals, as on the Python backend.
    assert (cpp._payment_ordinals is cpp._end_ordinals) == (not lag)


def test_an_empty_leg_on_both_backends():
    py, cpp = _legs(start_dates=[], end_dates=[], residuals=[])
    assert cpp.n_coupons == py.n_coupons == 0
    assert not cpp._par_schedule and not py._par_schedule


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"end_dates": [date(2026, 1, 5), date(2026, 4, 6)]}, "Coupon 0: a start date"),
        (
            {"payment_dates": [date(2026, 4, 6), date(2026, 7, 1)]},
            "Coupon 1: an end date",
        ),
        ({"fx_fixing_dates": [date(2026, 4, 6), date(2026, 7, 9)]}, "Coupon 1: an FX"),
        (
            {
                "start_dates": [date(2026, 4, 1), date(2026, 1, 5)],
                "end_dates": [date(2026, 7, 6)] * 2,
            },
            "schedule order",
        ),
        ({"residuals": [1.0]}, "one value per coupon"),
    ],
)
def test_invalid_legs_raise_the_python_errors_on_cpp(fields, match):
    base = {
        "start_dates": [date(2026, 1, 5), date(2026, 4, 6)],
        "end_dates": [date(2026, 4, 6), date(2026, 7, 6)],
        "residuals": [1.0, 1.0],
        "rate": RATE,
        "currency": USD,
    }
    with (
        settings.use_backend("cpp", "derivatives"),
        pytest.raises(ValueError, match=match),
    ):
        FixedLeg(**{**base, **fields})


def test_dates_that_are_not_plain_fall_back_to_python():
    # datetimes are date subclasses: the C++ call declines, the Python path converts.
    fields = {
        "start_dates": [datetime(2026, 1, 5), datetime(2026, 4, 6)],
        "end_dates": [datetime(2026, 4, 6), datetime(2026, 7, 6)],
        "residuals": [2.0, 1.0],
    }
    native = settings.native()
    assert (
        native.leg_arrays(
            fields["start_dates"], fields["end_dates"], None, None, np.array([2.0, 1.0])
        )
        is None
    )
    py, cpp = _legs(**fields)
    for name in ARRAYS:
        np.testing.assert_array_equal(
            getattr(cpp, name), getattr(py, name), err_msg=name
        )
