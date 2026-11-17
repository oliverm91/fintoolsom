"""The C++ parts of the C++ curve builder's planning and seeding:
``_native.maturity_tables`` (every pillar date of an instrument, read once per build)
and ``_native.fixed_leg_wealth`` (a fixed leg's seed), each against its Python code."""

import datetime as dt

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.curve_builder import _keys, _seeds, build_curves
from fintoolsom.curve_builder._keys import _instrument_of, _maturities, _maturity_tables
from fintoolsom.derivatives.swaps import FixedLeg, Swap
from fintoolsom.market import Currency
from fintoolsom.market.market import Market
from fintoolsom.rates import InterpolationMethod
from tests.factories import quote_factory
from tests.unit.curve_builder.test_keys_non_deliverable import _fixed_float_quote


def _instruments() -> tuple[list, Market]:
    market = quote_factory.quote_market()
    quotes = quote_factory.load_quotes()
    instruments = [_instrument_of(q, market) for q in quotes]
    instruments.append(
        _fixed_float_quote(
            is_deliverable=False, payment_currency=Currency.CLP, fx_fixing_lag=2
        ).get_instrument()
    )
    return instruments, market


def test_the_tables_are_the_python_ones():
    with settings.use_backend("cpp"):
        instruments, market = _instruments()
        rfi = market.risk_free_index
        tables = settings.native().maturity_tables(
            instruments,
            np.array([_keys._table_kind(i) for i in instruments], dtype=np.int64),
            rfi,
        )
        kinds = {type(i).__name__ for i in instruments}
        assert {"Swap", "NDF"} <= kinds or {"Swap", "Forward"} <= kinds
        for instrument, table in zip(instruments, tables):
            assert table is not None
            assert table == _maturities(instrument, rfi)
            assert type(table[1]) is dt.date


def test_an_instrument_cpp_does_not_take_goes_to_python():
    with settings.use_backend("cpp"):
        instruments, market = _instruments()
        with pytest.raises(TypeError, match="only supports Swap and Forward"):
            _maturity_tables([*instruments, object()], market.risk_free_index)


@pytest.mark.parametrize("builder", ["cpp", "python"])
def test_one_table_call_per_build_on_the_cpp_builder(monkeypatch, builder):
    calls = []
    native = settings.native()
    real = native.maturity_tables

    def spy(*args):
        calls.append(len(args[0]))
        return real(*args)

    monkeypatch.setattr(native, "maturity_tables", spy)
    with settings.use_backend("cpp"), settings.use_backend(builder, "curve_builder"):
        build_curves(
            quote_factory.load_quotes(),
            quote_factory.quote_market(),
            interp_method=InterpolationMethod.PiecewiseConstant,
        )
    assert calls == ([len(quote_factory.load_quotes())] if builder == "cpp" else [])


def _python_zero(monkeypatch, seeder, leg):
    with monkeypatch.context() as m:
        m.setitem(settings._backends, "curve_builder", "python")
        return seeder.fixed_leg_zero(leg)


@pytest.mark.parametrize("days", [0, 200, 400, 40_000])
def test_the_fixed_leg_seed_is_the_python_one(monkeypatch, days):
    """At the curve date, after some coupons and after the last one (None)."""
    with settings.use_backend("cpp"):
        instruments, market = _instruments()
        seeder = _seeds._Seeder(market, market.risk_free_index)
        seeder.t = market.t + dt.timedelta(days=days)
        fixed = [
            leg
            for i in instruments
            if isinstance(i, Swap)
            for leg in (i.receive_leg, i.pay_leg)
            if isinstance(leg, FixedLeg)
        ]
        assert fixed
        zeros = [seeder.fixed_leg_zero(leg) for leg in fixed]
        assert zeros == [_python_zero(monkeypatch, seeder, leg) for leg in fixed]
        assert (days == 40_000) == all(z is None for z in zeros)
