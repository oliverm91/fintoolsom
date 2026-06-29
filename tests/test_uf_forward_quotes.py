import math
from datetime import date

import pytest

from fintoolsom.market import (
    Locality,
    ForwardUFPriceQuote,
    ForwardUFPointsQuote,
)
from fintoolsom.market.quotes import ForwardUFQuote  # internal ABC — not re-exported

_PMT = date(2026, 12, 31)
_FIX = date(2026, 12, 29)


def test_uf_quote_base_is_abstract():
    with pytest.raises(TypeError):
        ForwardUFQuote(38000.0, _PMT, True, _FIX)  # type: ignore[abstract]


def test_uf_price_quote_returns_outright_level():
    q = ForwardUFPriceQuote(value=38150.25, payment_date=_PMT, is_buy=True, fixing_date=_FIX)
    assert q.to_outright(37000.0) == 38150.25


def test_uf_points_quote_adds_points_to_spot():
    q = ForwardUFPointsQuote(
        value=150.25, payment_date=_PMT, is_buy=True, fixing_date=_FIX, spot_uf=37000.0
    )
    assert math.isclose(q.to_outright(37000.0), 37150.25, rel_tol=1e-12)


def test_uf_points_quote_with_divisor():
    q = ForwardUFPointsQuote(
        value=15025.0, payment_date=_PMT, is_buy=True, fixing_date=_FIX,
        spot_uf=37000.0, points_divisor=100,
    )
    assert math.isclose(q.to_outright(37000.0), 37000.0 + 150.25, rel_tol=1e-12)


def test_uf_points_quote_rejects_non_positive_divisor():
    with pytest.raises(ValueError):
        ForwardUFPointsQuote(
            value=150.0, payment_date=_PMT, is_buy=True, fixing_date=_FIX,
            spot_uf=37000.0, points_divisor=0,
        )


def test_uf_quotes_default_to_chilean_locality():
    assert ForwardUFPriceQuote(
        value=38000.0, payment_date=_PMT, is_buy=True, fixing_date=_FIX
    ).locality == Locality.CL
    assert ForwardUFPointsQuote(
        value=100.0, payment_date=_PMT, is_buy=True, fixing_date=_FIX, spot_uf=37000.0
    ).locality == Locality.CL


def test_uf_price_quote_get_instrument_returns_uf_ndf():
    from fintoolsom.derivatives.forwards.forwards import NDF

    q = ForwardUFPriceQuote(value=38150.25, payment_date=_PMT, is_buy=True, fixing_date=_FIX)
    instrument = q.get_instrument()
    assert isinstance(instrument, NDF)
    assert instrument.is_uf_indexed
    assert instrument.strike == 38150.25
    assert instrument.payment_date == _PMT
    assert instrument.fixing_date == _FIX
    assert instrument.is_buy is True
    assert instrument.notional == 100.0


def test_uf_points_quote_get_instrument_uses_spot():
    from fintoolsom.derivatives.forwards.forwards import NDF

    q = ForwardUFPointsQuote(
        value=150.0, payment_date=_PMT, is_buy=False, fixing_date=_FIX,
        spot_uf=38000.0,
    )
    instrument = q.get_instrument()
    assert isinstance(instrument, NDF)
    assert instrument.is_uf_indexed
    assert math.isclose(instrument.strike, 38150.0, rel_tol=1e-12)
    assert instrument.is_buy is False
