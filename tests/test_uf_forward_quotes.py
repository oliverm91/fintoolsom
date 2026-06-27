import math
import pytest

from fintoolsom.market import (
    Locality,
    ForwardUFQuote,
    ForwardUFPriceQuote,
    ForwardUFPointsQuote,
)


def test_uf_quote_base_is_abstract():
    # UF is not a currency, so the family carries no CurrencyPair; the base cannot
    # be instantiated (to_outright is abstract).
    with pytest.raises(TypeError):
        ForwardUFQuote(38000.0)


def test_uf_price_quote_returns_outright_level():
    q = ForwardUFPriceQuote(38150.25)
    # Outright quote ignores spot and returns the agreed forward UF level.
    assert q.to_outright(37000.0) == 38150.25


def test_uf_points_quote_adds_points_to_spot():
    q = ForwardUFPointsQuote(150.25)  # default divisor 1
    assert math.isclose(q.to_outright(37000.0), 37150.25, rel_tol=1e-12)


def test_uf_points_quote_with_divisor():
    q = ForwardUFPointsQuote(15025.0, points_divisor=100)
    assert math.isclose(q.to_outright(37000.0), 37000.0 + 150.25, rel_tol=1e-12)


def test_uf_points_quote_rejects_non_positive_divisor():
    with pytest.raises(ValueError):
        ForwardUFPointsQuote(150.0, points_divisor=0)


def test_uf_quotes_default_to_chilean_locality():
    assert ForwardUFPriceQuote(38000.0).locality == Locality.CL
    assert ForwardUFPointsQuote(100.0).locality == Locality.CL
