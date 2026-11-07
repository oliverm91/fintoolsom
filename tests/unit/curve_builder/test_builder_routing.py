"""Curve-builder routing: every index gets exactly one curve.

A floating index whose curve discounts (the riskless index, or a collateral index) is
a discount-curve key; any other floating index is projection-only and must never
become a key of ``market.curves``.
"""

from types import SimpleNamespace
import pytest
from fintoolsom.curve_builder.builder import _curves_needed, _projection_indexes
from tests.factories.swap_factory import CAMARA, ICP, RISKLESS, TAB_3M, Ccy, LegCase, LegKind, SwapCase, SwapStatus, build_swap



FIXED_CLP = LegCase(Ccy.CLP, LegKind.FIXED)


def _quote(float_kind: LegKind, collateral=None) -> SimpleNamespace:
    # with_spread=True: the builders' float default spread is still a known bug.
    case = SwapCase("irs", FIXED_CLP, LegCase(Ccy.CLP, float_kind), SwapStatus.NOT_STARTED, True)
    swap = build_swap(case)
    return SimpleNamespace(get_instrument=lambda: swap, collateral_index=collateral)


@pytest.mark.parametrize(
    "float_kind, collateral, projection_only, discount_keys",
    [
        (LegKind.PRICE, None, set(), {ICP}),                   # riskless index: its curve discounts
        (LegKind.TERM, None, {TAB_3M}, {ICP}),                 # term index: projection-only
        (LegKind.OVERNIGHT, None, {CAMARA}, {ICP}),            # overnight, discounts nothing here
        (LegKind.OVERNIGHT, CAMARA, set(), {CAMARA}),          # own collateral: its curve discounts
    ],
)
def test_floating_index_routing(float_kind, collateral, projection_only, discount_keys):
    quote = _quote(float_kind, collateral)
    assert _projection_indexes(quote, RISKLESS) == projection_only
    assert _curves_needed(quote, RISKLESS) == frozenset(discount_keys)
