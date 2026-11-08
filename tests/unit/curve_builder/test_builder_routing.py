"""Curve-builder routing: every index gets exactly one curve.

A floating index whose curve discounts (the riskless index, or a collateral index) is
a discount-curve key; any other floating index is projection-only: it is pinned as
``_Proj(index)`` and solved into ``market.projection_curves``, never ``market.curves``.
"""

from types import SimpleNamespace
import pytest
from fintoolsom.curve_builder.builder import _curves_needed, _Proj
from tests.factories.swap_factory import CAMARA, ICP, RISKLESS, TAB_3M, Ccy, LegCase, LegKind, SwapCase, SwapStatus, build_swap



FIXED_CLP = LegCase(Ccy.CLP, LegKind.FIXED)


def _quote(float_kind: LegKind, collateral=None) -> SimpleNamespace:
    # with_spread=True: the builders' float default spread is still a known bug.
    case = SwapCase("irs", FIXED_CLP, LegCase(Ccy.CLP, float_kind), SwapStatus.NOT_STARTED, True)
    swap = build_swap(case)
    return SimpleNamespace(get_instrument=lambda: swap, collateral_index=collateral)


@pytest.mark.parametrize(
    "float_kind, collateral, pinned",
    [
        (LegKind.PRICE, None, {ICP}),                         # riskless index: its curve discounts
        (LegKind.TERM, None, {ICP, _Proj(TAB_3M)}),           # term index: projection-only
        (LegKind.OVERNIGHT, None, {ICP, _Proj(CAMARA)}),      # overnight, discounts nothing here
        (LegKind.OVERNIGHT, CAMARA, {CAMARA}),                # own collateral: its curve discounts
    ],
)
def test_floating_index_routing(float_kind, collateral, pinned):
    assert _curves_needed(_quote(float_kind, collateral), RISKLESS) == frozenset(pinned)


def test_projection_key_reads_its_index_currency():
    assert _Proj(TAB_3M).currency == TAB_3M.currency
    assert _Proj(TAB_3M) == _Proj(TAB_3M) and _Proj(TAB_3M) != TAB_3M
