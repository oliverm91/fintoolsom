"""Curve-builder routing: every index gets exactly one curve, keyed by the index.

Whether that curve also discounts (the risk-free index, or a collateral index) or
only projects (any other floating index), it is the same IndexCurve key; a
currency other than the risk-free index's is a DiscountCurve key.
"""

from types import SimpleNamespace
import pytest
from fintoolsom.market.currencies import Currency, CurrencyName
from fintoolsom.curve_builder._keys import _curve_id, _curves_needed
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
        (LegKind.TERM, None, {ICP, TAB_3M}),                  # term index: projection-only
        (LegKind.OVERNIGHT, None, {ICP, CAMARA}),             # overnight, discounts nothing here
        (LegKind.OVERNIGHT, CAMARA, {CAMARA}),                # own collateral: its curve discounts
    ],
)
def test_floating_index_routing(float_kind, collateral, pinned):
    assert _curves_needed(_quote(float_kind, collateral), RISKLESS) == frozenset(pinned)


def test_curve_ids_of_index_and_currency_keys():
    assert _curve_id(TAB_3M, RISKLESS) == TAB_3M.name
    assert _curve_id(Currency(CurrencyName.USD), RISKLESS) == f"USD_{RISKLESS.name}"
