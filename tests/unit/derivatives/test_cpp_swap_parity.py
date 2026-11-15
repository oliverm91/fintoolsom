"""The C++ swap valuation gives the Python one's results, to the bit.

Each factory swap is built twice, its legs under the Python and the C++
``"derivatives"`` backend, and valued on the same market. The arithmetic is IEEE
``+ * /`` and correctly rounded sums, with no libm call, so the results are equal on
every platform; on a market of Python curves the C++ code reads them through their
``_dfs_at_ordinals``, on C++ curves straight from their cores.
"""

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.derivatives.calculator import Calculator, _swap_pv
from fintoolsom.derivatives.swaps import (
    LegSide,
    NonStartedSimplifiedSwap,
    SimplifiedSwap,
)
from tests.factories.swap_factory import (
    REPORTING_CURRENCY,
    SwapStatus,
    build_market,
    build_swap,
    swap_cases,
)

pytest.importorskip("fintoolsom._native")


@pytest.fixture(scope="module", params=["python", "cpp"], ids=lambda b: f"{b}-curves")
def market(request):
    with settings.use_backend(request.param, "rates"):
        return build_market()


def _pair(case, wrap=None):
    """The case's swap with Python legs and with C++ legs."""
    out = []
    for backend in ("python", "cpp"):
        with settings.use_backend(backend, "derivatives"):
            swap = build_swap(case)
            if wrap is not None:
                swap = wrap(
                    receive_leg=swap.receive_leg,
                    pay_leg=swap.pay_leg,
                    collateral_index=swap.collateral_index,
                )
        out.append(swap)
    py, cpp = out
    assert not py.receive_leg._native and cpp.receive_leg._native
    return py, cpp


CASES = [pytest.param(c, id=c.id) for c in swap_cases()]
SIMPLIFIED = [pytest.param(c, id=c.id) for c in swap_cases() if not c.uses_uf]
NON_STARTED = [
    pytest.param(c, id=c.id)
    for c in swap_cases()
    if not c.uses_uf and c.status is SwapStatus.NOT_STARTED
]


@pytest.mark.parametrize("case", CASES)
def test_swap_mtm_and_leg_pvs_match(case, market):
    py, cpp = _pair(case)
    assert Calculator.valuate(cpp, market, REPORTING_CURRENCY) == Calculator.valuate(
        py, market, REPORTING_CURRENCY
    )
    for side in LegSide:
        assert Calculator.get_leg_pv(
            cpp, side, market, REPORTING_CURRENCY
        ) == Calculator.get_leg_pv(py, side, market, REPORTING_CURRENCY)


@pytest.mark.parametrize("case", CASES)
def test_leg_flows_match(case, market):
    py, cpp = _pair(case)
    for a, b in ((py.receive_leg, cpp.receive_leg), (py.pay_leg, cpp.pay_leg)):
        mask_a, flows_a = _swap_pv._leg_flows(a, market)
        mask_b, flows_b = _swap_pv._leg_flows(b, market)
        np.testing.assert_array_equal(mask_a, mask_b)
        np.testing.assert_array_equal(flows_a, flows_b)


@pytest.mark.parametrize("case", SIMPLIFIED)
def test_simplified_swap_matches(case, market):
    py, cpp = _pair(case, SimplifiedSwap)
    assert Calculator.valuate(cpp, market, REPORTING_CURRENCY) == Calculator.valuate(
        py, market, REPORTING_CURRENCY
    )


@pytest.mark.parametrize("case", NON_STARTED)
def test_non_started_swap_matches(case, market):
    py, cpp = _pair(case, NonStartedSimplifiedSwap)
    assert Calculator.valuate(cpp, market, REPORTING_CURRENCY) == Calculator.valuate(
        py, market, REPORTING_CURRENCY
    )


def test_a_missing_curve_raises_the_same_error(market):
    case = next(c for c in swap_cases() if c.collateral != "uncoll")
    py, cpp = _pair(case)
    empty = build_market()
    empty.curves.clear()
    errors = []
    for swap in (py, cpp):
        with pytest.raises(KeyError) as e:
            Calculator.valuate(swap, empty, REPORTING_CURRENCY)
        errors.append(str(e.value))
    assert errors[0] == errors[1]


def test_a_copied_leg_rebuilds_its_core(market):
    import copy
    import pickle

    case = CASES[0].values[0]
    _, cpp = _pair(case)
    expected = Calculator.valuate(cpp, market, REPORTING_CURRENCY)
    assert cpp.receive_leg._core is not None
    for clone in (copy.deepcopy(cpp), pickle.loads(pickle.dumps(cpp))):
        assert clone.receive_leg._core is None
        assert Calculator.valuate(clone, market, REPORTING_CURRENCY) == expected


def test_the_backend_is_read_when_the_leg_is_built():
    case = CASES[0].values[0]
    with settings.use_backend("cpp", "derivatives"):
        swap = build_swap(case)
    with settings.use_backend("python", "derivatives"):
        assert swap.receive_leg._native
