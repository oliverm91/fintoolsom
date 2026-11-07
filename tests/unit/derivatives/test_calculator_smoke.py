"""Smoke tests: every swap combination from the factory valuates without raising.

Values are not checked yet — only that Calculator.valuate runs and returns a float.
"""

import pytest
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.market import Market
from tests.factories.swap_factory import (
    REPORTING_CURRENCY,
    SwapCase,
    build_market,
    build_swap,
    swap_case_params,
)



@pytest.fixture(scope="module")
def swap_market() -> Market:
    return build_market()


@pytest.mark.parametrize("case", swap_case_params())
def test_swap_valuates_without_crashing(case: SwapCase, swap_market: Market):
    swap = build_swap(case)
    mtm = Calculator.valuate(swap, swap_market, REPORTING_CURRENCY)
    assert isinstance(mtm, float)
