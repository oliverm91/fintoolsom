from datetime import date

import pytest

from fintoolsom.fixedIncome.utils import get_irr


def test_get_irr_raises_value_error_when_no_cash_flows_after_valuation_date():
    with pytest.raises(ValueError, match="No cash flows after valuation date"):
        get_irr([100], [date(2023, 1, 1)], date(2024, 1, 1), 100)


def test_get_irr_raises_runtime_error_when_not_converged():
    with pytest.raises(RuntimeError):
        get_irr([100], [date(2024, 2, 1)], date(2024, 1, 1), -1000, max_iterations=2)
