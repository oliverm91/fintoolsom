from datetime import date, timedelta
import pytest
from fintoolsom.fixedIncome.utils import get_irr


def test_get_irr_raises_value_error_when_no_cash_flows_after_valuation_date():
    with pytest.raises(ValueError, match="No cash flows after valuation date"):
        get_irr([100], [date(2023, 1, 1)], date(2024, 1, 1), 100)


def test_get_irr_raises_runtime_error_when_not_converged():
    with pytest.raises(RuntimeError):
        get_irr([100], [date(2024, 2, 1)], date(2024, 1, 1), -1000, max_iterations=2)


@pytest.mark.parametrize("notional", [1_000_000.0, 385_000_000.0, 1_000_000_000.0])
def test_get_irr_converges_on_large_notionals(notional):
    t0 = date(2024, 1, 1)
    dates = [t0 + timedelta(days=round(182.5 * k)) for k in range(1, 21)]
    flows = [notional * 0.025] * 19 + [notional * 1.025]
    for i in range(30):
        r_true = 0.03 + i * 0.0017
        pv = sum(
            f / (1 + r_true) ** ((d - t0).days / 365.0) for f, d in zip(flows, dates)
        )
        assert get_irr(flows, dates, t0, pv) == pytest.approx(r_true, abs=1e-11)


def test_get_irr_stays_above_minus_one_and_validates_inputs():
    t0 = date(2024, 1, 1)
    # 100 / (1 + r)^2 == 300 => physical root is 1 / sqrt(3) - 1 ≈ -0.42265 (not -1.57735).
    expected = 1.0 / (3.0**0.5) - 1.0
    assert get_irr([100.0], [t0 + timedelta(days=730)], t0, 300.0) == pytest.approx(
        expected, abs=1e-11
    )

    with pytest.raises(ValueError):
        get_irr([100.0, 50.0], [t0 + timedelta(days=365)], t0, 100.0)

    with pytest.raises(ValueError, match="initial_guess"):
        get_irr([100.0], [t0 + timedelta(days=365)], t0, 95.0, initial_guess=-1.0)

    with pytest.raises(RuntimeError):
        get_irr([100.0], [t0 + timedelta(days=365)], t0, 95.0, max_iterations=0)
