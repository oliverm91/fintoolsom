"""Forwards updated in place give the curve built on them.

``PillarCurve._set_forward_rates`` is how the curve builder's solver moves its curves
on every iteration instead of constructing new ones: the curve after it must be,
to the bit, the one ``__post_init__`` builds on the new forwards, for every method,
with or without a spot rate or a daily grid. How the solver uses it is tested in
unit/curve_builder/test_solver.py.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.rates import (
    PillarCurve,
    InterpolationMethod,
    Rate,
    RateConvention,
)

T = date(2026, 6, 29)
KNOTS = [T + timedelta(days=d) for d in (0, 30, 91, 365, 730, 1825)]
OLD = [0.05, 0.048, 0.046, 0.045, 0.043]
NEW = [0.031, 0.052, 0.049, 0.038, 0.047]
SPOT = Rate(RateConvention(), 0.05)
PROBE = [T + timedelta(days=d) for d in range(-3, 2200, 3)]


def _index(method, forwards, spot=None, **kw):
    knots = [T + timedelta(days=2), *KNOTS[1:]] if spot else KNOTS
    return PillarCurve(T, knots, forwards, method, spot, **kw)


def _discount(method, forwards, **kw):
    return PillarCurve(T, KNOTS, forwards, method, **kw)


MAKERS = {
    "index": _index,
    "index_spot": lambda m, f, **kw: _index(m, f, SPOT, **kw),
    "discount": _discount,
}


@pytest.mark.filterwarnings("ignore::UserWarning")  # PiecewiseLinear zig-zags
@pytest.mark.parametrize("make", list(MAKERS.values()), ids=list(MAKERS))
@pytest.mark.parametrize("method", list(InterpolationMethod), ids=lambda m: m.name)
@pytest.mark.parametrize("precalculate", [True, False])
def test_updated_curve_is_the_curve_built_on_the_new_forwards(
    make, method, precalculate
):
    if method not in PillarCurve.allowed_methods:
        pytest.skip("not a pillar curve method")
    updated = make(method, OLD, precalculate=precalculate)
    updated._set_forward_rates(list(NEW))
    built = make(method, NEW, precalculate=precalculate)
    assert np.array_equal(updated.forward_rates, built.forward_rates)
    assert updated._log_df_effective == built._log_df_effective
    assert (updated._grid is None) is (built._grid is None)
    if built._grid is not None:
        assert np.array_equal(updated._grid, built._grid)
    assert np.array_equal(updated.get_dfs(PROBE), built.get_dfs(PROBE))
    updated._set_forward_rates(list(OLD))  # and back
    assert np.array_equal(
        updated.get_dfs(PROBE),
        make(method, OLD, precalculate=precalculate).get_dfs(PROBE),
    )


def test_the_number_of_forwards_is_checked():
    curve = _index(InterpolationMethod.PiecewiseConstant, OLD)
    with pytest.raises(ValueError, match="one entry per segment"):
        curve._set_forward_rates(OLD[:-1])
