"""The ``curve_builder`` backend: the switch, the C++ builder against the Python one
(the branch's acceptance criteria), and builds independent of ``PYTHONHASHSEED``."""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.curve_builder import RedundantQuoteWarning, build_curves
from fintoolsom.curve_builder._keys import _instrument_of
from fintoolsom.curve_builder.builder import _REPORT_CURRENCY
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.rates import InterpolationMethod
from tests.factories import quote_factory

METHODS = list(InterpolationMethod)
METHODS.remove(InterpolationMethod.LogLinear)  # legacy ZeroCouponCurve only
# The C++ builder's forwards stay this close to the Python builder's.
FORWARD_LIMIT = 1e-8
ROOT = Path(__file__).resolve().parents[3]


def _build(builder, method):
    """A build on the C++ valuation backend with the `builder` curve builder: the
    market, the positions of the quotes it ditched, and the MTM of the others."""
    quotes = quote_factory.load_quotes()
    with settings.use_backend("cpp"), settings.use_backend(builder, "curve_builder"):
        market = quote_factory.quote_market()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_curves(quotes, market, interp_method=method)
        mtm = [
            Calculator.valuate(_instrument_of(q, market), market, _REPORT_CURRENCY)
            for q in quotes
        ]
    position = {id(q): k for k, q in enumerate(quotes)}
    ditched = {
        position[id(d[0])]
        for w in caught
        if isinstance(w.message, RedundantQuoteWarning)
        for d in w.message.ditched
    }
    return market, ditched, [v for k, v in enumerate(mtm) if k not in ditched]


def test_the_global_backend_reaches_the_curve_builder():
    with settings.use_backend("cpp"):
        assert settings.get_backend("curve_builder") == "cpp"
        settings.set_backend("python", "curve_builder")
        assert settings.get_backend("curve_builder") == "python"
        assert settings.get_backend("rates") == "cpp"


@pytest.mark.parametrize("method", METHODS)
def test_cpp_builder_forwards_stay_within_the_limit_of_the_python_builder(method):
    ref, ref_ditched, ref_mtm = _build("python", method)
    cpp, cpp_ditched, cpp_mtm = _build("cpp", method)
    assert ref.curves.keys() == cpp.curves.keys()
    for cid, curve in ref.curves.items():
        np.testing.assert_allclose(
            cpp.curves[cid].forward_rates,
            curve.forward_rates,
            rtol=0,
            atol=FORWARD_LIMIT,
        )
    # Same quotes ditched, and the fitted ones reprice as well as the Python builder's.
    assert cpp_ditched == ref_ditched
    assert max(map(abs, cpp_mtm)) <= max(1e-8, 10 * max(map(abs, ref_mtm)))


@pytest.mark.parametrize(
    "method", [InterpolationMethod.CubicSpline, InterpolationMethod.PiecewiseConstant]
)
def test_builds_do_not_depend_on_the_hash_seed(method):
    """Two processes with different PYTHONHASHSEED build the same curves, to the bit."""
    script = (
        "import hashlib, warnings\n"
        "warnings.simplefilter('ignore')\n"
        "from fintoolsom.curve_builder import build_curves\n"
        "from fintoolsom.rates import InterpolationMethod\n"
        "from tests.factories import quote_factory\n"
        "m = quote_factory.quote_market()\n"
        f"build_curves(quote_factory.load_quotes(), m, interp_method=InterpolationMethod.{method.name})\n"
        "h = hashlib.sha1()\n"
        "for k in sorted(m.curves):\n"
        "    h.update(m.curves[k].forward_rates.tobytes())\n"
        "print(h.hexdigest())\n"
    )
    digests = set()
    for seed in ("0", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)}
        out = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.add(out.stdout.strip().splitlines()[-1])
    assert len(digests) == 1
