"""Interest conventions, RateConvention and Rate: df <-> rate, conversions, accessors."""

import math
from copy import copy
from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom.rates import (
    CompoundedInterestConvention,
    ExponentialInterestConvention,
    LinearInterestConvention,
    Rate,
    RateConvention,
)
from tests.factories.rate_factory import comp_act365, exp_act365, lin_act360, rate

LIN, COMP, EXP = (
    LinearInterestConvention,
    CompoundedInterestConvention,
    ExponentialInterestConvention,
)
CONVENTIONS = [LIN, COMP, EXP]

# The closed forms each convention must follow.
_DF = {
    LIN: lambda r, t: 1 / (1 + r * t),
    COMP: lambda r, t: (1 + r) ** -t,
    EXP: lambda r, t: math.exp(-r * t),
}
_RATE = {
    LIN: lambda df, t: (1 / df - 1) / t,
    COMP: lambda df, t: (1 / df) ** (1 / t) - 1,
    EXP: lambda df, t: -math.log(df) / t,
}


# ── Interest conventions ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "convention, name", [(LIN, "lin"), (COMP, "comp"), (EXP, "exp")]
)
def test_interest_conventions_have_short_names(convention, name):
    assert convention.__str__() == name


@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize(
    "rate_value, tf",
    [
        (0.05, 0.5),
        (0.04, 2),
        (0.03, 1.5),
        (0.035, 1.25),
        (0.02, 10),
        (0.0, 3),
        (-0.01, 1),
    ],
)
def test_df_from_rate_follows_its_formula_and_inverts(convention, rate_value, tf):
    df = convention.get_df_from_rate(rate_value, tf)
    assert df == pytest.approx(_DF[convention](rate_value, tf), rel=1e-14)
    assert np.sign(1 - df) == np.sign(rate_value)  # df > 1 exactly for negative rates
    assert convention.get_rate_from_df(df, tf) == pytest.approx(rate_value, abs=1e-15)


@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize(
    "df, tf",
    [
        (0.99, 70 / 360),
        (0.995, 0.25),
        (0.97, 0.75),
        (0.8, 5),
        (1.0, 4),
        (1.01, 2),
        (1.02, 3),
    ],
)
def test_rate_from_df_follows_its_formula(convention, df, tf):
    value = convention.get_rate_from_df(df, tf)
    assert value == pytest.approx(_RATE[convention](df, tf), rel=1e-14, abs=1e-15)
    assert np.sign(value) == np.sign(1 - df)


# ── RateConvention and Rate ─────────────────────────────────────────────────


def test_rate_convention_str_and_copy():
    conv = lin_act360()
    assert str(conv) == "lin/act/360"
    duplicate = copy(conv)
    assert duplicate == conv and duplicate is not conv


def test_rate_holds_its_value_in_rate():
    r = Rate(RateConvention(), 0.05)
    assert r.rate == 0.05
    r.rate = 0.06
    assert r.copy().rate == 0.06
    with pytest.raises(TypeError):
        Rate(lin_act360())


def test_rate_value_is_a_deprecated_alias():
    r = Rate(RateConvention(), 0.05)
    deprecated = r"Rate\.rate_value is deprecated; use Rate\.rate"
    with pytest.warns(DeprecationWarning, match=deprecated):
        assert r.rate_value == 0.05
    with pytest.warns(DeprecationWarning, match=deprecated):
        r.rate_value = 0.07
    assert r.rate == 0.07
    with pytest.warns(DeprecationWarning, match="use Rate.rate instead"):
        assert Rate(RateConvention(), rate_value=0.04).rate == 0.04
    with pytest.raises(TypeError, match="not both"), pytest.warns(DeprecationWarning):
        Rate(RateConvention(), 0.05, rate_value=0.04)


def test_rate_copy_is_independent_from_original():
    original = Rate(lin_act360(), 0.05)
    duplicate = original.copy()
    duplicate.rate = 0.10
    duplicate.rate_convention.time_fraction_base = 365
    assert original.rate == 0.05
    assert original.rate_convention.time_fraction_base == 360


def test_rate_accrued_interest_and_vectorised_discount_factors():
    t = date(2024, 1, 1)
    r = rate(0.05)  # compounded Act/365
    end = date(2024, 7, 1)
    assert r.get_accrued_interest(1_000, t, end) == pytest.approx(
        1_000 * (r.get_wealth_factor(t, end) - 1)
    )
    ends = [t + timedelta(days=365), t + timedelta(days=730)]
    assert r.get_discount_factor(t, ends) == pytest.approx([1 / 1.05, 1 / 1.05**2])


# ── Conversions ─────────────────────────────────────────────────────────────

_FACTORIES = {"lin": lin_act360, "comp": comp_act365, "exp": exp_act365}


@pytest.mark.parametrize(
    "source, target",
    [
        ("lin", "comp"),
        ("comp", "lin"),
        ("comp", "exp"),
        ("exp", "comp"),
        ("lin", "exp"),
        ("exp", "lin"),
    ],
)
@pytest.mark.parametrize("days", [25, 120, 365])
def test_convert_rate_convention_keeps_the_wealth_factor_and_round_trips(
    source, target, days
):
    start = date(2024, 1, 15)
    end = start + timedelta(days=days)
    r = Rate(_FACTORIES[source](), 0.045)
    wf = r.get_wealth_factor(start, end)
    to = _FACTORIES[target]()
    r.convert_rate_convention(to, start, end)
    assert r.rate_convention is to
    assert r.get_wealth_factor(start, end) == pytest.approx(wf, rel=1e-14)
    r.convert_rate_convention(_FACTORIES[source](), start, end)
    assert r.rate == pytest.approx(0.045, rel=1e-12)


def test_convert_linear_to_compounded_matches_the_closed_form():
    start = date(2023, 10, 17)
    end = start + timedelta(days=25)
    r = Rate(lin_act360(), 0.03)
    r.convert_rate_convention(comp_act365(), start, end)
    assert r.rate == pytest.approx((1 + 0.03 * 25 / 360) ** (365 / 25) - 1)
