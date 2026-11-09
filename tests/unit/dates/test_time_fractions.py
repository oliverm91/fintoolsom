"""Year fractions checked against a published reference.

Expected values are copied (once, by hand) from OpenGamma Strata's day-count test
table ``DayCountTest.java`` (modules/basics/src/test/java/com/opengamma/strata/
basics/date/), which includes the worked cases of the ISDA paper "The
Actual/Actual Day Count Fraction" (1999). No network access is needed here.
"""

from datetime import date

import pytest

from fintoolsom.dates import (
    TF_30_360BondBasis,
    TF_30E_360,
    TF_30E_360ISDA,
    TF_Actual30,
    TF_Actual360,
    TF_Actual360_25,
    TF_Actual364,
    TF_Actual365,
    TF_Actual365_Long,
    TF_ActualActualICMA,
    TF_ActualActualISDA,
    TF_NL_360,
    TF_NL_365,
)
from tests.factories.day_counts import D, calc360

# (start, end, expected) — Strata rows ``{CONVENTION, y1, m1, d1, y2, m2, d2, value}``.
ACT_ACT_ISDA = [
    (D(2011, 12, 28), D(2012, 2, 28), 4 / 365 + 58 / 366),
    (D(2011, 12, 28), D(2012, 2, 29), 4 / 365 + 59 / 366),
    (D(2011, 12, 28), D(2012, 3, 1), 4 / 365 + 60 / 366),
    (D(2011, 12, 28), D(2016, 2, 28), 4 / 365 + 58 / 366 + 4),
    (D(2011, 12, 28), D(2016, 2, 29), 4 / 365 + 59 / 366 + 4),
    (D(2011, 12, 28), D(2016, 3, 1), 4 / 365 + 60 / 366 + 4),
    (D(2012, 2, 29), D(2012, 3, 29), 29 / 366),
    (D(2012, 2, 29), D(2012, 3, 28), 28 / 366),
    (D(2012, 3, 1), D(2012, 3, 28), 27 / 366),
    # ISDA 1999 paper cases.
    (D(2003, 11, 1), D(2004, 5, 1), 61 / 365 + 121 / 366),
    (D(1999, 2, 1), D(1999, 7, 1), 150 / 365),
    (D(1999, 7, 1), D(2000, 7, 1), 184 / 365 + 182 / 366),
    (D(2002, 8, 15), D(2003, 7, 15), 334 / 365),
    (D(2003, 7, 15), D(2004, 1, 15), 170 / 365 + 14 / 366),
    (D(1999, 7, 30), D(2000, 1, 30), 155 / 365 + 29 / 366),
]

NL_PAIRS = [  # (start, end, days excluding 29 February in (start, end])
    (D(2011, 12, 28), D(2012, 2, 28), 62),
    (D(2011, 12, 28), D(2012, 2, 29), 62),
    (D(2011, 12, 28), D(2012, 3, 1), 63),
    (D(2011, 12, 28), D(2016, 2, 28), 62 + 4 * 365),
    (D(2011, 12, 28), D(2016, 2, 29), 62 + 4 * 365),
    (D(2011, 12, 28), D(2016, 3, 1), 63 + 4 * 365),
    (D(2012, 2, 28), D(2012, 3, 28), 28),
    (D(2012, 2, 29), D(2012, 3, 28), 28),
    (D(2012, 3, 1), D(2012, 3, 28), 27),
    (D(2011, 12, 1), D(2012, 12, 1), 365),
]

ACTUAL_PAIRS = [  # (start, end, actual days)
    (D(2011, 12, 28), D(2012, 2, 28), 62),
    (D(2011, 12, 28), D(2012, 2, 29), 63),
    (D(2011, 12, 28), D(2012, 3, 1), 64),
    (D(2011, 12, 28), D(2016, 2, 28), 62 + 366 + 3 * 365),
    (D(2012, 2, 28), D(2012, 3, 28), 29),
    (D(2012, 2, 29), D(2012, 3, 28), 28),
]

# Strata data_ACT365L, with the period end equal to the end date.
ACT_365L = [
    (D(2011, 12, 28), D(2012, 2, 28), 1, 62 / 365),
    (D(2011, 12, 28), D(2012, 2, 29), 1, 63 / 366),
    (D(2011, 12, 28), D(2012, 2, 28), 2, 62 / 366),
    (D(2011, 12, 28), D(2012, 2, 29), 2, 63 / 366),
    (D(2010, 12, 28), D(2011, 2, 28), 2, 62 / 365),
]

# Strata THIRTY_360_ISDA rows: 30/360 bond basis (2006 ISDA 4.16(f)).
THIRTY_360_ISDA = [
    (D(2011, 12, 28), D(2012, 2, 29), calc360(2011, 12, 28, 2012, 2, 29)),
    (D(2012, 5, 30), D(2013, 8, 29), calc360(2012, 5, 30, 2013, 8, 29)),
    (D(2012, 5, 29), D(2013, 8, 30), calc360(2012, 5, 29, 2013, 8, 30)),
    (D(2012, 5, 29), D(2013, 8, 31), calc360(2012, 5, 29, 2013, 8, 31)),
    (D(2012, 5, 30), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 30), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
]

# Strata THIRTY_E_360 rows (2006 ISDA 4.16(g)).
THIRTY_E_360 = [
    (D(2011, 2, 28), D(2012, 2, 29), calc360(2011, 2, 28, 2012, 2, 29)),
    (D(2012, 2, 29), D(2016, 2, 29), calc360(2012, 2, 29, 2016, 2, 29)),
    (D(2012, 5, 30), D(2013, 8, 29), calc360(2012, 5, 30, 2013, 8, 29)),
    (D(2012, 5, 29), D(2013, 8, 31), calc360(2012, 5, 29, 2013, 8, 30)),
    (D(2012, 5, 30), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 30), calc360(2012, 5, 30, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
]

# Strata data_30E360ISDA, "not maturity" column: a last-of-February end date is
# moved to 30 (the end date is not the termination date).
THIRTY_E_360_ISDA_NOT_MATURITY = [
    (D(2011, 12, 28), D(2012, 2, 28), calc360(2011, 12, 28, 2012, 2, 28)),
    (D(2011, 12, 28), D(2012, 2, 29), calc360(2011, 12, 28, 2012, 2, 30)),
    (D(2012, 2, 29), D(2012, 3, 28), calc360(2012, 2, 30, 2012, 3, 28)),
    (D(2011, 2, 28), D(2012, 2, 28), calc360(2011, 2, 30, 2012, 2, 28)),
    (D(2011, 2, 28), D(2012, 2, 29), calc360(2011, 2, 30, 2012, 2, 30)),
    (D(2012, 2, 29), D(2016, 2, 29), calc360(2012, 2, 30, 2016, 2, 30)),
    (D(2012, 5, 29), D(2013, 8, 31), calc360(2012, 5, 29, 2013, 8, 30)),
    (D(2012, 5, 31), D(2013, 8, 31), calc360(2012, 5, 30, 2013, 8, 30)),
]


@pytest.mark.parametrize("start, end, expected", ACT_ACT_ISDA)
def test_actual_actual_isda(start, end, expected):
    assert TF_ActualActualISDA().get_day_count_factor(start, end) == pytest.approx(
        expected, rel=1e-14
    )


def test_actual_actual_isda_is_zero_on_an_empty_period():
    assert (
        TF_ActualActualISDA().get_day_count_factor(D(2012, 2, 29), D(2012, 2, 29))
        == 0.0
    )


def test_actual_actual_icma_full_regular_period_is_one_over_frequency():
    # ISDA 1999 paper, semi-annual 1 Nov 2003 - 1 May 2004: 182 / (182 * 2).
    tf = TF_ActualActualICMA()
    assert tf.get_day_count_factor(D(2003, 11, 1), D(2004, 5, 1), 2) == pytest.approx(
        0.5, rel=1e-15
    )
    # Short final stub of the 1999 paper: regular period 30 Jul 1999 - 30 Jan 2000.
    assert tf.get_day_count_factor(D(1999, 7, 30), D(2000, 1, 30), 2) == pytest.approx(
        184 / (184 * 2)
    )


def test_actual_actual_icma_partial_accrual_inside_a_reference_period():
    tf = TF_ActualActualICMA()
    # Strata test_actActIcma_singlePeriod: accrued to the day before the coupon date.
    assert tf.get_day_count_factor(
        D(2003, 11, 1),
        D(2004, 4, 30),
        2,
        period_start=D(2003, 11, 1),
        period_end=D(2004, 5, 1),
    ) == pytest.approx(181 / (182 * 2), rel=1e-15)
    # Strata test_actActIcma_middle: quarterly period 30 Mar 2012 - 30 Jun 2012 (92 days).
    assert tf.get_day_count_factor(
        D(2012, 4, 10),
        D(2012, 5, 10),
        4,
        period_start=D(2012, 3, 30),
        period_end=D(2012, 6, 30),
    ) == pytest.approx(30 / (4 * 92), rel=1e-15)


@pytest.mark.parametrize("start, end, frequency, expected", ACT_365L)
def test_actual_365_long(start, end, frequency, expected):
    assert TF_Actual365_Long().get_day_count_factor(
        start, end, frequency
    ) == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("start, end, days", NL_PAIRS)
def test_no_leap_conventions_drop_29_february_after_the_start(start, end, days):
    assert TF_NL_360().get_day_count_factor(start, end) == pytest.approx(
        days / 360, rel=1e-15
    )
    assert TF_NL_365().get_day_count_factor(start, end) == pytest.approx(
        days / 365, rel=1e-15
    )


@pytest.mark.parametrize("start, end, days", ACTUAL_PAIRS)
def test_actual_over_fixed_denominators(start, end, days):
    for tf, base in [
        (TF_Actual360(), 360),
        (TF_Actual364(), 364),
        (TF_Actual365(), 365),
        (TF_Actual30(), 30),
        (TF_Actual360_25(), 365.25),  # Act/365.25 (Strata ACT_365_25).
    ]:
        assert tf.get_day_count_factor(start, end) == pytest.approx(
            days / base, rel=1e-15
        )


@pytest.mark.parametrize("start, end, days", THIRTY_360_ISDA)
def test_30_360_bond_basis(start, end, days):
    assert TF_30_360BondBasis().get_day_count_factor(start, end) == pytest.approx(
        days / 360, rel=1e-15
    )


@pytest.mark.parametrize("start, end, days", THIRTY_E_360)
def test_30e_360(start, end, days):
    assert TF_30E_360().get_day_count_factor(start, end) == pytest.approx(
        days / 360, rel=1e-15
    )


@pytest.mark.parametrize("start, end, days", THIRTY_E_360_ISDA_NOT_MATURITY)
def test_30e_360_isda_moves_a_last_of_february_end_date_to_30(start, end, days):
    assert TF_30E_360ISDA().get_day_count_factor(start, end) == pytest.approx(
        days / 360, rel=1e-15
    )


def test_coupon_factor_defaults_to_the_day_count_factor():
    tf = TF_Actual360()
    start, end = date(2024, 1, 15), date(2024, 7, 15)
    assert tf.get_coupon_factor(start, end) == tf.get_day_count_factor(start, end)
