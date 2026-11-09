from datetime import date

import pytest

from fintoolsom.dates import Calendar
from fintoolsom.dates.adjustments import ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit

ADJ = ModifiedFollowingConvention(Calendar())


@pytest.mark.parametrize(
    "value, unit, expected",
    [
        (2, TermUnit.D, date(2024, 1, 17)),  # business days
        (1, TermUnit.W, date(2024, 1, 22)),
        (1, TermUnit.M, date(2024, 2, 15)),
        (1, TermUnit.Y, date(2025, 1, 15)),
    ],
)
def test_term_advances_and_adjusts(value, unit, expected):
    assert Term(value, unit, ADJ).advance(date(2024, 1, 15)) == expected


def test_term_adjusts_a_weekend_maturity():
    # 2024-06-15 + 1M = Monday 2024-07-15 (no move); 2024-05-15 + 1M = Saturday -> Monday 17.
    assert Term(1, TermUnit.M, ADJ).advance(date(2024, 5, 15)) == date(2024, 6, 17)


def test_term_str():
    assert str(Term(3, TermUnit.M, ADJ)) == "3M"
