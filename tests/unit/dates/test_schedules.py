from datetime import date

import pytest

from fintoolsom.dates import Calendar, ScheduleGenerator, Tenor
from fintoolsom.dates.adjustments import (
    FollowingConvention,
    ModifiedFollowingConvention,
)

ADJ = ModifiedFollowingConvention(Calendar())
generate = ScheduleGenerator.generate_schedule


def test_schedule_generator_semiannual_schedule_over_one_year():
    schedule = generate(date(2024, 1, 1), "1y", "6m", ADJ)
    assert schedule == [date(2024, 1, 1), date(2024, 7, 1), date(2025, 1, 1)]


@pytest.mark.parametrize("stub_first", [True, False])
@pytest.mark.parametrize("long_stub", [True, False])
def test_regular_schedule_has_no_stub_whatever_the_stub_options(stub_first, long_stub):
    schedule = generate(
        date(2024, 1, 15), "1y", "6m", ADJ, stub_first=stub_first, long_stub=long_stub
    )
    assert schedule == [date(2024, 1, 15), date(2024, 7, 15), date(2025, 1, 15)]


@pytest.mark.parametrize(
    "stub_first, long_stub, expected_middle",
    [
        (True, True, [date(2024, 9, 15)]),  # long first stub: 15 Jan - 15 Sep
        (True, False, [date(2024, 3, 15), date(2024, 9, 15)]),  # short first stub
        (False, True, [date(2024, 7, 15)]),  # long final stub: 15 Jul - 15 Mar
        (False, False, [date(2024, 7, 15), date(2025, 1, 15)]),  # short final stub
    ],
)
def test_stub_placement_and_length(stub_first, long_stub, expected_middle):
    start, end = date(2024, 1, 15), date(2025, 3, 15)
    schedule = generate(
        start, "", "6m", ADJ, end_date=end, stub_first=stub_first, long_stub=long_stub
    )
    assert schedule == [start, *expected_middle, end]


def test_maturity_is_adjusted_with_its_own_convention():
    # 2024-06-15 + 1Y = Sunday 2025-06-15 -> Monday 2025-06-16.
    schedule = generate(
        date(2024, 6, 15),
        "1y",
        "1y",
        ADJ,
        maturity_adj_conv=FollowingConvention(Calendar()),
    )
    assert schedule == [date(2024, 6, 15), date(2025, 6, 16)]


def test_unsupported_frequency_is_rejected():
    with pytest.raises(ValueError):
        generate(date(2024, 1, 15), "1y", "2m", ADJ)


def test_zero_frequency_cannot_build_a_schedule():
    with pytest.raises(Exception, match="Max iterations"):
        generate(date(2024, 1, 15), "1y", "0m", ADJ)


# ── Tenor ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tenor, expected",
    [
        ("3d", date(2024, 1, 18)),  # business days (Mon 15 -> Thu 18)
        ("2w", date(2024, 1, 29)),
        ("3m", date(2024, 4, 15)),
        ("2y", date(2026, 1, 15)),
    ],
)
def test_tenor_unadjusted_maturity(tenor, expected):
    assert Tenor(tenor).get_unadjusted_maturity(date(2024, 1, 15)) == expected


def test_tenor_backwards_and_adjusted():
    t = Tenor("1M", ADJ)
    assert t.get_unadjusted_maturity(date(2024, 3, 15), forward=False) == date(
        2024, 2, 15
    )
    # 2024-08-31 is a Saturday; modified following stays in August -> Friday 30.
    assert t.get_adjusted_maturity(date(2024, 7, 31)) == date(2024, 8, 30)
    assert Tenor("1d").get_unadjusted_maturity(
        date(2024, 1, 15), forward=False
    ) == date(2024, 1, 12)


@pytest.mark.parametrize("bad", ["m", "3x"])
def test_tenor_rejects_malformed_strings(bad):
    with pytest.raises(ValueError):
        Tenor(bad)
