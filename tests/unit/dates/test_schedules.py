from datetime import date
from types import SimpleNamespace

import pytest

from fintoolsom.dates import Calendar, ScheduleGenerator, Tenor
from fintoolsom.dates.adjustments import (
    FollowingConvention,
    ModifiedFollowingConvention,
)

ADJ = ModifiedFollowingConvention(Calendar())
UNADJUSTED = SimpleNamespace(adjust=lambda d: d)
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
        # 15 Sep 2024 is a Sunday, adjusted to Monday 16.
        (True, True, [date(2024, 9, 16)]),  # long first stub: 15 Jan - 16 Sep
        (True, False, [date(2024, 3, 15), date(2024, 9, 16)]),  # short first stub
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


MID_MONTH_SCHEDULE = [
    date(2026, 5, 15),
    date(2026, 8, 17),  # Saturday 15 -> Monday 17
    date(2026, 11, 16),  # Sunday 15 -> Monday 16: rolls from the 15th, not the 17th
    date(2027, 2, 15),
    date(2027, 5, 17),  # Saturday 15 -> Monday 17
]


def test_inner_dates_are_adjusted_after_rolling_on_unadjusted_dates():
    assert generate(date(2026, 5, 15), "1y", "3m", ADJ) == MID_MONTH_SCHEDULE


def test_end_date_equal_to_the_adjusted_tenor_maturity_rolls_from_the_unadjusted_one():
    schedule = generate(date(2026, 5, 15), "1y", "3m", ADJ, end_date=date(2027, 5, 17))
    assert schedule == MID_MONTH_SCHEDULE


def test_end_date_off_the_tenor_maturity_rolls_from_itself():
    schedule = generate(
        date(2026, 5, 15), "1y", "3m", ADJ, end_date=date(2027, 5, 19), long_stub=False
    )
    assert schedule == [
        date(2026, 5, 15),
        date(2026, 5, 19),
        date(2026, 8, 19),
        date(2026, 11, 19),
        date(2027, 2, 19),
        date(2027, 5, 19),
    ]


def test_inner_date_adjusted_onto_the_maturity_is_dropped():
    # Saturday 15 Jun adjusts to Monday 17 Jun, the end date: no zero-length coupon.
    start, end = date(2024, 3, 15), date(2024, 6, 17)
    schedule = generate(start, "", "3m", ADJ, end_date=end, stub_first=False)
    assert schedule == [start, end]


@pytest.mark.parametrize("stub_first", [True, False])
@pytest.mark.parametrize("long_stub", [True, False])
@pytest.mark.parametrize(
    "start, maturity, frequency, expected",
    [
        (  # roll 31 through 30-day months and February, no stub
            date(2026, 8, 31),
            "2y",
            "3m",
            [
                date(2026, 8, 31),
                date(2026, 11, 30),
                date(2027, 2, 28),
                date(2027, 5, 31),
                date(2027, 8, 31),
                date(2027, 11, 30),
                date(2028, 2, 29),
                date(2028, 5, 31),
                date(2028, 8, 31),
            ],
        ),
        (  # roll 30 keeps its day after February
            date(2026, 8, 30),
            "1y",
            "3m",
            [
                date(2026, 8, 30),
                date(2026, 11, 30),
                date(2027, 2, 28),
                date(2027, 5, 30),
                date(2027, 8, 30),
            ],
        ),
        (  # the maturity itself is clamped to February: still regular, roll 31
            date(2026, 8, 31),
            "18m",
            "3m",
            [
                date(2026, 8, 31),
                date(2026, 11, 30),
                date(2027, 2, 28),
                date(2027, 5, 31),
                date(2027, 8, 31),
                date(2027, 11, 30),
                date(2028, 2, 29),
            ],
        ),
    ],
)
def test_short_months_do_not_shift_later_rolls(
    start, maturity, frequency, expected, stub_first, long_stub
):
    schedule = generate(
        start,
        maturity,
        frequency,
        UNADJUSTED,
        stub_first=stub_first,
        long_stub=long_stub,
    )
    assert schedule == expected


def test_stub_rolls_from_the_maturity_day():
    # 8 months quarterly from 31 Aug: short first stub, rolling on the maturity's 30th.
    schedule = generate(
        date(2026, 8, 31),
        "",
        "3m",
        UNADJUSTED,
        end_date=date(2027, 4, 30),
        long_stub=False,
    )
    assert schedule == [
        date(2026, 8, 31),
        date(2026, 10, 30),
        date(2027, 1, 30),
        date(2027, 4, 30),
    ]


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


def test_tenor_periods_multiply_the_step_from_the_same_date():
    t = Tenor("1m")
    assert t.get_unadjusted_maturity(date(2026, 1, 31), periods=2) == date(2026, 3, 31)
    assert t.get_unadjusted_maturity(
        date(2026, 3, 31), forward=False, periods=2
    ) == date(2026, 1, 31)


@pytest.mark.parametrize("bad", ["m", "3x"])
def test_tenor_rejects_malformed_strings(bad):
    with pytest.raises(ValueError):
        Tenor(bad)
