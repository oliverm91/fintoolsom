"""The C++ calendars and schedules give the Python ones' dates (and errors).

``ScheduleGenerator.generate_schedule`` runs in C++ under the ``"cpp"`` backend of
``"dates"``, on a C++ view of each :class:`Calendar` (its holidays loaded from the
``holidays`` object, many years per call). Every case is generated under both backends.
"""

import itertools
import random
from datetime import date, timedelta

import pytest

from fintoolsom import settings
from fintoolsom.dates import (
    Calendar,
    FollowingConvention,
    ModifiedFollowingConvention,
    ModifiedPrecedingConvention,
    PrecedingConvention,
    ScheduleGenerator,
    get_calendar,
)

pytest.importorskip("fintoolsom._native")

CONVENTIONS = [
    FollowingConvention,
    ModifiedFollowingConvention,
    PrecedingConvention,
    ModifiedPrecedingConvention,
]
KIND = {c: k for k, c in enumerate(CONVENTIONS)}


def _calendars():
    custom = Calendar(
        country="CL", custom_holidays=[date(2027, 3, 15), date(2031, 6, 2)]
    )
    return {
        "weekends": get_calendar(None),
        "CL": get_calendar("CL"),
        "US": get_calendar("US"),
        "GB": get_calendar("GB"),
        "JP": get_calendar("JP"),
        # Private operands: a holidays.HolidaySum whose operands are also queried
        # directly can raise "dictionary changed size during iteration" (in Python).
        "CL+US": Calendar(country="CL") + Calendar(country="US"),
        "CL custom": custom,
    }


CALENDARS = _calendars()


def _outcome(f):
    try:
        return f()
    except Exception as e:  # noqa: BLE001 - both backends must fail alike
        return type(e), str(e)


@pytest.mark.parametrize("name", list(CALENDARS))
def test_calendar_days_match(name):
    calendar = CALENDARS[name]
    core = calendar._native_core()
    start = date(2023, 1, 1)
    for i in range(0, 365 * 10):
        d = start + timedelta(days=i)
        assert core.is_holiday(d) == calendar.is_holiday(d), d
        for n in (-3, -1, 0, 1, 2, 5):
            assert core.add_business_days(d, n) == calendar.add_business_days(d, n), (
                d,
                n,
            )
        for convention in CONVENTIONS:
            assert core.adjust(KIND[convention], d) == convention(calendar).adjust(d), d


def test_a_custom_holiday_added_later_is_seen():
    calendar = Calendar(country="US")
    d = date(2030, 7, 10)
    assert not calendar._native_core().is_holiday(d)
    calendar.add_custom_holiday(d)
    assert calendar._native_core().is_holiday(d) and calendar.is_holiday(d)


def _schedules(seed, n):
    rng = random.Random(seed)
    maturities = ["1m", "2M", "3m", "6m", "9m", "1y", "18m", "2y", "5Y", "10y", "30y"]
    maturities += ["1w", "13w", "2d", "10d", "07m"]
    frequencies = ["1m", "3m", "6m", "1y", "1M", "3M", "0", "0m", "2m"]
    for _ in range(n):
        start = date(2024, 1, 1) + timedelta(days=rng.randrange(0, 3650))
        calendar = CALENDARS[rng.choice(list(CALENDARS))]
        adj = rng.choice(CONVENTIONS)(calendar)
        maturity_adj = rng.choice([None, rng.choice(CONVENTIONS)(calendar)])
        maturity = rng.choice(maturities)
        end = None
        if rng.random() < 0.3:
            end = start + timedelta(days=rng.randrange(-40, 4000))
            maturity = rng.choice([maturity, None, ""])
        yield (
            start,
            maturity,
            rng.choice(frequencies),
            adj,
            maturity_adj,
            end,
            rng.random() < 0.6,
            rng.random() < 0.5,
        )


@pytest.mark.parametrize("seed", range(8))
def test_schedules_match(seed):
    for args in _schedules(seed, 1500):
        out = []
        for backend in ("python", "cpp"):
            with settings.use_backend(backend, "dates"):
                out.append(_outcome(lambda: ScheduleGenerator.generate_schedule(*args)))
        assert out[0] == out[1], args


def test_exact_stubs_and_month_ends():
    calendar = CALENDARS["CL"]
    starts = [
        date(2024, 1, 31),
        date(2024, 2, 29),
        date(2025, 8, 30),
        date(2026, 3, 31),
    ]
    for start, maturity, frequency, stub_first, long_stub in itertools.product(
        starts,
        ["5m", "1y", "25m", "3y"],
        ["1m", "3m", "6m", "1y"],
        [True, False],
        [True, False],
    ):
        args = (
            start,
            maturity,
            frequency,
            ModifiedFollowingConvention(calendar),
            None,
            None,
            stub_first,
            long_stub,
        )
        out = []
        for backend in ("python", "cpp"):
            with settings.use_backend(backend, "dates"):
                out.append(ScheduleGenerator.generate_schedule(*args))
        assert out[0] == out[1], args


def test_unusual_inputs_fall_back_to_python():
    calendar = CALENDARS["CL"]
    for args in [
        (date(2025, 1, 6), "1y", "3m", None, None, None, True, True),  # no convention
        (date(2025, 1, 6), "1y", "0m", ModifiedFollowingConvention(calendar)),  # 0 freq
        (date(2025, 1, 6), " 1y", "3m", ModifiedFollowingConvention(calendar)),
    ]:
        out = []
        for backend in ("python", "cpp"):
            with settings.use_backend(backend, "dates"):
                out.append(_outcome(lambda: ScheduleGenerator.generate_schedule(*args)))
        assert out[0] == out[1]
