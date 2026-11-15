"""The C++ ``dates`` backend gives the Python one's results, bit for bit (and errors).

Under the ``"cpp"`` backend of ``"dates"`` (read at call time), calendars, the
adjustment conventions, :class:`Tenor`, :class:`Term`, day counts, time fractions and
``date_ordinals`` run in C++. Every case runs under both backends.
"""

import calendar as pycalendar
import random
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.dates import (
    Calendar,
    FollowingConvention,
    ModifiedFollowingConvention,
    ModifiedPrecedingConvention,
    PrecedingConvention,
    Tenor,
    get_calendar,
)
from fintoolsom.dates import calendars as calendars_module
from fintoolsom.dates import date_counts, time_fractions
from fintoolsom.dates.inflation import last_known_uf_date, next_cpi_publication_date
from fintoolsom.dates.term import Term, TermUnit

pytest.importorskip("fintoolsom._native")

CONVENTIONS = [
    FollowingConvention,
    ModifiedFollowingConvention,
    PrecedingConvention,
    ModifiedPrecedingConvention,
]
DAY_COUNTS = list(date_counts._KINDS)
TIME_FRACTIONS = list(time_fractions._KINDS)


def _outcome(f):
    try:
        return f()
    except Exception as e:  # noqa: BLE001 - both backends must fail alike
        return type(e), str(e)


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            type(a) is type(b)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
        )
    return type(a) is type(b) and a == b


def _both(f):
    out = []
    for backend in ("python", "cpp"):
        with settings.use_backend(backend, "dates"):
            out.append(_outcome(f))
    return out


def _assert_same(f, *context):
    p, c = _both(f)
    assert _same(p, c), (context, p, c)


def _dates(rng, n):
    """Random dates, with month ends, 29 Februaries and 31sts over-represented."""
    out = []
    for _ in range(n):
        y = rng.randrange(1990, 2080)
        m = rng.randrange(1, 13)
        last = pycalendar.monthrange(y, m)[1]
        d = rng.choice([1, 15, 28, 29, 30, 31, last, rng.randrange(1, last + 1)])
        out.append(date(y, m, min(d, last)))
    return out


@pytest.mark.parametrize("seed", range(4))
def test_day_counts_and_time_fractions_match(seed):
    rng = random.Random(seed)
    starts, ends = _dates(rng, 400), _dates(rng, 400)
    for cls in DAY_COUNTS:
        for sd, ed in zip(starts, ends):
            _assert_same(lambda: cls.get_day_count(sd, ed), cls, sd, ed)
            for base in (360, 365, 365.25):
                _assert_same(lambda: cls.get_time_fraction(sd, ed, base), cls, sd, ed)
        sd, ed = starts[0], ends[0]
        for args in [
            (sd, ends),
            (starts, ed),
            (starts, ends),
            (starts, ends[:-1]),  # unequal lengths
            ([], []),
            (sd, []),
            (datetime(2025, 1, 31), ed),  # not a mapped type
            ([datetime(2025, 1, 31)], [ed]),
        ]:
            _assert_same(lambda: cls.get_day_count(*args), cls, args[:1])
            for base in (360, 365.25, 0):
                _assert_same(lambda: cls.get_time_fraction(*args, base), cls, base)


@pytest.mark.parametrize("seed", range(4))
def test_time_fractions_match(seed):
    rng = random.Random(100 + seed)
    starts, ends = _dates(rng, 500), _dates(rng, 500)
    for cls in TIME_FRACTIONS:
        tf = cls()
        for sd, ed in zip(starts, ends):
            if cls is time_fractions.TF_Actual365_Long:
                for frequency in (1, 2, 4):
                    _assert_same(
                        lambda: tf.get_day_count_factor(sd, ed, frequency), cls
                    )
            elif cls is time_fractions.TF_ActualActualICMA:
                ps, pe = sorted(rng.sample([sd, ed, sd - timedelta(days=91)], 2))
                for frequency in (1, 2, 4):
                    _assert_same(
                        lambda: tf.get_day_count_factor(sd, ed, frequency), cls
                    )
                    _assert_same(
                        lambda: tf.get_day_count_factor(sd, ed, frequency, ps, pe), cls
                    )
            else:
                _assert_same(lambda: tf.get_day_count_factor(sd, ed), cls, sd, ed)
                _assert_same(lambda: tf.get_coupon_factor(sd, ed), cls, sd, ed)


def test_time_fraction_edge_cases_match():
    icma = time_fractions.TF_ActualActualICMA()
    d = date(2025, 3, 31)
    _assert_same(lambda: icma.get_day_count_factor(d, d, 2))  # ZeroDivisionError
    _assert_same(lambda: icma.get_day_count_factor(d, d + timedelta(1), 0))
    when = datetime(2025, 1, 1, 12)
    for cls in TIME_FRACTIONS:  # datetimes stay in Python
        if cls in (
            time_fractions.TF_Actual365_Long,
            time_fractions.TF_ActualActualICMA,
        ):
            _assert_same(lambda: cls().get_day_count_factor(when, d, 1), cls)
        else:
            _assert_same(lambda: cls().get_day_count_factor(when, d), cls)


def test_date_ordinals_match():
    days = [date(2024, 1, 1) + timedelta(days=37 * i) for i in range(200)]
    for make in [
        lambda: days,
        lambda: tuple(days),
        lambda: [],
        lambda: [datetime(2025, 5, 6, 7), date(2025, 1, 1)],
        lambda: (d for d in days),
        lambda: np.array(days, dtype=object),
        lambda: [date(2025, 1, 1), "2025-01-02"],
    ]:
        _assert_same(lambda: date_counts.date_ordinals(make()), type(make()))


def _calendars():
    return {
        "weekends": get_calendar(None),
        "CL": get_calendar("CL"),
        "US": get_calendar("US"),
        "JP": get_calendar("JP"),
        "CL+US": Calendar(country="CL") + Calendar(country="US"),
    }


CALENDARS = _calendars()


@pytest.mark.parametrize("name", list(CALENDARS))
def test_tenors_terms_and_conventions_match(name):
    calendar = CALENDARS[name]
    rng = random.Random(name)
    tenors = [
        "1d",
        "2D",
        "5d",
        "-3d",
        "1w",
        "13W",
        "1m",
        "3M",
        "7m",
        "1y",
        "10Y",
        "-2m",
    ]
    for _ in range(600):
        t = date(2024, 1, 1) + timedelta(days=rng.randrange(0, 5000))
        convention = rng.choice(CONVENTIONS)(calendar)
        _assert_same(lambda: convention.adjust(t), name, t)
        _assert_same(lambda: calendar.is_holiday(t), name, t)
        n = rng.randrange(-30, 30)
        _assert_same(lambda: calendar.add_business_days(t, n), name, t, n)
        tenor = Tenor(rng.choice(tenors), convention)
        forward = rng.random() < 0.7
        periods = rng.randrange(-3, 12)
        _assert_same(lambda: tenor.get_unadjusted_maturity(t, forward, periods), t)
        _assert_same(lambda: tenor.get_adjusted_maturity(t, forward), tenor, t)
        term = Term(rng.randrange(0, 40), rng.choice(list(TermUnit)), convention)
        _assert_same(lambda: term.advance(t), str(term), t)


def test_inflation_dates_match():
    for i in range(0, 365 * 4):
        t = date(2024, 1, 1) + timedelta(days=i)
        _assert_same(lambda: last_known_uf_date(t), t)
        _assert_same(lambda: next_cpi_publication_date(t), t)


def test_off_path_inputs_stay_in_python():
    calendar = CALENDARS["CL"]
    when = datetime(2025, 1, 1, 9)
    convention = ModifiedFollowingConvention(calendar)
    for f in [
        lambda: calendar.is_holiday(when),
        lambda: calendar.add_business_days(when, 2),
        lambda: calendar.add_business_days(date(2025, 1, 1), 2.0),
        lambda: convention.adjust(when),
        lambda: Tenor("3m", convention).get_unadjusted_maturity(when),
        lambda: Tenor("3m", convention).get_unadjusted_maturity(
            date(2025, 1, 31), True, 1.5
        ),
        lambda: Term(3, TermUnit.M, convention).advance(when),
    ]:
        _assert_same(f)


def test_holidays_load_in_one_call(monkeypatch):
    """A calendar's core loads every year it needs from Python in one call; a date far
    outside them extends the window in one more."""
    calls = []
    original = calendars_module._range_flags

    def spy(holidays, first, last):
        calls.append((first, last))
        return original(holidays, first, last)

    monkeypatch.setattr(calendars_module, "_range_flags", spy)
    calendar = Calendar(country="CL")
    convention = ModifiedFollowingConvention(calendar)
    with settings.use_backend("cpp", "dates"):
        start = date(2025, 3, 14)
        for months in range(0, 12 * 50, 3):  # a 50-year quarterly schedule
            Tenor(f"{months}m", convention).get_adjusted_maturity(start)
        assert len(calls) == 1
        first, last = calls[0]
        assert first <= 2025 and last >= 2075
        assert calendar._native_core().years == (first, last)
        calendar.is_holiday(date(last + 1, 6, 1))
        calendar.is_holiday(date(first - 1, 6, 1))
        assert len(calls) == 3
