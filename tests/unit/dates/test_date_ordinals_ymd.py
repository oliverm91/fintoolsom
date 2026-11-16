"""``date_ordinals`` of an (n, 3) array of (year, month, day) rows."""

from datetime import date, timedelta

import numpy as np
import pytest

from fintoolsom import settings
from fintoolsom.dates.date_counts import date_ordinals


def _ymd(dates):
    return np.array([(d.year, d.month, d.day) for d in dates])


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_rows_give_the_dates_ordinals(backend):
    rng = np.random.default_rng(0)
    first = date(1, 1, 1).toordinal()
    span = date(9999, 12, 31).toordinal() - first
    dates = [date.fromordinal(int(first + k)) for k in rng.integers(0, span, 5000)]
    # Month ends, leap days and century years.
    dates += [
        date(1900, 2, 28),
        date(2000, 2, 29),
        date(2024, 2, 29),
        date(2100, 3, 1),
        date(1, 1, 1),
        date(9999, 12, 31),
        *(date(2025, 1, 1) + timedelta(d) for d in range(366)),
    ]
    with settings.use_backend(backend, "dates"):
        out = date_ordinals(_ymd(dates))
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, [d.toordinal() for d in dates])


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.uint16])
def test_any_integer_dtype(dtype):
    rows = np.array([[2026, 11, 16], [1999, 12, 31]], dtype=dtype)
    assert date_ordinals(rows).tolist() == [
        date(2026, 11, 16).toordinal(),
        date(1999, 12, 31).toordinal(),
    ]


def test_an_empty_array():
    assert date_ordinals(np.empty((0, 3), dtype=int)).shape == (0,)


@pytest.mark.parametrize(
    "row",
    [
        (2025, 2, 29),
        (1900, 2, 29),
        (2025, 13, 1),
        (2025, 0, 1),
        (2025, 4, 31),
        (2025, 1, 0),
        (0, 1, 1),
        (10000, 1, 1),
    ],
)
def test_invalid_dates_are_rejected(row):
    rows = np.array([(2025, 1, 1), row])
    with pytest.raises(ValueError, match=r"Row 1 is not a date"):
        date_ordinals(rows)


@pytest.mark.parametrize(
    "rows", [np.zeros((2, 4), dtype=int), np.array([[2025.0, 1.0, 1.0]])]
)
def test_rows_must_be_integer_triples(rows):
    with pytest.raises(ValueError, match="integer"):
        date_ordinals(rows)
