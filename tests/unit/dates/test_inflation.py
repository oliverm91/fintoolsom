from datetime import date

import pytest

from fintoolsom.dates import last_known_uf_date, next_cpi_publication_date


@pytest.mark.parametrize(
    "t, last_known, next_release",
    [
        # 8-Aug-2024 is a Thursday: released on the 8th.
        (date(2024, 8, 7), date(2024, 8, 9), date(2024, 8, 8)),
        (date(2024, 8, 8), date(2024, 9, 9), date(2024, 9, 6)),  # 8-Sep is a Sunday
        # 9-Nov-2026 is a Monday: released Friday 6-Nov.
        (date(2026, 11, 5), date(2026, 11, 9), date(2026, 11, 6)),
        (date(2026, 11, 6), date(2026, 12, 9), date(2026, 12, 7)),  # 8-Dec holiday
        (date(2026, 11, 7), date(2026, 12, 9), date(2026, 12, 7)),
        # Year roll: released Monday 7-Dec-2026, then Friday 8-Jan-2027.
        (date(2026, 12, 7), date(2027, 1, 9), date(2027, 1, 8)),
        (date(2026, 12, 20), date(2027, 1, 9), date(2027, 1, 8)),
    ],
)
def test_uf_and_cpi_dates(t, last_known, next_release):
    assert last_known_uf_date(t) == last_known
    assert next_cpi_publication_date(t) == next_release
    assert next_release > t
