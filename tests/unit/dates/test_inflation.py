from datetime import date

import pytest

from fintoolsom.dates import last_known_uf_date, next_cpi_publication_date

# (today, last UF date known, next CPI release). The CPI is released on the 8th, or
# the preceding CL business day: before it the UF is known up to the 9th of this
# month, from it on up to the 9th of the next.
UF_DATES = [
    # 8-Aug-2024 is a Thursday: released on the 8th.
    (date(2024, 8, 4), date(2024, 8, 9), date(2024, 8, 8)),
    (date(2024, 8, 7), date(2024, 8, 9), date(2024, 8, 8)),
    (date(2024, 8, 8), date(2024, 9, 9), date(2024, 9, 6)),  # 8-Sep is a Sunday
    # 8-Jun-2026 is a Monday with no CL holiday.
    (date(2026, 6, 3), date(2026, 6, 9), date(2026, 6, 8)),
    (date(2026, 6, 25), date(2026, 7, 9), date(2026, 7, 8)),
    # 9-Nov-2026 is a Monday: released Friday 6-Nov.
    (date(2026, 11, 5), date(2026, 11, 9), date(2026, 11, 6)),
    (date(2026, 11, 6), date(2026, 12, 9), date(2026, 12, 7)),  # 8-Dec holiday
    (date(2026, 11, 7), date(2026, 12, 9), date(2026, 12, 7)),
    # Year roll. 8-Dec-2026 (Inmaculada Concepción) is a CL holiday, so the release
    # is Monday 7-Dec: a weekends-only calendar would keep the 7th before it.
    (date(2026, 12, 7), date(2027, 1, 9), date(2027, 1, 8)),
    (date(2026, 12, 20), date(2027, 1, 9), date(2027, 1, 8)),
]


@pytest.mark.parametrize("t, last_known, next_release", UF_DATES)
def test_uf_and_cpi_dates(t, last_known, next_release):
    assert last_known_uf_date(t) == last_known
    assert next_cpi_publication_date(t) == next_release
    assert next_release > t


def test_uf_last_published_date_is_last_known_uf_date():
    from fintoolsom.market import uf

    assert [uf.last_published_date(t) for t, _, _ in UF_DATES] == [
        last_known for _, last_known, _ in UF_DATES
    ]
