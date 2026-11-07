from datetime import date
import pytest
from fintoolsom.market import uf



@pytest.mark.parametrize(
    "today, expected",
    [(date(2024, 8, 4), date(2024, 8, 9)), (date(2024, 8, 8), date(2024, 9, 9))],
    ids=["before-cpi", "cpi-day"],
)
def test_last_published_date_follows_the_cpi_release(today, expected):
    assert uf.last_published_date(today) == expected
