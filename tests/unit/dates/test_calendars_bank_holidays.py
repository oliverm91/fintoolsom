"""Calendar holiday categories: bank holidays only in CL and JP; subdivisions kept."""

from datetime import date
import pytest
from fintoolsom.dates import Calendar


@pytest.mark.parametrize(
    "country, bank_day",
    [("CL", date(2024, 12, 31)), ("JP", date(2025, 1, 2)), ("JP", date(2025, 1, 3))],
)
def test_bank_holidays_close_cl_and_jp(country, bank_day):
    assert bank_day.weekday() < 5
    assert Calendar(country=country).is_holiday(bank_day)


@pytest.mark.parametrize("country", ["US", "GB", "SE", "KR", "AU"])
def test_other_countries_use_public_holidays_only(country):
    calendar = Calendar(country=country)
    assert calendar._extra_categories == ()
    assert set(calendar._calendar.categories) == {"public"}


def test_us_new_years_eve_is_a_business_day():
    assert not Calendar(country="US").is_holiday(date(2024, 12, 31))


@pytest.mark.parametrize(
    "subdiv, regional_holiday",
    [("AP", date(2024, 6, 7)), ("NB", date(2024, 8, 20))],  # Arica; Chillán
)
def test_cl_subdivision_holidays_survive_the_bank_rebuild(subdiv, regional_holiday):
    assert Calendar(country="CL", subdiv=subdiv).is_holiday(regional_holiday)
    assert Calendar(country="CL", subdiv=subdiv).is_holiday(date(2024, 12, 31))
    assert not Calendar(country="CL").is_holiday(regional_holiday)
