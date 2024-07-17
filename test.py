from datetime import date


def print_cl_holidays(year: int) -> list[date]:
    from fintoolsom.dates.calendars import get_cl_calendar
    cl_calendar = get_cl_calendar()
    year_holidays = cl_calendar.get_holidays(year)
    return year_holidays

print_cl_holidays(2024)