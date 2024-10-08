from dataclasses import dataclass, field
from datetime import date
from typing import Self

from holidays import country_holidays, HolidayBase

@dataclass(slots=True)
class Calendar:
    country: str = field(default=None)
    subdiv: str = field(default=None)
    custom_holidays: list[date] = field(default_factory=list)
    _calendar: HolidayBase = field(default=None)

    _weekend_weekdays: set[int] = field(init=False, default_factory=set)
    def __post_init__(self):
        self._weekend_weekdays = {5,6}

        if self._calendar is None:
            if self.custom_holidays is None:
                self.custom_holidays = []
            if self.country is None:
                self._calendar = HolidayBase()
            else:
                self._calendar = country_holidays(self.country, subdiv=self.subdiv)
        
            self._calendar.update(self.custom_holidays)
    
    def add_custom_holiday(self, custom_holiday: date):
        self._calendar.append(custom_holiday)

    def add_custom_holidays(self, custom_holidays: list[date]):
        for ch in custom_holidays:
            self.add_custom_holiday(ch)
    
    def is_holiday(self, date: date) -> bool:
        return date in self._calendar or date.weekday() in self._weekend_weekdays

    def add_business_days(self, t: date, business_days: int) -> date:
        return self._calendar.get_nth_working_day(t, business_days)

    def combine(self, other: Self) -> Self:
        combined_calendar = self._calendar + other._calendar
        return Calendar(_calendar=combined_calendar)
    
    def __add__(self, other: Self) -> Self:
        return self.combine(other)