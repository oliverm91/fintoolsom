from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from dateutil.relativedelta import relativedelta

class Weekday(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

class Month(Enum):
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12


class UniversalHolidays:
    @staticmethod
    def get_year_holidays(year: int, add_easter_monday: bool=True) -> set[date]:
        easter = UniversalHolidays.get_easter(year, add_easter_monday)
        holidays = set([
            date(year, 1, 1),  # New Year's Day
            date(year, 12, 25),# Christmas Day
            date(year, 12, 31) # New Year's Eve
        ])
        for eh in easter:
            holidays.add(eh)
        return holidays

    @staticmethod
    def get_easter(year: int, add_easter_monday: bool) -> tuple[date, date]:
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1

        easter_sunday = date(year, month, day)
        if add_easter_monday:
            return easter_sunday - timedelta(days=2), easter_sunday + timedelta(days=1)
        else:
            return (easter_sunday - timedelta(days=2),)


@dataclass
class HolidayRule(ABC):
    month: Month
    @abstractmethod
    def get_holiday(self, year: int) -> date:
        pass


@dataclass
class MonthOrdinalWeekdayRule(HolidayRule):
    ordinal: int
    weekday: Weekday

    def get_holiday(self, year: int) -> date:
        first_day_of_month = date(year, self.month.value, 1)
        first_weekday_delta = (self.weekday.value - first_day_of_month.weekday() + 7) % 7
        first_weekday = first_day_of_month + timedelta(days=first_weekday_delta)
        holiday = first_weekday + timedelta(weeks=self.ordinal - 1)
        if holiday.month != self.month.value:
            ordinal_str_suffix = 'st' if self.ordinal==1 else ('nd' if self.ordinal==2 else ('rd' if self.ordinal==3 else 'th'))
            raise ValueError(f'{self.ordinal}{ordinal_str_suffix} {self.weekday.name} not found in {self.month.name} {year}')
        return holiday
    
@dataclass
class MonthLastWeekdayRule(HolidayRule):
    weekday: Weekday

    def get_holiday(self, year: int) -> date:
        start_month = date(year, self.month, 1)
        next_month = start_month + relativedelta(months=1)
        end_month = next_month - timedelta(days=1)

        # Calculate the offset to the previous occurrence of the specified weekday
        offset = (end_month.weekday() - self.weekday.value) % 7
        last_weekday = end_month - timedelta(days=offset)
        
        return last_weekday

@dataclass
class MonthDayHolidayRule(HolidayRule):
    day: int

    def get_holiday(self, year: int) -> date:
        return date(year, self.month, self.day)


@dataclass(slots=True)
class LocalHolidays:
    holiday_rules: list[HolidayRule] = field(default=list)
    add_easter_monday: bool=field(default=True)

    def get_year_holidays(self, year: int) -> set[date]:
        own_holidays = {rule.get_holiday(year) for rule in self.holiday_rules}
        universal_holidays = UniversalHolidays.get_year_holidays(year, add_easter_monday=self.add_easter_monday)
        return own_holidays|universal_holidays
    
    def add_month_day_holiday_rule(self, month: Month, day: int):
        self.holiday_rules.append(MonthDayHolidayRule(month, day))

    def remove_month_day_holiday_rule(self, month: Month, day: int):
        self.holiday_rules = [hr for hr in self.holiday_rules 
                                if not (isinstance(hr, MonthDayHolidayRule) and hr.month == month and hr.day == day)]

    def add_month_ordinal_dayweek_holiday_rule(self, month: Month, ordinal: int, weekday: Weekday):
        self.holiday_rules.append(MonthOrdinalWeekdayRule(month, ordinal, weekday))

    def remove_month_ordinal_dayweek_holiday_rule(self, month: Month, ordinal: int, weekday: Weekday):
        self.holiday_rules = [hr for hr in self.holiday_rules 
                                if not (isinstance(hr, MonthOrdinalWeekdayRule) and hr.month == month and hr.ordinal == ordinal and hr.weekday == weekday)]