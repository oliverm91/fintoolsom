from abc import ABC, abstractmethod
import calendar
from datetime import date, timedelta
from typing import Self


class HolidayRule(ABC):
    @abstractmethod
    def get_date(self, year: int) -> date:
        pass

    @abstractmethod
    def copy(self) -> Self:
        pass

    def __copy__(self) -> Self:
        return self.copy()


class OrdinalWeekWeekdayRule(HolidayRule):
    def __init__(self, ordinal: int, weekday: int, month: int):
        self.month = month
        self.ordinal = ordinal
        self.weekday = weekday

    def get_date(self, year: int) -> date:
        first_day = date(year, self.month, 1)
        days_to_add = (self.weekday - first_day.weekday()) % 7
        first_weekday_occurrence = first_day + timedelta(days=days_to_add)
        return first_weekday_occurrence + timedelta(weeks=self.ordinal - 1)
    
    def copy(self) -> Self:
        return OrdinalWeekWeekdayRule(self.ordinal, self.weekday, self.month)


class LastWeekWeekdayRule(HolidayRule):
    def __init__(self, weekday: int, month: int):
        self.month = month
        self.weekday = weekday

    def get_date(self, year: int) -> date:
        last_day = date(year, self.month, calendar.monthrange(year, self.month)[1])
        days_to_subtract = (last_day.weekday() - self.weekday) % 7
        return last_day - timedelta(days=days_to_subtract)
    
    def copy(self) -> Self:
        return LastWeekWeekdayRule(self.weekday, self.month)


class MonthDayRule(HolidayRule):
    def __init__(self, month: int, day: int, monday_adjustable: bool=False):
        self.month = month
        self.day = day
        self.monday_adjustable = monday_adjustable

    def get_date(self, year: int) -> date:
        t = date(year, self.month, self.day)
        if not self.monday_adjustable:
            return t
        if self.monday_adjustable:
            wd = t.weekday()
            if 1 <= wd <= 3:
                return t + timedelta(days=4-wd)
            elif wd==4:
                return t + timedelta(days=3)
            else:
                return t
    
    def copy(self) -> Self:
        return MonthDayRule(self.month, self.day, monday_adjustable=self.monday_adjustable)
            

class ConsecutiveHolidaySandwichRule(HolidayRule):
    def __init__(self, consecutive_holiday_rules: tuple[MonthDayRule]):
        '''
        consecutive_holiday_rules must be 2 sorted consecutive HolidayRule objects.
        '''
        if len(consecutive_holiday_rules)!=2:
            raise ValueError(f'consecutive_holiday_rules must be of len 2.')
        self.consecutive_holiday_rules: tuple[MonthDayRule] = consecutive_holiday_rules

    def get_date(self, year: int) -> date:
        if self.consecutive_holiday_rules[0].get_date(year).weekday()==1:
            return self.consecutive_holiday_rules[0].get_date(year) - timedelta(days=1)
        elif self.consecutive_holiday_rules[1].get_date(year).weekday()==3:
            return self.consecutive_holiday_rules[1].get_date(year) + timedelta(days=1)
        else:
            return None
    
    def copy(self) -> Self:
        return ConsecutiveHolidaySandwichRule(self.consecutive_holiday_rules)
        

def _easter_sunday(year: int) -> date:
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
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)
class MondayEasterRule(HolidayRule):
    def get_date(self, year: int) -> date:
        return _easter_sunday(year) + timedelta(days=1)
    
    def copy(self) -> Self:
        return MondayEasterRule()

class FridayEasterRule(HolidayRule):
    def get_date(self, year: int) -> date:
        return _easter_sunday(year) - timedelta(days=2)
    
    def copy(self) -> Self:
        return FridayEasterRule()