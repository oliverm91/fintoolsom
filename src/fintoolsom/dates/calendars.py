from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Self

from .holidays import HolidayRule, FridayEasterRule, MonthDayRule, ConsecutiveHolidaySandwichRule, LastWeekWeekdayRule, OrdinalWeekWeekdayRule

@dataclass(slots=True)
class Calendar:
    holiday_rules: list[HolidayRule] = field(default_factory=list)
    custom_holidays: list[date] = field(default_factory=list)

    def __post_init__(self):
        self._add_one_day_week_days: set = {0, 1, 2, 3, 6}
        self._substract_one_day_week_days: set = {1, 2, 3, 4, 5}
    
    def add_custom_holiday(self, custom_holiday: date):
        self.custom_holidays.append(custom_holiday)
        self._fix_custom_holidays()

    def add_custom_holidays(self, custom_holidays: list[date]):
        self.custom_holidays += custom_holidays
        self._fix_custom_holidays()

    def _fix_custom_holidays(self):
        self.custom_holidays = list(set(self.custom_holidays))
        self.custom_holidays.sort()        

    def delete_custom_holidays(self):
        self.custom_holidays.clear()

    def is_holiday(self, date: date) -> bool:
        if date in self.custom_holidays:
            return True
        return any(date == rule.get_date(date.year) for rule in self.holiday_rules)
    
    def get_holidays(self, year: int) -> list[date]:
        customs = [ch for ch in self.custom_holidays if ch.year==year]
        rule_holidays = [r.get_date(year) for r in self.holiday_rules]
        year_holidays = list(set(customs + rule_holidays))
        year_holidays.sort()
        return year_holidays

    def add_business_day(self, t: date) -> date:
        wd = t.weekday()
        following = t + timedelta(days=1 if wd in self._add_one_day_week_days else 3 if wd==4 else 2)
        while self.is_holiday(following):
            following += timedelta(days=1)
        return following

    def add_business_days(self, t: date, business_days: int) -> date:
        result_date = t
        for _ in range(business_days):
            result_date = self.add_business_day(result_date)
        return result_date

    def subtract_business_day(self, t: date) -> date:
        wd = t.weekday()
        preceding = t - timedelta(days=1 if wd in self._substract_one_day_week_days else 3 if wd==0 else 2)
        while self.is_holiday(preceding):
            preceding -= timedelta(days=1)
        return preceding

    def subtract_business_days(self, t: date, business_days: int) -> date:
        result_date = t
        for _ in range(business_days):
            result_date = self.subtract_business_day(result_date)
        return result_date

    def combine(self, other: Self) -> Self:
        combined_rules = self.holiday_rules + other.holiday_rules
        return Calendar(holiday_rules=combined_rules)
    
    def __add__(self, other: Self) -> Self:
        return self.combine(other)


def get_ny_calendar(custom_holidays: list[date]=None) -> Calendar:
    rules = [
        MonthDayRule(1, 1),  # New Year's Day
        OrdinalWeekWeekdayRule(3, 0, 1),  # Martin Luther King Jr. Day
        OrdinalWeekWeekdayRule(3, 0, 2),  # Presidents' Day
        FridayEasterRule(),  # Good Friday
        LastWeekWeekdayRule(0, 5),  # Memorial Day
        MonthDayRule(6, 19),  # Juneteenth
        MonthDayRule(7, 4),  # Independence Day
        OrdinalWeekWeekdayRule(1, 0, 9),  # Labor Day
        OrdinalWeekWeekdayRule(2, 0, 10),  # Columbus Day
        MonthDayRule(11, 11),  # Veterans Day
        OrdinalWeekWeekdayRule(4, 3, 11),  # Thanksgiving Day
        MonthDayRule(12, 25),  # Christmas Day
    ]
    return Calendar(holiday_rules=rules, custom_holidays=custom_holidays)

def get_cl_calendar(custom_holidays: list[date]=None, add_banking_holiday: bool=True) -> Calendar:
    rules = [
        MonthDayRule(1, 1),  # Año nuevo
        FridayEasterRule(),  # Pascua
        MonthDayRule(5, 1),  # Trabajador
        MonthDayRule(5, 21),  # Pacífico
        MonthDayRule(6, 20),  # Pueblos indígenas,
        MonthDayRule(6, 29, monday_adjustable=True),  # San Pedro y San Pablo. Monday adjustable, law 19.668,
        MonthDayRule(7, 16),  # Virgen del Carmen
        MonthDayRule(8, 15),  # Asunción de la Virgen
        MonthDayRule(9, 18),  # Fiestas patrias-18
        MonthDayRule(9, 19),  # Fiestas patrias-19
        ConsecutiveHolidaySandwichRule([MonthDayRule(9, 18), MonthDayRule(9, 19)]), # Law 20.215
        MonthDayRule(10, 12, monday_adjustable=True),  # Encuentro de 2 mundos. Monday adjustable, law 19.668
        MonthDayRule(10, 31),  # Iglesias evangélicas
        MonthDayRule(11, 1),  # Día de Todos los Santos
        MonthDayRule(12, 8),  # Inmaculada concepción
        MonthDayRule(12, 25),  # Navidad
    ]
    if add_banking_holiday:
        rules.append(
            MonthDayRule(12, 31)  # Feriado bancario
        )
    return Calendar(holiday_rules=rules, custom_holidays=custom_holidays)