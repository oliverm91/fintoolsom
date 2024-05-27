from dataclasses import dataclass, field
from datetime import date
from enum import Enum, unique

from .holiday_rules import LocalHolidays, MonthDayHolidayRule, MonthLastWeekdayRule, MonthOrdinalWeekdayRule, Weekday, Month

@unique
class Currency(Enum):
    USD = 'usd'
    EUR = 'eur'
    GBP = 'gbp'
    CLP = 'clp'
    COP = 'cop'
    BRL = 'brl'
    PEN = 'pen'
    JPY = 'jpy'
    CNY = 'cny'
    AUD = 'aud'
    CAD = 'cad'

@dataclass(slots=True)
class Locality:
    currency: Currency
    name: str
    local_holidays: LocalHolidays = field(default_factory=LocalHolidays)

    def get_year_holidays(self, year: int) -> set[date]:
        return self.local_holidays.get_year_holidays(year)
    
locality_chile = Locality(Currency.CLP, 'Chile', LocalHolidays([
    MonthDayHolidayRule(Month.MAY, 1),  # Labor Day
    MonthDayHolidayRule(Month.JUNE, 20),  # National Indigenous Peoples Day
    MonthDayHolidayRule(Month.JULY, 16),  # Our Lady of Mount Carmel
    MonthDayHolidayRule(Month.AUGUST, 15),  # Assumption of Mary
    MonthDayHolidayRule(Month.SEPTEMBER, 18),  # Independence Day
    MonthDayHolidayRule(Month.SEPTEMBER, 19),  # Army Day
    MonthDayHolidayRule(Month.OCTOBER, 12),  # Columbus Day
    MonthDayHolidayRule(Month.OCTOBER, 31),  # Reformation Day
    MonthDayHolidayRule(Month.NOVEMBER, 1),  # All Saints' Day
    MonthDayHolidayRule(Month.DECEMBER, 8),  # Immaculate Conception
    MonthDayHolidayRule(Month.DECEMBER, 24),  # Christmas Eve
], add_easter_monday=False))

locality_usa = Locality(Currency.USD, 'UnitedStates', LocalHolidays([
    MonthOrdinalWeekdayRule(Month.JANUARY, 3, Weekday.MONDAY),  # Martin Luther King Jr. Day
    MonthOrdinalWeekdayRule(Month.FEBRUARY, 3, Weekday.MONDAY),  # Presidents' Day
    MonthLastWeekdayRule(Month.MAY, Weekday.MONDAY),  # Memorial Day
    MonthDayHolidayRule(Month.JUNE, 19),  # Juneteenth National Independence Day
    MonthDayHolidayRule(Month.JULY, 4),  # Independence Day
    MonthOrdinalWeekdayRule(Month.SEPTEMBER, 1, Weekday.MONDAY),  # Labor Day
    MonthOrdinalWeekdayRule(Month.OCTOBER, 2, Weekday.MONDAY),  # Columbus Day
    MonthDayHolidayRule(Month.NOVEMBER, 11),  # Veterans Day
    MonthOrdinalWeekdayRule(Month.NOVEMBER, 4, Weekday.THURSDAY),  # Thanksgiving Day
], add_easter_monday=False))


locality_eurozone = Locality(Currency.EUR, 'Eurozone', LocalHolidays([
    MonthDayHolidayRule(Month.MAY, 1),      # Labour Day
    MonthDayHolidayRule(Month.DECEMBER, 26) # Boxing Day
]))
locality_england = Locality(Currency.GBP, 'England', LocalHolidays([
    MonthDayHolidayRule(Month.MAY, 8),        # Early May Bank Holiday (example date)
    MonthLastWeekdayRule(Month.MAY, Weekday.MONDAY),  # Spring Bank Holiday
    MonthDayHolidayRule(Month.AUGUST, 31),    # Summer Bank Holiday (example date)
    MonthDayHolidayRule(Month.DECEMBER, 28)   # Boxing Day (substitute day)
]))