from abc import ABC, abstractmethod
from datetime import date

from .calendars import Calendar


class AdjustmentDateConventionBase(ABC):
    def __init__(self, calendar: Calendar):
        self.calendar = calendar

    @abstractmethod
    def adjust(self, date_to_adjust: date) -> date:
        pass


class FollowingConvention(AdjustmentDateConventionBase):
    def adjust(self, date_to_adjust: date) -> date:
        if self.calendar.is_holiday(date_to_adjust):
            return self.calendar.add_business_day(date_to_adjust)
        else:
            return date_to_adjust


class ModifiedFollowingConvention(AdjustmentDateConventionBase):
    def __init__(self, calendar: Calendar):
        super().__init__(calendar)
        self._fc = FollowingConvention(calendar)
        self._pc = PrecedingConvention(calendar)

    def adjust(self, date_to_adjust: date) -> date:
        new_date = self._fc.adjust(date_to_adjust)
        if new_date.month==date_to_adjust.month:
            return new_date
        else:
            return self._pc.adjust(date_to_adjust)


class PrecedingConvention(AdjustmentDateConventionBase):
    def adjust(self, date_to_adjust: date) -> date:
        if self.calendar.is_holiday(date_to_adjust):
            return self.calendar.subtract_business_day(date_to_adjust)
        else:
            return date_to_adjust


class ModifiedPrecedingConvention(AdjustmentDateConventionBase):
    def __init__(self, calendar: Calendar):
        super().__init__(calendar)
        self._fc = FollowingConvention(calendar)
        self._pc = PrecedingConvention(calendar)

    def adjust(self, date_to_adjust: date) -> date:
        new_date = self._pc.adjust(date_to_adjust)
        if new_date.month==date_to_adjust.month:
            return new_date
        else:
            return self._fc.adjust(date_to_adjust)