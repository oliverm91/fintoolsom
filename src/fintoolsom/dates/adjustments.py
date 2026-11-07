from abc import ABC, abstractmethod
from datetime import date

from .calendars import Calendar, resolve_calendar


class AdjustmentDateConventionBase(ABC):
    def __init__(self, calendar: Calendar | str | None = None):
        # A country code (or None, weekends only) resolves to the shared calendar.
        self.calendar = resolve_calendar(calendar)

    @abstractmethod
    def adjust(self, date_to_adjust: date) -> date:
        pass


class FollowingConvention(AdjustmentDateConventionBase):
    def adjust(self, date_to_adjust: date) -> date:
        if self.calendar.is_holiday(date_to_adjust):
            return self.calendar.add_business_days(date_to_adjust, 1)
        else:
            return date_to_adjust


class ModifiedFollowingConvention(AdjustmentDateConventionBase):
    def __init__(self, calendar: Calendar | str | None = None):
        super().__init__(calendar)
        self._fc = FollowingConvention(self.calendar)
        self._pc = PrecedingConvention(self.calendar)

    def adjust(self, date_to_adjust: date) -> date:
        new_date = self._fc.adjust(date_to_adjust)
        if new_date.month == date_to_adjust.month:
            return new_date
        else:
            return self._pc.adjust(date_to_adjust)


class PrecedingConvention(AdjustmentDateConventionBase):
    def adjust(self, date_to_adjust: date) -> date:
        if self.calendar.is_holiday(date_to_adjust):
            return self.calendar.add_business_days(date_to_adjust, -1)
        else:
            return date_to_adjust


class ModifiedPrecedingConvention(AdjustmentDateConventionBase):
    def __init__(self, calendar: Calendar | str | None = None):
        super().__init__(calendar)
        self._fc = FollowingConvention(self.calendar)
        self._pc = PrecedingConvention(self.calendar)

    def adjust(self, date_to_adjust: date) -> date:
        new_date = self._pc.adjust(date_to_adjust)
        if new_date.month == date_to_adjust.month:
            return new_date
        else:
            return self._fc.adjust(date_to_adjust)
