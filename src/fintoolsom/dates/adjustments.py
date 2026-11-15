from abc import ABC, abstractmethod
from datetime import date

from .. import settings
from .calendars import Calendar, resolve_calendar


class AdjustmentDateConventionBase(ABC):
    def __init__(self, calendar: Calendar | str | None = None):
        # A country code (or None, weekends only) resolves to the shared calendar.
        self.calendar = resolve_calendar(calendar)

    def _native_kind(self, d) -> int | None:
        """This convention's kind for the C++ calendar (``CalendarCore.adjust``), or None
        to adjust in Python: the ``"python"`` backend of ``"dates"``, a date that isn't a
        ``date``, a convention or calendar subclass."""
        if (
            settings.is_cpp("dates")
            and type(d) is date
            and type(self.calendar) is Calendar
        ):
            return _KINDS.get(type(self))
        return None

    @abstractmethod
    def adjust(self, date_to_adjust: date) -> date:
        pass


class FollowingConvention(AdjustmentDateConventionBase):
    def adjust(self, date_to_adjust: date) -> date:
        if (kind := self._native_kind(date_to_adjust)) is not None:
            return self.calendar._native_core().adjust(kind, date_to_adjust)
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
        if (kind := self._native_kind(date_to_adjust)) is not None:
            return self.calendar._native_core().adjust(kind, date_to_adjust)
        new_date = self._fc.adjust(date_to_adjust)
        if new_date.month == date_to_adjust.month:
            return new_date
        else:
            return self._pc.adjust(date_to_adjust)


class PrecedingConvention(AdjustmentDateConventionBase):
    def adjust(self, date_to_adjust: date) -> date:
        if (kind := self._native_kind(date_to_adjust)) is not None:
            return self.calendar._native_core().adjust(kind, date_to_adjust)
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
        if (kind := self._native_kind(date_to_adjust)) is not None:
            return self.calendar._native_core().adjust(kind, date_to_adjust)
        new_date = self._pc.adjust(date_to_adjust)
        if new_date.month == date_to_adjust.month:
            return new_date
        else:
            return self._fc.adjust(date_to_adjust)


# The kinds of fintoolsom._native (CalendarCore.adjust, generate_schedule), by class.
_KINDS: dict[type, int] = {
    FollowingConvention: 0,
    ModifiedFollowingConvention: 1,
    PrecedingConvention: 2,
    ModifiedPrecedingConvention: 3,
}
