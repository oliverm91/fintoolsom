from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from .adjustments import AdjustmentDateConventionBase
from .schedules import _add_months


class TermUnit(Enum):
    D = "D"
    W = "W"
    M = "M"
    Y = "Y"


@dataclass
class Term:
    """A maturity tenor with its own adjustment convention.

    Carries unit and value explicitly so callers never parse strings.
    adj_convention determines how the raw maturity date is adjusted
    (e.g., ModifiedFollowing on a joint calendar for XCCY swaps).
    """

    value: int
    unit: TermUnit
    adj_convention: AdjustmentDateConventionBase

    def advance(self, from_date: date) -> date:
        """Return from_date + this tenor, adjusted by adj_convention."""
        if self.unit == TermUnit.W:
            raw = from_date + timedelta(days=7 * self.value)
        elif self.unit == TermUnit.M:
            raw = _add_months(from_date, self.value)
        elif self.unit == TermUnit.Y:
            raw = _add_months(from_date, 12 * self.value)
        else:  # D
            raw = self.adj_convention.calendar.add_business_days(from_date, self.value)
        return self.adj_convention.adjust(raw)

    def __str__(self) -> str:
        return f"{self.value}{self.unit.value}"
