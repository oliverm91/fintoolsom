from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from dateutil.relativedelta import relativedelta

from .adjustments import AdjustmentDateConventionBase


class TermUnit(Enum):
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
            raw = from_date + relativedelta(weeks=self.value)
        elif self.unit == TermUnit.M:
            raw = from_date + relativedelta(months=self.value)
        else:  # Y
            raw = from_date + relativedelta(years=self.value)
        return self.adj_convention.adjust(raw)

    def __str__(self) -> str:
        return f"{self.value}{self.unit.value}"
