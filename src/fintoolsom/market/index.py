from dataclasses import dataclass, field
from datetime import date
from typing import Self

from ..rates import Rate

@dataclass
class Index:
    index_date: date
    value: float

    name: str = field(default=None)

    def accrue_rate(self, rate: Rate, t: date) -> Self:
        new_value = self.value + rate.get_accrued_interest(self.value, self.index_date, t)
        return Index(t, new_value, name=self.name)
    
    def accrue_rates(self, rates: list[Rate], dates: list[date], include_index_date: bool = False) -> list[Self]:
        new_indexes = [self]
        if len(rates) != len(dates):
            raise ValueError (f"Rates and dates must have the same length. Got {len(rates)} rates and {len(dates)} dates.")
        for ix, rate, t in enumerate(zip(rates, dates)):
            new_indexes.append(new_indexes[ix].accrue_rate(rate, t))

        if include_index_date:
            return new_indexes
        else:
            return new_indexes[1:]