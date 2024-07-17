from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Self

from ..rates import Rate
from ..dates import Calendar


@dataclass
class Index:
    name: str


@dataclass
class IndexData(ABC):
    name: str
    
    @abstractmethod
    def get_accrued_interest(self, notional: float, start_date: date, end_date: date):
        pass

@dataclass
class OvernightIndexData(IndexData):
    index_values: dict[date, float] = field(default_factory=dict)
    overnight_rates: dict[date, Rate] = field(default_factory=dict)

    calendar: Calendar = field(default=None)
    def __post_init__(self):
        if self.calendar is None:
            self.calendar = Calendar()
        # If index_values dict is empty, fill it with artificial index starting from 100 at min(overnight_rates.keys())
        if len(self.index_values) == 0:
            if len(self.overnight_rates) == 0:
                raise ValueError("Either index_values or overnight_rates must be non-empty")
            start_t = min(self.overnight_rates.keys())
            max_t = max(self.overnight_rates.keys())
            self.index_values[start_t] = 100
            self._fill_index_period(self, start_t, max_t)

    def _fill_index_period(self, start_date: date, end_date: date, find_start_date: bool=False):
        if find_start_date:
            start_date = max(t for t in self.index_values.keys() if t <= start_date)
            self._fill_index_period(start_date, end_date, find_start_date=False)
            return
        while start_date < end_date:
            next_t = self.calendar.add_business_days(start_date, 1)
            adj_next = min(next_t, end_date)
            if adj_next not in self.index_values:
                if start_date in self.overnight_rates:
                    r = self.overnight_rates[start_date]
                self.index_values[adj_next] = self.index_values[start_date] * r.get_wealth_factor(start_date, adj_next)
            start_date = next_t

    def add_index_value(self, t: date, index_value: float):
        self.index_values[t] = index_value

    def add_rate(self, t: date, rate: Rate, fill_index_for_next_t: bool=False):
        self.overnight_rates[t] = rate
        if fill_index_for_next_t:
            next_t = self.calendar.add_business_days(t, 1)
            if next_t not in self.index_values:
                self.index_values[next_t] = self.index_values[t] * rate.get_wealth_factor(t, next_t)

    def get_accrued_interest(self, notional: float, start_date: date, end_date: date, fill_index_values_with_rates: bool=False) -> float:
        if start_date in self.index_values and end_date in self.index_values:
            return notional * (self.index_values[end_date] / self.index_values[start_date] - 1)
        
        if fill_index_values_with_rates:
            self._fill_index_period(start_date, end_date)
            return self.get_accrued_interest(notional, start_date, end_date, fill_index_values_with_rates=False)
        

@dataclass
class IborRateIndex(Index):
    '''
    Examples: Ibor rates
    '''
    rate: Rate

    def copy(self) -> Self:
        return IborRateIndex(self.name, self.rate.copy())
    
    def __str__(self) -> str:
        return f'{self.name}:{self.rate.rate_value}'


class IborRateIndexData(ABC):
    name: str
    historic_data: dict[date, IborRateIndex]

    def __post_init__(self):
        for ri in self.historic_data.values():
            if ri.name!=self.name:
                raise ValueError(f'All values in historic_data of RateIndex object must match RateIndexData object name. RateIndexData object name: {self.name}, found {ri.name} in historic_data')
            
    def add_date(self, t: date, rate_index: IborRateIndex):
        self.historic_data[t] = rate_index

    def get_rate_index(self, t: date) -> IborRateIndex:
        if t in self.historic_data:
            return self.historic_data[t]
        else:
            raise KeyError(f'Date {t} not found in {self.name} data.')
    
    def get_accrued_interest(self, notional: float, start_date: date, end_date: date, fixing_date: date=None) -> float:
        if fixing_date is None:
            fixing_date = start_date
        fixed_rate = self.get_rate_index(fixing_date)
        return fixed_rate.rate.get_accrued_interest(notional, start_date, end_date)