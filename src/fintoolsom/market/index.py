from dataclasses import dataclass
from datetime import date
from typing import Self

from ..rates import Rate
from ..dates import Tenor, AdjustmentDateConventionBase, FollowingConvention, ModifiedFollowingConvention, PrecedingConvention, ModifiedPrecedingConvention


@dataclass
class Index:
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

@dataclass
class RateIndex(Index):
    '''
    Examples: Ibor rates, SOFR, TPM, EFFR
    '''
    rate: Rate
    tenor: str

    def copy(self) -> Self:
        return RateIndex(self.name, self.rate.copy())
    
    def __str__(self) -> str:
        return f'{self.name}:{self.rate.rate_value}'


@dataclass
class ValuedRateIndex(Index):
    name: str
    value: float

    def copy(self) -> Self:
        return ValuedRateIndex(self.name, self.value)
    
    def __str__(self) -> str:
        return f'{self.name}:{self.value}'


@dataclass(slots=True)
class RateIndexData:
    name: str
    historic_data: dict[date, RateIndex]

    def __post_init__(self):
        for ri in self.historic_data.values():
            if ri.name!=self.name:
                raise ValueError(f'All values in historic_data of RateIndex object must match RateIndexData object name. RateIndexData object name: {self.name}, found {ri.name} in historic_data')
            
    def add_date(self, t: date, rate_index: RateIndex):
        self.historic_data[t] = rate_index

    def get_rate_index(self, t: date) -> RateIndex:
        if t in self.historic_data:
            return self.historic_data[t]
        else:
            raise KeyError(f'Date {t} not found in {self.name} data.')