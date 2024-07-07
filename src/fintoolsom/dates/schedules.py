from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable
from dateutil.relativedelta import relativedelta
from enum import Enum

from .calendars import Calendar
from .adjustments import AdjustmentDateConvention, ModifiedFollowingConvention

_def_cal = Calendar()
_def_adj_conv = ModifiedFollowingConvention(_def_cal)
@dataclass(slots=True)
class Tenor:
    str_tenor: str
    adj_conv: AdjustmentDateConvention = field(default=None)
    
    tenor_unit: str = field(init=False)
    tenor_value: int = field(init=False)
    _rel_mapper: dict = field(init=False)
    _get_umaturity_func: Callable[[date], date] = field(init=False)
    def __post_init__(self):
        if len(self.str_tenor) < 2:
            raise ValueError(f'str_tenor length must be at least 2. str_tenor received was {self.str_tenor}')
        
        if self.adj_conv is None:
            self.adj_conv = _def_adj_conv
        
        self.tenor_unit = self.str_tenor[-1].lower()
        self.tenor_value = int(self.str_tenor[:-1])
        self._rel_mapper = {}
        if self.tenor_unit=='d':
            def f(t: date) -> date:
                return self.adj_conv.calendar.add_business_days(t, self.tenor_value)
        elif self.tenor_unit=='m':
            self._rel_mapper['kwarg'] = 'months'
            self._rel_mapper['multiplier'] = 1
        elif self.tenor_unit=='y':
            self._rel_mapper['kwarg'] = 'months'
            self._rel_mapper['multiplier'] = 12
        elif self.tenor_unit=='w':
            self._rel_mapper['kwarg'] = 'days'
            self._rel_mapper['multiplier'] = 7
        else:
            raise ValueError(f'Last letter of str_tenor must be of D, W, M or Y. Got {self.tenor_unit}')
        
        if self.tenor_unit!='d':
            def f(t: date) -> date:
                kwargs = {self._rel_mapper['kwarg']: self._rel_mapper['multiplier'] * self.tenor_value}
                return t + relativedelta(**kwargs)
            
        self._get_umaturity_func = f

    def get_unadjusted_maturity(self, t: date) -> date:
        return self._get_umaturity_func(t)
    
    def get_adjusted_maturity(self, t: date) -> date:
        u_mat = self.get_unadjusted_maturity(t)
        return self.adj_conv.adjust(u_mat)