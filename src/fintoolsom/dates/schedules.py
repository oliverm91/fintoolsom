from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from dateutil.relativedelta import relativedelta

from .calendars import Calendar
from .adjustments import AdjustmentDateConventionBase, ModifiedFollowingConvention

_def_cal = Calendar()
_def_adj_conv = ModifiedFollowingConvention(_def_cal)
@dataclass(slots=True)
class Tenor:
    str_tenor: str
    adj_conv: AdjustmentDateConventionBase = field(default=None)
    
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
            def f(t: date, forward: bool=True) -> date:
                return self.adj_conv.calendar.add_business_days(t, self.tenor_value * (1 if forward else -1))
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
            def f(t: date, forward: bool=True) -> date:
                kwargs = {self._rel_mapper['kwarg']: self._rel_mapper['multiplier'] * self.tenor_value * (1 if forward else -1)}
                return t + relativedelta(**kwargs)
            
        self._get_umaturity_func = f # TEST THAT F LIVES AFTER INIT

    def get_unadjusted_maturity(self, t: date, forward: bool=True) -> date:
        return self._get_umaturity_func(t, forward=forward)
    
    def get_adjusted_maturity(self, t: date, forward: bool=True) -> date:
        u_mat = self.get_unadjusted_maturity(t, forward=forward)
        return self.adj_conv.adjust(u_mat)
    

@dataclass
class ScheduleGenerator:
    @staticmethod
    def generate_schedule(start_date: date, maturity_tenor: str, frequency_tenor: str, adj_conv: AdjustmentDateConventionBase,
                          maturity_adj_conv: AdjustmentDateConventionBase=None, stub_first: bool=True, long_stub: bool=True) -> list[date]:
        frequency_tenor = frequency_tenor.lower()
        accepted_frequency_tenors = ['0', 0, '0m', '0y', '1m', '3m', '6m', '1y']
        if frequency_tenor not in accepted_frequency_tenors:
            raise ValueError(f'Frequency tenor accepted values are: {accepted_frequency_tenors}.')
        freq_tenor = Tenor(frequency_tenor, adj_conv)
        if maturity_adj_conv is None:
            maturity_adj_conv = adj_conv
        tenor = Tenor(maturity_tenor, maturity_adj_conv)
        umat = tenor.get_unadjusted_maturity(start_date)
        adj_mat = maturity_adj_conv.adjust(umat)

        if stub_first:
            starting_date = umat
            stop_date = start_date
            relativedelta_sign = -1
        else:
            starting_date = start_date
            stop_date = umat
            relativedelta_sign = 1

        u_schedule = [starting_date]

        just_in_case_counter = 0
        max_iterations = 10 + int((umat - start_date).days / 30)
        
        # This should make a list so that if stub_first is True then starting_date should end being <= stop_date (final to start). Else it stops being >= stop_date (start to final)
        # Check for EOM adj. If start_date== march/31 and maturity tenor is 3M, maturity will be june/30. If freq is 3M or 1M, last added date back to front will be march/30. It should be adjusted.
        while (starting_date > stop_date) == (relativedelta_sign==-1):
            starting_date = freq_tenor.get_unadjusted_maturity(starting_date, forward=(1==relativedelta_sign))
            u_schedule.append(starting_date)
            just_in_case_counter += 1
            if just_in_case_counter>max_iterations:
                raise Exception(f'Could not generate a schedule. Max iterations reached ({max_iterations}). start_date: {start_date}, maturity_tenor: {maturity_tenor}, frequency_tenor: {frequency_tenor}, stub_first: {stub_first}, long_stub: {long_stub}. Unadjusted Schedule list was. {u_schedule}')


        u_schedule.sort()
        if stop_date!=starting_date: # If is the same, no need for stub adjusting.
            if stub_first:
                u_schedule = u_schedule[1:] # removes date past starting_date
                u_schedule.insert(0, start_date)
                if len(u_schedule)>2: # If it is 2. It should be starting_date and umat.
                    if long_stub:
                        u_schedule.pop(1) # Removes second date so tenor between u_schedule[0] and u_schedule[1] is long (long_stub)
            else:
                u_schedule = u_schedule[:-1] # removes date past umat
                if len(u_schedule)>2: # If it is 2. It should be starting_date and umat.
                    if long_stub:
                        u_schedule.pop(-2) # Removes penultimate date so tenor between u_schedule[-2] and u_schedule[-1] is long (long_stub)

        adj_schedule = [u_date for u_date in u_schedule[:-1]]
        adj_schedule.append(adj_mat)

        return adj_schedule