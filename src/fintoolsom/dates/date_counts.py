from datetime import date
from enum import Enum
import calendar
from dateutil.relativedelta import relativedelta
from typing import Iterable
import numpy as np
from multimethod import multimethod

from .calendars import Calendar
from .adjustments import AdjustmentDateConvention


class DayCountConvention(Enum):
    Actual = 'actual'
    Days30A = 'days 30a'
    Days30U = 'days 30u'
    Days30E = 'days 30e'
    Days30E_ISDA = 'days 30e isda'
    BUS_DAYS = 'business days'
    

def add_tenor(date: date, tenor: str, holidays: Iterable[date]=None, adj_convention: AdjustmentDateConvention=None, calendar: Calendar=None) -> date:
    tenor = tenor.replace('/', '').lower()
    tenor = tenor.replace('on', '1D')
    tenor = tenor.replace('tn', '2D')
    tenor_unit = tenor[-1:]
    adding_units = int(tenor[:-1])
    if tenor_unit == 'd':
        if calendar is None:
            holidays = [] if holidays is None else holidays
            calendar = Calendar(custom_holidays=holidays)
        end_date = calendar.add_business_days(date, adding_units, holidays=holidays)
        return end_date
    elif tenor_unit == 'w':
        days_to_add = 7 * adding_units
        end_date = date + relativedelta(days=days_to_add)
    elif tenor_unit in ('m', 'y'):
        month_mult = 1 if tenor_unit == 'm' else 12
        adding_months = int(adding_units * month_mult)
        end_date = date + relativedelta(months=adding_months)
    else:
        raise NotImplementedError(f'Tenor unit {tenor_unit} not implemented. Only d, m, y are accepted.')
        
    end_date = adj_convention.adjust(end_date)
    return end_date


@multimethod
def _get_day_count_actual(start_date: date, end_date: date) -> int:
    return (end_date - start_date).days

@multimethod
def _get_day_count_actual(start_date: date, end_date: Iterable[date]) -> np.ndarray:
    end_date_np = np.array(end_date)
    count = (end_date_np - start_date).astype('timedelta64[D]')/np.timedelta64(1, 'D')
    return count

@multimethod
def _get_day_count_actual(start_date: Iterable[date], end_date: Iterable[date]) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    start_date_np = np.array(start_date)
    end_date_np = np.array(end_date)
    count = (end_date_np - start_date_np).astype('timedelta64[D]')/np.timedelta64(1, 'D')
    return count

@multimethod
def _get_day_count_actual(start_date: Iterable[date], end_date: date) -> np.ndarray:
    start_date_np = np.array(start_date)
    count = (end_date - start_date_np).astype('timedelta64[D]')/np.timedelta64(1, 'D')
    return count

@multimethod
def _get_day_count_30a(start_date: date, end_date: date) -> int:
    d1, d2 = start_date.day, end_date.day
    m1, m2 = start_date.month, end_date.month
    y1, y2 = start_date.year, end_date.year
        
    d1 = min(d1, 30)
    d2 = min(d2, 30) if d1 > 29 else d2
    count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return count

@multimethod
def _get_day_count_30a(start_date: Iterable[date], end_date: Iterable[date]) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    count = [_get_day_count_30a(sd, ed) for sd, ed in zip(start_date, end_date)]
    return np.array(count)

@multimethod
def _get_day_count_30u(start_date: date, end_date: date) -> int:
    d1, d2 = start_date.day, end_date.day
    m1, m2 = start_date.month, end_date.month
    y1, y2 = start_date.year, end_date.year
    start_date_month_info = calendar.monthrange(start_date.year, start_date.month)
    start_date_month_end_day = start_date_month_info[1]
    end_date_month_info = calendar.monthrange(end_date.year, end_date.month)
    end_date_month_end_day = end_date_month_info[1]

    is_eom = d2 == end_date_month_info[1]
    start_date_last_day_of_february = start_date.month == 2 and d1 == start_date_month_end_day
    end_date_last_day_of_february = end_date.month == 2 and d2 == end_date_month_end_day
    if is_eom and start_date_last_day_of_february and end_date_last_day_of_february:
        d2 = 30
    if is_eom and start_date_last_day_of_february:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    if d1 == 31:
        d1 = 30

    count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return count

@multimethod
def _get_day_count_30u(start_date: Iterable[date], end_date: Iterable[date]) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    count = [_get_day_count_30u(sd, ed) for sd, ed in zip(start_date, end_date)]
    return np.array(count)

@multimethod
def _get_day_count_30e(start_date: date, end_date: date) -> int:
    d1, d2 = start_date.day, end_date.day
    m1, m2 = start_date.month, end_date.month
    y1, y2 = start_date.year, end_date.year

    d1 = min(d1, 30)
    d2 = min(d1, 30)

    count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return count

@multimethod
def _get_day_count_30e(start_date: Iterable[date], end_date: Iterable[date]) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    count = [_get_day_count_30e(sd, ed) for sd, ed in zip(start_date, end_date)]
    return np.array(count)

@multimethod
def _get_day_count_30e_isda(start_date: date, end_date: date) -> int:
    d1, d2 = start_date.day, end_date.day
    m1, m2 = start_date.month, end_date.month
    y1, y2 = start_date.year, end_date.year
    start_date_month_info = calendar.monthrange(start_date.year, start_date.month)
    start_date_month_end_day = start_date_month_info[1]
    end_date_month_info = calendar.monthrange(end_date.year, end_date.month)
    end_date_month_end_day = end_date_month_info[1]
    if d1 == start_date_month_end_day:
        d1 = 30
    if d2 == end_date_month_end_day and m2 != 2:
        d2 = 30
    count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return count

@multimethod
def _get_day_count_30e_isda(start_date: Iterable[date], end_date: Iterable[date]) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    count = [_get_day_count_30e_isda(sd, ed) for sd, ed in zip(start_date, end_date)]
    return np.array(count)

_day_count_router = {
    DayCountConvention.Actual: _get_day_count_actual,
    DayCountConvention.Days30A: _get_day_count_30a,
    DayCountConvention.Days30E: _get_day_count_30e,
    DayCountConvention.Days30U: _get_day_count_30u,
    DayCountConvention.Days30E_ISDA: _get_day_count_30e_isda    
}
_day_count_cache: dict[str, np.ndarray | int] = {}
def get_day_count(start_date: Iterable[date] | date, end_date: Iterable[date] | date, day_count_convention: DayCountConvention) -> np.ndarray | int:
    hashable_input = str(start_date)+str(end_date)+str(day_count_convention.value)
    if hashable_input in _day_count_cache:
        return _day_count_cache[hashable_input]
    days = _day_count_router[day_count_convention](start_date, end_date)
    _day_count_cache[hashable_input] = days
    return days

def get_time_fraction(start_date: Iterable[date] | date, end_date: Iterable[date] | date, day_count_convention: DayCountConvention, base_convention: int=360) -> np.ndarray | float:
    day_count = get_day_count(start_date, end_date, day_count_convention)
    time_fraction = day_count / base_convention
    return time_fraction