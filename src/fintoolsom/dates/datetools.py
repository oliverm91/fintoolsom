from datetime import timedelta, date, datetime
from enum import Enum
import calendar
from collections.abc import Sequence
from typing import Union, get_args
import numpy as np
from multimethod import multimethod


class AdjustmentDateConvention(Enum):
    Following = 'following'
    ModifiedFollowing = 'modified following'
    Preceding = 'preceding'
    ModifiedPreceding = 'modified preceding'
    
class DayCountConvention(Enum):
    Actual = 'actual'
    Days30A = 'days 30a'
    Days30U = 'days 30u'
    Days30E = 'days 30e'
    Days30E_ISDA = 'days 30e isda'
    BUS_DAYS = 'business days'
    
def is_business_date(date: date, holidays: set=None) -> bool:
    if holidays is None:
        holidays = set()
    return date.weekday() <= 4 and date not in holidays
    
def add_business_days(date: date, days: int, holidays:set=None, adj_convention: AdjustmentDateConvention=AdjustmentDateConvention.Following) -> date:
    days_to_add = abs(days)
    while days_to_add > 0:
        sign = int(abs(days)/days)
        date += timedelta(days=1*sign)      
        if is_business_date(date, holidays=holidays):
            days_to_add -= 1
    return date

def following(date: date, holidays: set=None) -> date:
    while not is_business_date(date, holidays=holidays):
        date += timedelta(days=1)
    return date

def modified_following(date: date, holidays: set=None) -> date:
    date2 = date
    while not is_business_date(date, holidays=holidays):
        date2 += timedelta(days=1)
    if date2.month != date.month:
        preceding(date, holidays=holidays)
    return date

def preceding(date: date, holidays: set=None) -> date:
    while not is_business_date(date, holidays=holidays):
        date -= timedelta(days=1)
    return date

def modified_preceding(date: date, holidays: set=None) -> date:
    date2 = date
    while not is_business_date(date, holidays=holidays):
        date -= timedelta(days=1)
    if date2.month != date.month:
        following(date, holidays=holidays)
    return date

def adjust_date(date: date, holidays: set=None, adj_convention: AdjustmentDateConvention=AdjustmentDateConvention.Following) -> date:
    if adj_convention==AdjustmentDateConvention.Following:
        date = following(date, holidays=holidays)
    elif adj_convention==AdjustmentDateConvention.Preceding:
        date = preceding(date, holidays=holidays)
    elif adj_convention==AdjustmentDateConvention.ModifiedFollowing:
        date = modified_following(date, holidays=holidays)
    elif adj_convention==AdjustmentDateConvention.ModifiedPreceding:
        date = modified_preceding(date, holidays=holidays)
    elif adj_convention is None:
        return date
    else:
        raise NotImplementedError(f'Adjustment Date Convention {adj_convention} has no implemented method.')
    return date

def add_months(date: date, months: int) -> date:
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, calendar.monthrange(year, month)[1])
    return date.replace(year=year, month=month, day=day)

def add_tenor(date: date, tenor: str, holidays: list=[], adj_convention: AdjustmentDateConvention=None) -> date:
    tenor = tenor.replace('/', '')
    tenor = tenor.replace('ON', '1D')
    tenor = tenor.replace('TN', '2D')
    tenor_unit = tenor[-1:].lower()
    adding_units = int(tenor[:-(len(tenor)-1)])
    if tenor_unit == 'd':
        end_date = date + add_business_days(date, adding_units, holidays=holidays)
        return end_date
    elif tenor_unit == 'w':
        tenor = str(7 * adding_units) + 'd'
        end_date = add_tenor(date, tenor, holidays=holidays, adj_convention=adj_convention)
    elif tenor_unit in ('m', 'y'):
        month_mult = 1 if tenor_unit == 'm' else 12
        adding_months = int(adding_units * month_mult)
        end_date = add_months(date, adding_months)
    else:
        raise NotImplementedError(f'Tenor unit {tenor_unit} not implemented. Only d, m, y are accepted.')
        
    end_date = adjust_date(date, adj_convention=adj_convention)
    return end_date

@multimethod
def _get_day_count_actual(start_date: date, end_date: date, holidays: set=None) -> int:
    return (end_date - start_date).days

@multimethod
def _get_day_count_actual(start_date: date, end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    end_date_np = np.array(end_date)
    count = (end_date_np - start_date).astype('timedelta64[D]')/np.timedelta64(1, 'D')
    return count

@multimethod
def _get_day_count_actual(start_date: Union[Sequence, np.ndarray], end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    start_date_np = np.array(start_date)
    end_date_np = np.array(end_date)
    count = (end_date_np - start_date_np).astype('timedelta64[D]')/np.timedelta64(1, 'D')
    return count

@multimethod
def _get_day_count_actual(start_date: Union[Sequence, np.ndarray], end_date: date, holidays: set=None) -> np.ndarray:
    start_date_np = np.array(start_date)
    count = (end_date - start_date_np).astype('timedelta64[D]')/np.timedelta64(1, 'D')
    return count

@multimethod
def _get_day_count_bus_days(start_date: date, end_date: date, holidays: set=None) -> int:
    count = 0
    if start_date > end_date:
        raise ValueError(f'Start date {start_date} must be before end date {end_date}.')
    ref_date = start_date
    while ref_date < end_date:
        ref_date += timedelta(days=1)
        if is_business_date(ref_date, holidays):
            count += 1
    return count

@multimethod
def _get_day_count_bus_days(start_date: date, end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    count = [_get_day_count_bus_days(start_date, ed, holidays) for ed in end_date]
    return np.array(count)

@multimethod
def _get_day_count_bus_days(start_date: Union[Sequence, np.ndarray], end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    count = [_get_day_count_bus_days(sd, ed, holidays) for sd, ed in zip(start_date, end_date)]
    return np.array(count)

@multimethod
def _get_day_count_bus_days(start_date: Union[Sequence, np.ndarray], end_date: date, holidays: set=None) -> np.ndarray:
    count = [_get_day_count_bus_days(sd, end_date, holidays) for sd in start_date]
    return np.array(count)

@multimethod
def _get_day_count_30a(start_date: date, end_date: date, holidays: set=None) -> int:
    d1, d2 = start_date.day, end_date.day
    m1, m2 = start_date.month, end_date.month
    y1, y2 = start_date.year, end_date.year
        
    d1 = min(d1, 30)
    d2 = min(d2, 30) if d1 > 29 else d2
    count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return count

@multimethod
def _get_day_count_30a(start_date: date, end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    count = [_get_day_count_30a(start_date, ed, holidays) for ed in end_date]
    return np.array(count)

@multimethod
def _get_day_count_30a(start_date: Union[Sequence, np.ndarray], end_date: date, holidays: set=None) -> np.ndarray:
    count = [_get_day_count_30a(sd, end_date, holidays) for sd in start_date]
    return np.array(count)

@multimethod
def _get_day_count_30a(start_date: Union[Sequence, np.ndarray], end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    count = [_get_day_count_30a(sd, ed, holidays) for sd, ed in zip(start_date, end_date)]
    return np.array(count)

@multimethod
def _get_day_count_30u(start_date: date, end_date: date, holidays: set=None) -> int:
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
def _get_day_count_30u(start_date: date, end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    count = [_get_day_count_30u(start_date, ed, holidays) for ed in end_date]
    return np.array(count)

@multimethod
def _get_day_count_30u(start_date: Union[Sequence, np.ndarray], end_date: date, holidays: set=None) -> np.ndarray:
    count = [_get_day_count_30u(sd, end_date, holidays) for sd in start_date]
    return np.array(count)

@multimethod
def _get_day_count_30u(start_date: Union[Sequence, np.ndarray], end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    count = [_get_day_count_30u(sd, ed, holidays) for sd, ed in zip(start_date, end_date)]
    return np.array(count)

@multimethod
def _get_day_count_30e(start_date: date, end_date: date, holidays: set=None) -> int:
    d1, d2 = start_date.day, end_date.day
    m1, m2 = start_date.month, end_date.month
    y1, y2 = start_date.year, end_date.year

    d1 = min(d1, 30)
    d2 = min(d1, 30)

    count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return count

@multimethod
def _get_day_count_30e(start_date: date, end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    count = [_get_day_count_30e(start_date, ed, holidays) for ed in end_date]
    return np.array(count)

@multimethod
def _get_day_count_30e(start_date: Union[Sequence, np.ndarray], end_date: date, holidays: set=None) -> np.ndarray:
    count = [_get_day_count_30e(sd, end_date, holidays) for sd in start_date]
    return np.array(count)

@multimethod
def _get_day_count_30e(start_date: Union[Sequence, np.ndarray], end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    count = [_get_day_count_30e(sd, ed, holidays) for sd, ed in zip(start_date, end_date)]
    return np.array(count)

@multimethod
def _get_day_count_30e_isda(start_date: date, end_date: date, holidays: set=None) -> int:
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
def _get_day_count_30e_isda(start_date: date, end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    count = [_get_day_count_30e_isda(start_date, ed, holidays) for ed in end_date]
    return np.array(count)

@multimethod
def _get_day_count_30e_isda(start_date: Union[Sequence, np.ndarray], end_date: date, holidays: set=None) -> np.ndarray:
    count = [_get_day_count_30e_isda(sd, end_date, holidays) for sd in start_date]
    return np.array(count)

@multimethod
def _get_day_count_30e_isda(start_date: Union[Sequence, np.ndarray], end_date: Union[Sequence, np.ndarray], holidays: set=None) -> np.ndarray:
    if len(start_date) != len(end_date):
        raise ValueError(f'Start and end dates must have the same length. Start date length: {len(start_date)}, end date length: {len(end_date)}')
    count = [_get_day_count_30e_isda(sd, ed, holidays) for sd, ed in zip(start_date, end_date)]
    return np.array(count)

_day_count_router = {
    DayCountConvention.Actual: _get_day_count_actual,
    DayCountConvention.BUS_DAYS: _get_day_count_bus_days,
    DayCountConvention.Days30A: _get_day_count_30a,
    DayCountConvention.Days30E: _get_day_count_30e,
    DayCountConvention.Days30U: _get_day_count_30u,
    DayCountConvention.Days30E_ISDA: _get_day_count_30e_isda    
}

def get_day_count(start_date: Union[Sequence, np.ndarray, date], end_date: Union[Sequence, np.ndarray, date], day_count_convention: DayCountConvention, holidays: set=None) -> Union[np.ndarray, int]:    
    func = _day_count_router[day_count_convention]
    return func(start_date, end_date, holidays)

def get_time_fraction(start_date: Union[Sequence, np.ndarray, date], end_date: Union[Sequence, np.ndarray, date], day_count_convention: DayCountConvention, base_convention: float=360.0) -> Union[np.ndarray, float]:
    day_count = get_day_count(start_date, end_date, day_count_convention)
    time_fraction = day_count / base_convention
    return time_fraction