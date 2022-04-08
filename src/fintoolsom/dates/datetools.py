from datetime import timedelta, date, datetime
from dateutil.relativedelta import relativedelta
from enum import Enum
import calendar
from collections.abc import Sequence
from typing import Union, get_args
import numpy as np


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
    
def is_business_date(date: date, holidays: list=[]) -> bool:
    return date.weekday() <= 4 and date not in holidays
    
def add_business_days(date: date, days: int, holidays:list=[], adj_convention: AdjustmentDateConvention=AdjustmentDateConvention.Following) -> date:
    days_to_add = abs(days)
    while days_to_add > 0:
        sign = int(abs(days)/days)
        date += timedelta(days=1*sign)      
        if is_business_date(date, holidays=holidays):
            days_to_add -= 1
    return date

def following(date: date, holidays: list=[]) -> date:
    while not is_business_date(date, holidays=holidays):
        date += timedelta(days=1)
    return date

def modified_following(date: date, holidays: list=[]) -> date:
    date2 = date
    while not is_business_date(date, holidays=holidays):
        date2 += timedelta(days=1)
    if date2.month != date.month:
        preceding(date, holidays=holidays)
    return date

def preceding(date: date, holidays: list=[]) -> date:
    while not is_business_date(date, holidays=holidays):
        date -= timedelta(days=1)
    return date

def modified_preceding(date: date, holidays: list=[]) -> date:
    date2 = date
    while not is_business_date(date, holidays=holidays):
        date -= timedelta(days=1)
    if date2.month != date.month:
        following(date, holidays=holidays)
    return date

def adjust_date(date: date, holidays: list=[], adj_convention: AdjustmentDateConvention=AdjustmentDateConvention.Following) -> date:
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
        raise Exception(f'Adjustment Date Convention {adj_convention} has no implemented method.')
    return date

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
        end_date = date + relativedelta(months=adding_months)
    else:
        raise Exception(f'Tenor unit {tenor_unit} not implemented. Only d, m, y are accepted.')
        
    end_date = adjust_date(date, adj_convention=adj_convention)
    return end_date

def get_business_day_count(start_date: date, end_date: date, holidays: list=[]) -> int:
    count = 0
    if start_date >= end_date:
        return 0
    ref_date = start_date
    while ref_date < end_date:
        ref_date += timedelta(days=1)
        if is_business_date(ref_date, holidays):
            count += 1    
    return count

def get_day_count(start_date: Union[Sequence, date], end_date: Union[Sequence, date], day_count_convention: DayCountConvention, holidays: list=[]) -> Union[np.ndarray, int]:
    if isinstance(start_date, get_args(Union[Sequence, np.ndarray])) and isinstance(end_date, get_args(Union[Sequence, np.ndarray])):
        if len(start_date) == len(end_date):
            start_end_dates = zip(start_date, end_date)
            result = np.array([get_day_count(s, e, day_count_convention) for s, e in start_end_dates])
            return result
        else:
            raise Exception(f'start_date vector length {len(start_date)} and end_date vector length {len(end_date)} mismatch.')
    if isinstance(start_date, get_args(Union[Sequence, np.ndarray])) and type(end_date)==date:
        end_date = [end_date for i in range(len(start_date))]
        return get_day_count(start_date, end_date, day_count_convention)
    if isinstance(end_date, get_args(Union[Sequence, np.ndarray])) and type(start_date)==date:
        start_date = [start_date for i in range(len(end_date))]
        return get_day_count(start_date, end_date, day_count_convention)
    if not (isinstance(start_date, date) and isinstance(end_date, date)):
        raise Exception(f'start_date and end_date must be date objects. start_date: {start_date} {type(start_date)}, end_date: {end_date} {type(end_date)}')
    
    if day_count_convention == DayCountConvention.Actual:
        if type(start_date)==np.datetime64:
            ts = (start_date - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            start_date = datetime.utcfromtimestamp(ts)
        if type(end_date)==np.ndarray:
            if end_date.dtype==np.dtype('<M8[D]') and type(start_date)==date:
                count = (end_date.astype('O') - start_date).astype(np.timedelta64) / np.timedelta(1, 'D')
        else:
            if type(end_date)==date and type(start_date)==date:
                count = (end_date - start_date).days
            elif type(end_date)==np.datetime64 and type(start_date)==date:
                count = (end_date - np.datetime64(start_date)) / np.timedelta64(1, 'D')
            else:
                raise Exception(f'Unhandled case. End date: {type(end_date)}, Start date: {type(start_date)}.')
                
    elif day_count_convention == DayCountConvention.BUS_DAYS:
        count = get_business_day_count(start_date, end_date, holidays)
    else:
        d1, d2 = start_date.day, end_date.day
        m1, m2 = start_date.month, end_date.month
        y1, y2 = start_date.year, end_date.year
        start_date_month_info = calendar.monthrange(start_date.year, start_date.month)
        start_date_month_end_day = start_date_month_info[1]
        end_date_month_info = calendar.monthrange(end_date.year, end_date.month)
        end_date_month_end_day = end_date_month_info[1]
        
        if day_count_convention == DayCountConvention.Days30A:
            d1 = min(d1, 30)
            d2 = min(d2, 30) if d2 > 29 else d2 
        elif day_count_convention == DayCountConvention.Days30U:
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
        elif day_count_convention == DayCountConvention.Days30E:
            if d1 == 31:
                d1 = 30
            if d2 == 31:
                d2 = 30
        elif day_count_convention == DayCountConvention.Days30E_ISDA:
            if d1 == start_date_month_end_day:
                d1 = 30
            if d2 == end_date_month_end_day and m2 != 2:
                d2 = 30
        else:
            raise Exception(f'DayCountConvention {day_count_convention} not impletmented.')
        count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return count

def get_time_fraction(start_date: Union[Sequence, date], end_date: Union[Sequence, date], day_count_convention: DayCountConvention, base_convention: float=360.0) -> Union[np.ndarray, float]:
    day_count = get_day_count(start_date, end_date, day_count_convention)
    time_fraction = day_count / base_convention
    return time_fraction