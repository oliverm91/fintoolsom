from abc import ABC, abstractmethod
from datetime import date
import calendar

import numpy as np
    

def _gdc_dd(cls, sd, ed):
    return cls._get_day_count(sd, ed)
def _gdc_dl(cls, sd, eds):
    return np.array([cls._get_day_count(sd, ed) for ed in eds])
def _gdc_ld(cls, sds, ed):
    return np.array([cls._get_day_count(sd, ed) for sd in sds])
def _gdc_ll(cls, sds, eds):
    if len(sds) == len(eds):
        return np.array([cls._get_day_count(sd, ed) for sd, ed in zip(sds, eds)])
    else:
        raise ValueError(f'Length of start_dates and end_dates must match. Got lengths {len(sds)} and {len(eds)}')

class DayCountConventionBase(ABC):

    _method_mapper = {
        (date, date): _gdc_dd,
        (date, list): _gdc_dl,
        (list, date): _gdc_ld,
        (list, list): _gdc_ll
    }
    @staticmethod
    @abstractmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        pass

    @classmethod
    def get_day_count(cls, start_date: date | list[date], end_date: date | list[date]) -> int | np.ndarray:
        try:
            return cls._method_mapper[(type(start_date), type(end_date))](cls, start_date, end_date)
        except KeyError as _:
            raise TypeError(f'Input types start_date: {type(start_date)} and end_date: {type(end_date)} are not mapped. Mapped values:\n{list(cls._method_mapper.keys())}')

    @classmethod
    def get_time_fraction(cls, start_date: date | list[date], end_date: date | list[date], time_fraction_base: int) -> float | np.ndarray:
        days = cls.get_day_count(start_date, end_date)
        return days/time_fraction_base
    
    @abstractmethod
    def _strfy() -> str:
        pass
    
    @classmethod
    def __str__(cls) -> str:
        return cls._strfy()


class ActualDayCountConvention(DayCountConventionBase):
    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        return (end_date - start_date).days
    
    @staticmethod
    def _strfy():
        return 'act'
    

class Days30ADayCountConvention(DayCountConventionBase):
    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        d1, d2 = start_date.day, end_date.day
        m1, m2 = start_date.month, end_date.month
        y1, y2 = start_date.year, end_date.year
            
        d1 = min(d1, 30)
        d2 = min(d2, 30) if d1 > 29 else d2
        count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return count

    @staticmethod
    def _strfy():
        return '30a'

class Days30UDayCountConvention(DayCountConventionBase):
    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
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
    
    @staticmethod
    def _strfy():
        return '30u'


class Days30EDayCountConvention(DayCountConventionBase):
    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        d1, d2 = start_date.day, end_date.day
        m1, m2 = start_date.month, end_date.month
        y1, y2 = start_date.year, end_date.year

        d1 = min(d1, 30)
        d2 = min(d2, 30)

        count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return count
    
    @staticmethod
    def _strfy():
        return '30e'


class Days30EISDADayCountConvention(DayCountConventionBase):
    @staticmethod
    def _get_day_count(start_date: date, end_date: date) -> int:
        d1, d2 = start_date.day, end_date.day
        m1, m2 = start_date.month, end_date.month
        y1, y2 = start_date.year, end_date.year
        start_date_month_end_day = calendar.monthrange(y1, m1)[1]
        end_date_month_end_day = calendar.monthrange(y2, m2)[1]
        if d1 == start_date_month_end_day:
            d1 = 30
        if d2 == end_date_month_end_day and m2 != 2:
            d2 = 30
        count = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return count
    
    @staticmethod
    def _strfy():
        return '30eISDA'