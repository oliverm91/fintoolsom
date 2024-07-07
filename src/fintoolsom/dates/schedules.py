from datetime import date
from typing import Iterable
from dateutil.relativedelta import relativedelta
from enum import Enum

from .calendars import Calendar
from .adjustments import AdjustmentDateConvention


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