from abc import ABC, abstractmethod
import calendar
from dataclasses import dataclass, field
from datetime import date


@dataclass
class TimeFractionBase(ABC):

    @abstractmethod
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        pass


    @abstractmethod
    def get_coupon_factor(self, start_date: date, end_date: date) -> float:
        pass


@dataclass
class TF_30_360Base(TimeFractionBase):
    def get_day_count_factor(self, start_accrual_date: date, end_accrual_date: date) -> float:
        d1, m1, y1 = self.get_dmy(start_accrual_date)
        d2, m2, y2 = self.get_dmy(end_accrual_date)
        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1) / 30

    def get_coupon_factor(self, start_accrual_date: date, payment_date: date) -> float:
        d1, m1, y1 = self.get_dmy(start_accrual_date)
        d3, m3, y3 = self.get_dmy(payment_date)
        return 360 * (y3 - y1) + 30 * (m3 - m1) + (d3 - d1) / 30
    
    @dataclass
    def get_dmy(self, t: date) -> tuple[int, int, int]:
        pass


@dataclass
class TF_30_360Base(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        m1, m2, y1, y2 = start_date.month, end_date.month, start_date.year, end_date.year
        d1, d2 = self.get_d_1_and_2(start_date, end_date)
        return (360 * (y2 - y1) + 30 * (m2 - m1) + d2 - d1) / 360
    
    @abstractmethod
    def get_d_1_and_2(self, start_date: date, end_date: date) -> tuple[int, int]:
        pass


@dataclass
class TF_30_360BondBasis(TF_30_360Base):
    def get_d_1_and_2(self, start_date: date, end_date: date) -> tuple[int, int]:
        '''
        From 2006 ISDA Definitions - 4.16 (f).
        “D1” is the first calendar day, expressed as a number, of the Calculation Period or
        Compounding Period, unless such number would be 31, in which case D1 will be 30; and
        “D2” is the calendar day, expressed as a number, immediately following the last day
        included in the Calculation Period or Compounding Period, unless such number would be 31 and
        D1 is greater than 29, in which case D2 will be 30
        '''
        d1 = min(start_date.day, 30)
        d2 = end_date.day
        d2 = d2 if not (d2==31 and d1 > 29) else 30
        return d1, d2


@dataclass
class TF_30E_360(TF_30_360Base):
    def get_d_1_and_2(self, start_date: date, end_date: date) -> tuple[int, int]:
        '''
        From 2006 ISDA Definitions - 4.16 (g).
        “D1” is the first calendar day, expressed as a number, of the Calculation Period or
        Compounding Period, unless such number would be 31, in which case D1 will be 30; and
        “D2” is the calendar day, expressed as a number, immediately following the last day
        included in the Calculation Period or Compounding Period, unless such number would be 31, in
        which case D2 will be 30. 
        '''
        d1 = min(start_date.day, 30)
        d2 = min(end_date.day, 30)
        return d1, d2


@dataclass
class TF_30E_360ISDA(TF_30_360Base):
    def get_d_1_and_2(self, start_date: date, end_date: date) -> tuple[int, int]:
        """
        From 2006 ISDA Definitions - 4.16 (h).
        “D1” is the first calendar day, expressed as a number, of the Calculation Period or
        Compounding Period, unless (i) that day is the last day of February or (ii) such number would be
        31, in which case D1 will be 30; and
        “D2” is the calendar day, expressed as a number, immediately following the last day
        included in the Calculation Period or Compounding Period, unless (i) that day is the last day of
        February but not the Termination Date or (ii) such number would be 31, in which case D2 will be
        30.
        """
        d1 = start_date.day
        sd_eom_day = calendar.monthrange(start_date.year, start_date.month)[1]
        d1 = d1 if not ((start_date.month==2 and start_date.day==sd_eom_day) or d1==31) else 30

        d2 = end_date.day
        ed_eom_day = calendar.monthrange(end_date.year, end_date.month)[1]
        d2 = d2 if not ((end_date.month==2 and end_date.day==ed_eom_day) or d2==31) else 30

        return d1, d2


@dataclass
class TF_Actual365(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        return (end_date - start_date).days / 365


@dataclass
class TF_Actual360(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        return (end_date - start_date).days / 360


@dataclass
class TF_Actual364(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        return (end_date - start_date).days / 364
    

@dataclass
class TF_Actual30(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        return (end_date - start_date).days / 30


@dataclass
class TF_Actual360_25(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        return (end_date - start_date).days / 365.25


@dataclass
class TF_Actual365_Long(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date, frequency: int) -> float:
        '''
        From Strata OpenGamma.
        The numerator is the actual number of days in the requested period.
        The denominator is determined by examining the frequency and the period end date (the date of the next coupon).
        If the frequency is annual then the denominator is 366 if the period contains February 29th, if not it is 365.
        The first day in the period is excluded, the last day is included.
        If the frequency is not annual, the denominator is 366 if the period end date is in a leap year, if not it is 365.
        '''
        numerator = (end_date - start_date).days

        if frequency == 1:
            denominator = 365
            for year in range(start_date.year, end_date.year + 1):
                if calendar.isleap(year):                    
                    if year != end_date:
                        denominator = 366
                        break
                    elif end_date >= date(year, 2, 29):
                        denominator = 366
                        break
        else:
            denominator = 366 if calendar.isleap(end_date.year) else 365
        return numerator / denominator


@dataclass
class TF_NL_360(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        '''
        From Strata OpenGamma.
        The result is a simple division.
        The numerator is the actual number of days in the requested period minus the number of occurrences of February 29.
        The denominator is always 360.
        The first day in the period is excluded, the last day is included.
        '''
        feb_29_count = 0    
        # Loop through each year in the range
        for year in range(start_date.year, end_date.year + 1):
            if calendar.isleap(year):
                feb_29 = date(year, 2, 29)
                if start_date <= feb_29 < end_date:
                    feb_29_count += 1

        return ((end_date - start_date).days - feb_29_count) / 360


@dataclass
class TF_NL_365(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        '''
        From Strata OpenGamma.
        The result is a simple division.
        The numerator is the actual number of days in the requested period minus the number of occurrences of February 29.
        The denominator is always 360.
        The first day in the period is excluded, the last day is included.
        '''
        feb_29_count = 0    
        # Loop through each year in the range
        for year in range(start_date.year, end_date.year + 1):
            if calendar.isleap(year):
                feb_29 = date(year, 2, 29)
                if start_date <= feb_29 < end_date:
                    feb_29_count += 1

        return ((end_date - start_date).days - feb_29_count) / 365


@dataclass
class TF_ActualActualISDA(TimeFractionBase):
    def get_day_count_factor(self, start_date: date, end_date: date) -> float:
        """
        From 2006 ISDA Definitions - 4.16 (b).
        “Actual/Actual”, “Actual/Actual (ISDA)”, “Act/Act” or “Act/Act (ISDA)” is specified,
        the actual number of days in the Calculation Period or Compounding Period in respect of which payment
        is being made divided by 365 (or, if any portion of that Calculation Period or Compounding Period falls
        in a leap year, the sum of (i) the actual number of days in that portion of the Calculation Period or
        Compounding Period falling in a leap year divided by 366 and (ii) the actual number of days in that
        portion of the Calculation Period or Compounding Period falling in a non-leap year divided by 365)
        """
        t = start_date
        t_year_end = date(t.year, 12, 31)

        time_factor = 0
        while t < end_date:
            next_t = min(t_year_end, end_date)
            period_base = 366 if calendar.isleap(t.year) else 365
            time_factor += (next_t - t).days / period_base
            t = next_t

        return time_factor


@dataclass
class TF_ActualActualICMA(TimeFractionBase):

    _actactisda_calc = field(init=False)

    def __post_init__(self):
        self._actactisda_calc = TF_ActualActualISDA()
    def get_day_count_factor(self, start_date: date, end_date: date, frequency: int) -> float:
        """
        From 2006 ISDA Definitions - 4.16 ().
        “Actual/Actual (ICMA)” or “Act/Act (ICMA)” is specified, a fraction equal to
        “number of days accrued/number of days in year”, as such terms are used in Rule 251 of the statutes,
        bylaws, rules and recommendations of the International Capital Market Association (the “ICMA Rule
        Book”), calculated in accordance with Rule 251 of the ICMA Rule Book as applied to non US dollar
        denominated straight and convertible bonds issued after December 31, 1998, as though the interest
        coupon on a bond were being calculated for a coupon period corresponding to the Calculation Period or
        Compounding Period in respect of which payment is being made.

        In summary is a Actual/Actual (ISDA) but divided on the frequency. Frequency is 1 for annual payments, 2 for semi-annual, 4 for quarterly...
        """
        return self._actactisda_calc.get_day_count_factor(start_date, end_date) / frequency