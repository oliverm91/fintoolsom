from datetime import timedelta

from ..rates import ZeroCouponCurve, RateConvention, Rate

class Index:
    def __init__(self, name: str, curve: ZeroCouponCurve, currency: str, fixings: dict, locality: str, is_risk_free: bool, is_rate: bool, is_overnight: bool, rate_convention: RateConvention=None):
        if is_rate and rate_convention is None:
            raise ValueError('Index with attribute is_rate set to True must have a rate convention.')
        if rate_convention is not None and not is_rate:
            raise ValueError('Index with attribute is_rate set to False must not have a rate convention.')

        self.name = name
        self.locality = locality
        self.curve = curve
        self.fixings = fixings
        self.currency = currency
        self.is_risk_free = is_risk_free
        self.is_rate = is_rate
        self.is_overnight = is_overnight
        self.fixed_factors = self.__get_fixed_factors() if self.is_overnight and fixings is not None else None
        self.locality = locality

    def __get_complete_fixed_factors(self):
        incomplete_dict = {key: value for key, value in self.fixings.items()}
        min_date = min(incomplete_dict.keys())
        max_date = max(incomplete_dict.keys())
        date = min_date + timedelta(days=1)
        while date < max_date:
            if date not in incomplete_dict.keys():
                previous_fixings = {date_i: index
                                    for date_i, index in list(incomplete_dict.items()) if date_i < date}
                if len(previous_fixings.keys()) < 2:
                    incomplete_dict[date] = max(previous_fixings.values())
                else:
                    previous_dates = list(previous_fixings.keys())
                    previous_dates.sort(reverse=True)
                    days = (previous_dates[0] - previous_dates[1]).days
                    previous_rate = (previous_fixings[previous_dates[0]]/previous_fixings[previous_dates[1]] - 1) * 360/days
                    days_to_date = (date - previous_dates[0]).days
                    incomplete_dict[date] = previous_fixings[previous_dates[0]]*(1 + previous_rate * days_to_date / 360)
            date = date + timedelta(days=1)
        return incomplete_dict

    def __get_fixed_factors(self):
        fixed_factors = {}
        if self.is_rate:
            fixing_dates = list(self.fixings.keys())
            min_fixing_date = min(fixing_dates)
            max_fixing_date = max(fixing_dates)
            fixing_dates.sort()
            fixed_factors[min_fixing_date] = 100.000000
            
            previous_fixing_date = min_fixing_date
            rate = self.fixings[min_fixing_date]
            while previous_fixing_date < max_fixing_date:
                fixing_date = previous_fixing_date + timedelta(days=1)
                rate = self.fixings[fixing_date] if fixing_date in fixing_dates else rate
                wf = rate.get_wealth_factor(previous_fixing_date, fixing_date)
                fixed_factors[fixing_date] = fixed_factors[previous_fixing_date] * wf
                previous_fixing_date = fixing_date
        else:
            fixed_factors = self.__get_complete_fixed_factors()
        
        return fixed_factors