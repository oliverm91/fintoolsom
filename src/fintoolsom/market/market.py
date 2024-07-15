from dataclasses import dataclass, field
from datetime import date

from .currencies import Currency, CurrencyPair, FX_Rate, FX_RateData
from .index import Index
from .localities import Locality
from ..rates import Rate, ZeroCouponCurve
from ..dates import Calendar


@dataclass(slots=True)
class Market:
    t: date
    fx_history: dict[CurrencyPair, FX_RateData] = field(default_factory=dict)
    indexes_history: dict[str, dict[date, Index]] = field(default_factory=dict)
    interest_rates: dict[str, dict[date, Rate]] = field(default_factory=dict)
    index_to_interest_rate_map: dict[str, str] = field(default_factory=dict)

    interest_rate_to_index_map: dict[str, str] = field(init=False)
    
    # How should curves be assigned?
    #    Curves have a locality
    #    All curves have a currency. Curve currency might be different from locality currency. Ex: CL locality can have a USD and EUR curves.
    #    Some curves have an index. Indexed curves are not dependant on locality, but on currency. Ex: SOFR index have a CLP and a USD curve.
    #    
    #    At the end, every curve has a currency. Therefore, this is the first filter to ask for a curve.
    #    Indexed curves, do not need locality. Non-indexed curves, do.
    #    
    #    Proposed data structure to store curves: dict[str, dict[str, ZeroCouponCurve]]
    #    Explained structure: Dict with currency name as key. Value is a dict that can have either a locality or index name as key and a ZeroCouponCurve as value.
    zero_coupon_curve_mapper: dict[str, ZeroCouponCurve | dict[Locality, dict[Currency, ZeroCouponCurve]]] = field(default_factory=dict) # First key is index_name which can be None.

    def __post_init__(self):
        for cp, fx_data in self.fx_history.items():
            # Check fx_history FX_RateData is correctly mapped to CurrencyPairs.
            if cp!=fx_data.currency_pair:
                raise ValueError(f'fx_history contained a key-value pair inconsistent in CurrencyPair. Key: {cp}, Value CurrencyPair: {fx_data.currency_pair}')
            # Add inverted data
            if cp.invert() not in self.fx_history:
                self.fx_history[cp.invert()] = fx_data.invert()


        for k in list(self.zero_coupon_curve_mapper.items()):
            v = self.zero_coupon_curve_mapper.pop(k)
            self.zero_coupon_curve_mapper[k.upper()] = v

        for k in list(self.interest_rates.items()):
            v = self.interest_rates.pop(k)
            self.interest_rates[k.upper()] = v

        for k in list(self.indexes_history.items()):
            v = self.indexes_history.pop(k)
            self.indexes_history[k.upper()] = v

        self.interest_rate_to_index_map = {
            rate_name: index_name for index_name, rate_name in self.index_to_interest_rate_map.items()
        }

        # Invert currency pairs
        for cp_history_dict in self.currency_pairs_history.values():
            for cp_date, cp in cp_history_dict.items():
                inverted_cp = cp.invert()
                self.currency_pairs_history[inverted_cp.name][cp_date] = inverted_cp

    def add_index(self, index: Index):
        if index.name not in self.indexes_history:
            self.indexes_history[index.name] = {}
        self.indexes_history[index.name][index.index_date] = index

    def add_currency_pair(self, currency_pair: CurrencyPair):
        if currency_pair.name not in self.currency_pairs_history:
            self.currency_pairs_history[currency_pair.name] = {}
        self.currency_pairs_history[currency_pair.name][currency_pair.cp_date] = currency_pair
        inverted_pair = currency_pair.invert()
        self.currency_pairs_history[currency_pair.name][inverted_pair.cp_date] = inverted_pair

    def get_index(self, t: date, name: str) -> Index:
        if name in self.indexes_history:
            if t in self.indexes_history[name]:
                return self.indexes_history[name][t]
            raise KeyError(f"Index {name} not found for date {t}.")

        raise KeyError(f"Index {name} not found in market.")
            
    def get_rate(self, t: date, name: str, use_closest_past_rate: bool = False) -> Rate:
        if name in self.interest_rates:
            if t in self.interest_rates[name]:
                return self.interest_rates[name][t]
            if not use_closest_past_rate:
                raise ValueError(f"Rate {name} not found for date {t}.")
            else:
                past_dates = [x for x in self.interest_rates if x < t and name in self.interest_rates[x]]
                if len(past_dates) == 0:
                    raise ValueError(f"Rate {name} not found for date {t} and there were no past dates.")
                last_rate_date = max(past_dates)
                return self.interest_rates[name][last_rate_date]

        raise ValueError(f"Rate {name} not found.")

    def accrue_rates_reset_business_days(self, notional: float, rate_name: str, start_date: date, end_date: date, fixing_lag: int=0, calendar: Calendar=None, use_closest_past_rate_for_fixing: bool=False) -> float:
        if calendar is None:
            calendar = Calendar()
        if fixing_lag < 0:
            raise ValueError(f"fixing_lag must be greater than or equal to 0. Got {fixing_lag}.")
        if start_date > end_date:
            raise ValueError(f"Start date must be earlier than end date. Start date: {start_date}, End date: {end_date}.")
        t = start_date
        acrrued_interest = 0
        while t < end_date:
            reset_date = calendar.add_business_day(t, -fixing_lag)
            rate = self.get_rate(reset_date, rate_name, use_closest_past_rate=use_closest_past_rate_for_fixing)
            next_business_day = calendar.add_business_day(t, 1)
            acrrued_interest += rate.get_accrued_interest(notional + acrrued_interest, t, min(next_business_day, end_date))
            t = calendar.add_business_day(t, 1)

        return acrrued_interest

    def accrue_rates_custom_reset_days(self, notional: float, rate_name: str, start_date: date, end_date: date, reset_dates: list[date], use_closest_past_rate_for_fixing: bool=False) -> float:
        reset_dates.sort()
        if reset_dates[0] > start_date:
            raise ValueError(f"First reset date must be after start date. Got Start date: {start_date} and min reset date: {reset_dates[0]}.")
        if start_date > end_date:
            raise ValueError(f"Start date must be earlier than end date. Start date: {start_date}, End date: {end_date}.")
        t = start_date
        reset_counter = 0
        while t < end_date:
            reset_date = reset_dates[reset_counter]
            rate = self.get_rate(reset_date, rate_name, use_closest_past_rate=use_closest_past_rate_for_fixing)

            # If we have reached the end of the reset dates, we need to accrue till the end date. Case when custom dates are within accrual period
            if reset_counter == len(reset_dates) - 1:
                next_t = end_date
            else:
                next_t = reset_dates[reset_counter + 1]

            # If we have not reach the end of reset dates AND next_t is after end_date, we need to accrue till the end date. Case when custom dates are after accrual period
            if next_t > end_date:
                next_t = end_date # As t becomes next_t, loop will end.            
            acrrued_interest = rate.get_accrued_interest(notional + acrrued_interest, t, next_t)
            reset_counter += 1
            t = next_t
        return acrrued_interest
    
    def accrue_rates_single_reset_day(self, notional: float, rate_name: str, start_date: date, end_date: date, reset_date: date, use_closest_past_rate_for_fixing: bool=False) -> float:
        if start_date > end_date:
            raise ValueError(f"Start date must be earlier than end date. Start date: {start_date}, End date: {end_date}.")
        if reset_date > start_date:
            raise ValueError(f"Reset date must be earlier than start date. Reset date: {reset_date}, Start date: {start_date}.")
        rate = self.get_rate(reset_date, rate_name, use_closest_past_rate=use_closest_past_rate_for_fixing)
        return rate.get_accrued_interest(notional, start_date, end_date)
    
    def get_zero_coupon_curve(self, currency: Currency, locality: Locality=None, index_name: str=None) -> ZeroCouponCurve:
        currency_str = currency.value
        if currency_str not in self.zero_coupon_curve_mapper:
            raise KeyError(f"No zero coupon curve found for currency {currency_str}.")
        currency_curves_dict = self.zero_coupon_curve_mapper[currency_str] # This dict has locality and index_name as keys

        if locality is None and index_name is None:
            raise ValueError("locality and index_name cannot both be None.")
        
        if locality is not None:
            if locality not in currency_curves_dict:
                raise KeyError(f"No zero coupon curve found for currency {currency_str} and locality {locality}.")
            return currency_curves_dict[locality]
        else:
            index_name = index_name.upper()
            if index_name not in currency_curves_dict:
                raise KeyError(f"No zero coupon curve found for currency {currency_str} and index {index_name}.")
            return currency_curves_dict[index_name]
        
    def get_discount_curve(self, currency: Currency, collateral_index_name: str=None, locality: Locality=None) -> ZeroCouponCurve:
        return self.get_zero_coupon_curve(currency, locality=locality, index_name=collateral_index_name)
    
    def add_fx_rate(self, t: date, fx_rate: FX_Rate):
        if fx_rate.currency_pair not in self.fx_history:
            self.fx_history[fx_rate.currency_pair] = FX_RateData(fx_rate.currency_pair, {t: fx_rate})
        else:
            self.fx_history[fx_rate.currency_pair].add_date(t, fx_rate)

    def get_fx_rate(self, t: date, currency_pair: CurrencyPair) -> FX_Rate:
        if currency_pair in self.fx_history:
            return self.fx_history[currency_pair].get_fx_rate(t)
        else:
            raise KeyError(f'No {currency_pair} data loaded in market.')