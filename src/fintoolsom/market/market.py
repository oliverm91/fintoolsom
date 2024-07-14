from dataclasses import dataclass, field
from datetime import date

from .currencies import Currency, CurrencyPair
from .index import Index
from .localities import Locality
from ..rates import Rate, ZeroCouponCurve
from ..dates import Calendar


@dataclass(slots=True)
class Market:
    t: date
    indexes_history: dict[str, dict[date, Index]] = field(default_factory=dict)
    currency_pairs_history: dict[str, dict[date, CurrencyPair]] = field(default_factory=dict)
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
        # upper names
        for k in list(self.currency_pairs_history.items()):
            v = self.currency_pairs_history.pop(k)
            self.currency_pairs_history[k.upper()] = v

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

    def get_currency_pair(self, t: date, base_currency: Currency, quote_currency: Currency) -> CurrencyPair:
        name = f'{base_currency.value}/{quote_currency.value}'
        if name in self.currency_pairs_history:
            if t in self.currency_pairs_history[name]:
                return self.currency_pairs_history[name][t]
            
        # If name not in history or t not in history for name, try to build with data from date t.
        
        # Check for direct inversion
        inverted_name = f"{quote_currency.value}/{base_currency.value}"
        if inverted_name in self.currency_pairs_history:
            inverted_name_dict = self.currency_pairs_history[inverted_name]
            if t in inverted_name_dict:
                inverted_cp = inverted_name_dict[t]
                cp = inverted_cp.invert()
                self.currency_pairs_history[name][t] = cp
                return cp
        else:
            # Get all CurrencyPairs with base and quote currency in it.
            base_values: dict[str, CurrencyPair] = {}
            for pair_name in self.currency_pairs_history.keys():
                if t in self.currency_pairs_history[pair_name]:
                    pair = self.currency_pairs_history[pair_name][t]
                    if pair.base_currency == base_currency:
                        base_values[pair_name] = pair
                    if pair.quote_currency == base_currency:
                        iv_pair = pair.invert()
                        base_values[pair.name] = iv_pair
            quote_values: dict[str, CurrencyPair] = {}
            for pair_name in self.currency_pairs_history.keys():
                if t in self.currency_pairs_history[pair_name]:
                    pair = self.currency_pairs_history[pair_name][t]
                    if pair.base_currency == quote_currency:
                        iv_pair = pair.invert()
                        quote_values[pair_name] = iv_pair
                    if pair.quote_currency == quote_currency:
                        quote_values[pair.name] = pair

            if len(base_values) == 0 or len(quote_values) == 0:
                raise ValueError(f"Could not build Currency pair {name} date {t}.")

            # Lookup for a connection: base_value.key[-3:]==quote_value.key[:3]
            for base_key, base_value in base_values.items():
                for quote_key, quote_value in quote_values.items():
                    if base_key[-3:] == quote_key[:3]:
                        combined_value = base_value.value * quote_value.value
                        result = CurrencyPair(t, base_currency, quote_currency, combined_value)
                        self.currency_pairs_history[t][result.name] = result
                        return result

            raise ValueError(f"Could not build Currency pair {name} date {t}.")
    
    def accrue_currency_pair(self, base_currency: Currency, quote_currency: Currency, t: date, locality: Locality) -> CurrencyPair:
        current_currency_pair = self.get_currency_pair(self.t, base_currency, quote_currency)
        foreign_curve = self.get_zero_coupon_curve(base_currency, locality=locality)
        domestic_curve = self.get_zero_coupon_curve(quote_currency, locality=locality)

        return current_currency_pair.accrue(foreign_curve, domestic_curve, t)