import numpy as np
from datetime import date
from collections.abc import Sequence
from typing import Union

from .Index import Index

from ..rates import ZeroCouponCurve


class Market:
    def __init__(self, indexes: dict, fx_rates: dict, market_date: date, base_currency: str='clp'):
        self.indexes = indexes if indexes is not None else {}
        self.fx_rates = fx_rates
        base_currency = base_currency.lower()
        if base_currency not in fx_rates.keys():
            fx_rates[base_currency] = {}
            for date in fx_rates[[*fx_rates.keys()][0]]:
                fx_rates[base_currency][date] = 1

        self.market_date = market_date
        self.derivatives_collateralized_curves = {}

    def __copy__(self):
        return Market(self.indexes, self.fx_rates, self.market_date)
    
    def __np_date_to_datetime(self, d):
        d = d.astype('U')[:10].split('-')
        return date(int(d[0]), int(d[1]), int(d[2]))

    def get_fx(self, base_currency: str, value_currency: str, date: date) -> float:
        value_date = self.market_date if base_currency+value_currency=='clfclp' else date
        return self.fx_rates[base_currency][date] / self.fx_rates[value_currency][value_date]
    
    def get_discount_index(self, locality: str, currency: str) -> Index:
        try:
            discount_index = [i for i in self.indexes.values() if i.locality==locality and i.currency==currency and i.is_risk_free][0]
        except:
            raise Exception(f'Error trying to get index with locality {locality} and currency {currency}.')
        return discount_index
    
    def get_discount_curve(self, locality: str, currency: str, collateral_index_name: str=None) -> ZeroCouponCurve:
        if collateral_index_name is None:
            index = self.get_discount_index(locality, currency)
            return index.curve
        else:
            #Try to find if already calculated
            try:
                lacks_locality = True
                lacks_currency = True
                collateralized_locality_curves = self.derivatives_collateralized_curves[locality]
                lacks_locality = False
                collateralized_loc_cur_curves = collateralized_locality_curves[currency]
                lacks_currency = False
                discount_curve = collateralized_loc_cur_curves[collateral_index_name]
                return discount_curve
            except KeyError:
                #Calculate it.
                # Example 1: Locality: CL. Currency: CLP. Collateral index: SOFR.
                # Should be: DF_CLP_CL_SOFR = DF_CLP_CL * DF_SOFR / DF_USD_CL 
                # Example 2: Locality: US. Currency: CLP. Colateralized: Euribor 3M.
                # Should be: DF_CLP_US_L3M = DF_CLP_US * DF_Euribor3M / DF_EUR_US <- self.get_discount_index. 
                # Generalized: DF_{currency}_{locality} * DF_{index} / DF_{collateral_currency}_{locality}.
                ## DF_{currency}_{locality} from self.get_discount_index(locality, currency).curve. Local currency curve
                ## DF_{index} from self.indexes[collateral_index_name].curve. Collateral index curve
                ## DF_{collateral_currency}_{locality} from self.get_discount_index(locality, self.indexes[collateral_index_name].currency).curve. Local collateral currency curve
                local_currency_index = self.get_discount_index(locality, currency)
                local_currency_curve = local_currency_index.curve
                
                collateral_index = self.indexes[collateral_index_name]
                collateral_curve = collateral_index.curve
                local_colaterallCurrency_curve = self.get_discount_index(locality, collateral_index.currency).curve
                
                discount_curve = collateral_curve.copy()
                if local_currency_curve.name != local_colaterallCurrency_curve.name:
                    collateralized_curve = local_currency_curve.combine_curve(collateral_curve, multiplication_combination=True)
                    collateralized_curve = collateralized_curve.combine_curve(local_colaterallCurrency_curve, multiplication_combination=False)
                    discount_curve = collateralized_curve.copy()
                
                #Save it so its not calculated again
                if lacks_locality:
                    self.derivatives_collateralized_curves[locality] = {currency: {collateral_index_name: discount_curve}}
                elif lacks_currency:
                    self.derivatives_collateralized_curves[locality][currency] = {collateral_index_name: discount_curve}
                else:
                    self.derivatives_collateralized_curves[locality][currency][collateral_index_name] = discount_curve
                
            return discount_curve
        

    def get_projected_fxs(self, base_currency: str, value_currency: str, dates: Union[Sequence, np.ndarray], locality: str) -> np.ndarray:
        dates.sort()
        base_projection_index = self.get_discount_index(locality, base_currency)
        value_projection_index = self.get_discount_index(locality, value_currency)
        base_projection_curve = base_projection_index.curve
        value_projection_curve = value_projection_index.curve
        
        if dates[0] < self.market_date:
            past_dates = dates[dates<=self.market_date]
            future_dates = dates[dates>self.market_date]
            past_fxs = np.array([self.get_fx(base_currency, value_currency, date) for date in past_dates])
            future_fxs = self.get_projected_fxs(base_currency, value_currency, future_dates)
            projected_fxs = np.concatenate(past_fxs, future_fxs)
        else:
            if base_currency+value_currency == 'clfclp':
                fixed_dates = np.array(list(self.fx_rates['clf'].keys())).astype('M')
                max_fixed_date = self.__np_date_to_datetime(np.max(fixed_dates))
                fixed_dates_fxs = np.array([self.get_fx(base_currency, value_currency, date) for date in dates if date <= max_fixed_date])
                
                non_fixed_dates = dates[dates > max_fixed_date]
                if len(non_fixed_dates) > 0:
                    base_curve_dfs = base_projection_curve.get_dfs(non_fixed_dates, starting_date=max_fixed_date)
                    value_curve_dfs = value_projection_curve.get_dfs(non_fixed_dates, starting_date=max_fixed_date)
                    max_date_fx = self.fx_rates['clf'][max_fixed_date]
                    future_projected_fxs = np.array(max_date_fx * base_curve_dfs / value_curve_dfs)
                    if future_projected_fxs.ndim == 0:
                        future_projected_fxs = np.array([future_projected_fxs])
                    projected_fxs = np.concatenate([fixed_dates_fxs, future_projected_fxs]) if len(fixed_dates_fxs) > 0 else future_projected_fxs
                else:
                    projected_fxs = fixed_dates_fxs
            else:
                base_curve_dfs = base_projection_curve.get_dfs(dates)
                value_curve_dfs = value_projection_curve.get_dfs(dates)
                spot = self.get_fx(base_currency, value_currency, self.market_date)
                projected_fxs = spot * base_curve_dfs / value_curve_dfs

        return projected_fxs


