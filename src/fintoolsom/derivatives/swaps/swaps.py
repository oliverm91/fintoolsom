from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ...market import Currency, CurrencyPair, Market, Locality, Index
from ...rates import ZeroCouponCurve
from ...dates import Calendar

from .legs import SwapLegBase, DefaultSwapLeg, IborRateSwapLeg, OvernightRateSwapLeg
from .coupons import SwapCouponBase

@dataclass
class SwapBase(ABC):
    receive_leg: SwapLegBase
    pay_leg: SwapLegBase

    collateral_index_name: str = field(default=None)

    def __post_init__(self):
        if self.receive_leg.currency == self.pay_leg.currency and self.receive_leg.notional != self.pay_leg.notional:
            raise ValueError(f'If receive and pay legs are of the same currency, notional must match. Receive leg notional: {self.receive_leg.notional}, Pay leg notional: {self.pay_leg.notional}')

    @abstractmethod    
    def get_receive_flows_value(self, market: Market=None, locality: Locality=None) -> float:
        pass
    
    @abstractmethod
    def get_pay_flows_value(self, market: Market=None, locality: Locality=None) -> float:
        pass
    
    def valuate(self, market: Market=None, locality: Locality=None) -> float:
        receive_discount_curve = market.get_discount_curve(self.receive_leg.currency, collateral_index_name=self.collateral_index_name, locality=locality)
        pay_discount_curve = market.get_discount_curve(self.pay_leg.currency, collateral_index_name=self.collateral_index_name, locality=locality)

        receive_flows_value = self.get_receive_flows_value(market=market, locality=locality)
        pay_flows_value = self.get_pay_flows_value(market=market, locality=locality)

        receive_future_payment_dates = [c.payment_date for c in self.receive_leg.coupons]
        pay_future_payment_dates = [c.payment_date for c in self.pay_leg.coupons]

        return receive_flows_value * receive_discount_curve.get_dfs(receive_future_payment_dates) - pay_flows_value * pay_discount_curve.get_dfs(pay_future_payment_dates)
    
@dataclass
class DefaultSwap(SwapBase):
    receive_leg: SwapLegBase
    pay_leg: SwapLegBase

    collateral_index_name: str = field(default=None)

    def __post_init__(self):
        if self.receive_leg.currency == self.pay_leg.currency and self.receive_leg.notional != self.pay_leg.notional:
            raise ValueError(f'If receive and pay legs are of the same currency, notional must match. Receive leg notional: {self.receive_leg.notional}, Pay leg notional: {self.pay_leg.notional}')
        
    def get_receive_flows_value(self, market: Market=None, locality: Locality=None) -> float:
        return self.receive_leg.get_flows(market=market, locality=locality)
    
    def get_pay_flows_value(self, market: Market=None, locality: Locality=None) -> float:
        return self.pay_leg.get_flows(market=market, locality=locality)

@dataclass
class FXCompensatedSwap(SwapBase):
    calendar: Calendar
    compensation_currency: Currency
    fx_fixing_lag: int = field(default=0)

    fx_fixing_dates: list[date] = field(default=None)
    
    def __post_init__(self):
        super().__post_init__()

        rp_coups = list(zip(self.receive_leg.coupons, self.pay_leg.coupons))
        for rc, pc in rp_coups:
            if rc.payment_date != pc.payment_date:
                raise ValueError(f'Payment dates of receive and pay legs must match. Receive leg payment date: {rc.payment_date}, Pay leg payment date: {pc.payment_date}')

        # Check that all coupons from both legs match payment dates if all ok add fx_fixing_date
        if self.fx_fixing_dates is None:
            self.fx_fixing_dates = []
            for rc, pc in rp_coups:
                self.fx_fixing_dates.append(self.calendar.add_business_days(rc.payment_date, -self.fx_fixing_lag))
        else:
            # Check fx_fixing_dates length
            if len(self.fx_fixing_dates) != len(rp_coups):
                raise ValueError(f'fx_fixing_dates length must match the number of swap coupons. fx_fixing_dates length: {len(self.fx_fixing_dates)}, number of swap coupons: {len(rp_coups)}')
            self.fx_fixing_dates.sort()

            rf_ts = list(zip(self.receive_leg.coupons, self.fx_fixing_dates))

            # Check that all coupons have an fx_fixing_date between start and payment_date
            for rc, fd in rf_ts:
                if not rc.start_accrual_date <= fd <= rc.payment_date:
                    raise ValueError(f'fx_fixing_dates must be between start and payment dates. fx_fixing_date: {fd}, start_accrual_date: {rc.start_accrual_date}, payment_date: {rc.payment_date}')

    def get_fx_rate_values(self, market: Market, original_currency: Currency, fx_fixing_dates: list[date]) -> np.ndarray:
        past_fx_rates = [market.get_currency_pair(t, original_currency, self.compensation_currency).value for t in fx_fixing_dates if t <= market.t]
        if len(past_fx_rates) == len(self.fx_fixing_dates):
            return np.array(past_fx_rates)
        else:
            future_fx_rates = [market.accrue_currency_pair(original_currency, self.compensation_currency, t).value for t in fx_fixing_dates[len(past_fx_rates):]]
            all_fx = past_fx_rates + future_fx_rates
            return np.array(all_fx)

    def get_receive_flows_value(self, market: Market=None, locality: Locality=None) -> np.ndarray:
        original_currency_receive_flows = self.receive_leg.get_flows(market=market, locality=locality)
        fx_fixing_dates = self.fx_fixing_lag[-len(original_currency_receive_flows):]
        fx_rates = self.get_fx_rate_values(market, self.receive_leg.currency, fx_fixing_dates)
        return original_currency_receive_flows * fx_rates
        
    def get_pay_flows_value(self, market: Market=None, locality: Locality=None) -> float:
        original_currency_pay_flows = self.pay_leg.get_flows(market=market, locality=locality)
        fx_fixing_dates = self.fx_fixing_lag[-len(original_currency_pay_flows):]
        fx_rates = self.get_fx_rate_values(market, self.pay_leg.currency, fx_fixing_dates)
        return original_currency_pay_flows * fx_rates