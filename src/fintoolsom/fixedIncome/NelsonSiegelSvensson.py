from dataclasses import dataclass, field
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional
from scipy.optimize import minimize

import numpy as np
from numba import njit, vectorize

from .Bonds import Bond
from ..rates.Rates import Rate
from ..rates.ZeroCouponCurve import ZeroCouponCurve


@vectorize(nopython=True)
def _nss_rate(t: float | np.ndarray, beta0: float, beta1: float, beta2: float, beta3: float, lambda_: float, mu_: float) -> float | np.ndarray:
    lambda_t = lambda_ * t
    mu_t = mu_ * t

    e_minus_lambda_t = np.exp(-lambda_t)
    aux_term_lambda = (1 - e_minus_lambda_t) / lambda_t                
    e_minus_mu_t = np.exp(-mu_t)
    return beta0 + beta1 * aux_term_lambda  + beta2 * (aux_term_lambda - e_minus_lambda_t) + beta3 * ((1 - e_minus_mu_t) / mu_t - e_minus_mu_t)


@dataclass
class NelsonSiegelSvensson:
    b0: float = field(init=False)
    b1: float = field(init=False)
    b2: float = field(init=False)
    b3: float = field(init=False)
    lambda_: float = field(init=False)
    mu_: float = field(init=False)

    def __post_init__(self):
        self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_ = 0.03, 0.01, 0, 0.01, 0.5, 0.2
        
    def get_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        return NelsonSiegelSvensson._get_rate(t, self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_)
    
    def get_df(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.exp(-self.get_rate(t) * t)
    
    def calibrate(self, calibration_date: date, bonds_irr_list: list[tuple[Bond, Rate]], initial_guess: Optional[list[float]] = None, method: Optional[str]='powell'):
        bonds_irr_list.sort(key=lambda b_irr_tuple: (b_irr_tuple[0].get_maturity_date() - calibration_date).days)
        mkt_pvs = np.array([bond.get_present_value(calibration_date, irr) for bond, irr in bonds_irr_list])
        bond_t_flows_lst = [(np.array([(ed - calibration_date).days / 365 for ed in bond.coupons.get_remaining_end_dates(calibration_date)]),
                                bond.coupons.get_remaining_flows(calibration_date)) for bond, _ in bonds_irr_list]
        bonds_ts, bonds_flows = zip(*bond_t_flows_lst)
        flat_flows = np.concatenate(bonds_flows)
        flat_ts = np.concatenate(bonds_ts)
        starts_ts = np.cumsum([len(bond_ts) for bond_ts in bonds_ts])
        starts_ts = np.insert(starts_ts, 0, 0)
        lengths_ts = np.diff(starts_ts)
        starts_ts = starts_ts[:-1]
        starts_lengths = list(zip(starts_ts, lengths_ts))
        ns_pvs = np.zeros(len(bonds_ts))
        def get_valuation_error(params):
            ns_dfs = np.exp(-NelsonSiegelSvensson._calculate_rates_flatted(flat_ts, starts_ts, lengths_ts, *params) * flat_ts)
            flows_vps = ns_dfs * flat_flows
            for ix, start_length in enumerate(starts_lengths):
                start, length = start_length
                ns_pvs[ix] = np.sum(flows_vps[start:start+length])
            return sum((ns_pvs - mkt_pvs)**2)
        
        if initial_guess is None:
            self.b0 = sum([irr.rate_value for _, irr in bonds_irr_list])/len(bonds_irr_list)
            short_term_yields = [irr.rate_value for _, irr in bonds_irr_list[:min(3, len(bonds_irr_list))]]
            self.b1 = sum(short_term_yields)/len(short_term_yields) - self.b0
            self.b2 = 0
            self.b3 = 0.01
            self.lambda_ = 1/2 # First hump around 2Y
            self.mu_ = 1/5 # Second hump arond 5Y
            initial_guess = [self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_]

        with np.errstate(over='ignore'):
            bounds = [
                (None, None),  # beta0
                (None, None),  # beta1
                (None, None),  # beta2
                (None, None),  # beta3
                (1/7, 1/0.5), # lambda | First hump to be between 0.5 and 7Y
                (1/20, 1/3)  # mu | Second hump to be 3 and 20Y
            ]
            result = minimize(get_valuation_error, initial_guess, method=method, bounds=bounds, options={"maxiter": 300}, tol=0.01)

        if result.success is False:
            raise ValueError(result.message)
        
        self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_ = result.x
    
    @staticmethod
    def _get_rate(t: float | np.ndarray, b0: float, b1: float, b2: float, b3: float, lambda_: float, mu_: float) -> float | np.ndarray:
        return _nss_rate(t, b0, b1, b2, b3, lambda_, mu_)
    
    @staticmethod
    def _get_df(t: float | np.ndarray, b0: float, b1: float, b2: float, b3: float, lambda_: float, mu_: float) -> float | np.ndarray:
        return np.exp(-NelsonSiegelSvensson._get_rate(t, b0, b1, b2, b3, lambda_, mu_) * t)
    
    @staticmethod
    @njit
    def _calculate_rates_flatted(flat_times, starts, lengths, beta0, beta1, beta2, beta3, lambda_, mu_) -> np.ndarray:
        num_bonds = len(starts)
        flat_rates = np.zeros_like(flat_times)
        for i in range(num_bonds):
            start = starts[i]
            end = start + lengths[i]
            flat_rates[start:end] = _nss_rate(flat_times[start:end], beta0, beta1, beta2, beta3, lambda_, mu_)
        return flat_rates

    def get_curve(self, calibration_date: date, bonds_irr_list: list[tuple[Bond, Rate]], initial_guess: list[float] | None = None, method: str = 'powell') -> ZeroCouponCurve:
        bonds_irr_list = [(bond, irr) for bond, irr in bonds_irr_list if bond.get_maturity_date() > calibration_date]
        self.calibrate(calibration_date, bonds_irr_list, initial_guess=initial_guess, method=method)
        dates = [calibration_date + relativedelta(months=i) + relativedelta(days=1) for i in range(0, 12*20, 1)]
        dfs = self.get_df(np.array([(mat - calibration_date).days / 365 for mat in dates]))
        curve = ZeroCouponCurve(calibration_date, date_dfs=list(zip(dates, dfs)))
        return curve
    
    def get_params(self) -> list[float]:
        return [self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_]