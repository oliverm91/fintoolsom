from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ..fixedIncome import Bond
from .Rates import Rate
from .ZeroCouponCurve import ZeroCouponCurve


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
        '''
        Returns the exponential rate at time t calculated with Act/365 day count convention.

        Parameters
        ----------
        t : float | np.ndarray
            Time in years (Act/365)

        Returns
        -------
        float | np.ndarray
            exponential rate
        '''
        return NelsonSiegelSvensson._get_rate_params(t, self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_)
    
    @staticmethod
    def _get_rate_params(t: float | np.ndarray, b0: float, b1: float, b2: float, b3: float, lambda_: float, mu_: float):
        lambda_t = lambda_ * t
        mu_t = mu_ * t
        e_minus_lambda_t = np.exp(-lambda_t)
        e_minus_mu_t = np.exp(-mu_t)
        aux_term_lambda = (1 - e_minus_lambda_t) / lambda_t
        aux_term_mu = (1 - e_minus_mu_t) / mu_t

        return b0 + b1 * aux_term_lambda  + b2 * (aux_term_lambda - e_minus_lambda_t) + b3 * (aux_term_mu - e_minus_mu_t)

    def get_df(self, t: float | np.ndarray) -> float | np.ndarray:
        '''
        Returns the discount factor at time t calculated with Act/365 day count convention.

        Parameters
        ----------
        t : float | np.ndarray
            Time in years (Act/365)

        Returns
        -------
        float | np.ndarray
            exponential rate
        '''
        return np.exp(-self.get_rate(t) * t)
    
    @staticmethod
    def _get_df_params(t: float | np.ndarray, b0: float, b1: float, b2: float, b3: float, lambda_: float, mu_: float):
        return np.exp(-NelsonSiegelSvensson._get_rate_params(t, b0, b1, b2, b3, lambda_, mu_) * t)

    def calibrate(self, calibration_date: date, bonds_irr_list: list[tuple[Bond, Rate]], initial_guess: list[float] | None = None):
        bonds_irr_list.sort(key=lambda b_irr_tuple: (b_irr_tuple[0].get_maturity_date() - calibration_date).days)
        mkt_pvs = ([bond.get_present_value(calibration_date, irr) for bond, irr in bonds_irr_list])
        bond_t_flows_lst = [(np.array([(ed - calibration_date).days / 365 for ed in bond.coupons.get_remaining_end_dates(calibration_date)]),
                                bond.coupons.get_remaining_flows(calibration_date)) for bond, _ in bonds_irr_list]
        def get_valuation_error(params):
            nss_pv = np.array([sum(self._get_df_params(bond_t, *params) * bond_flows) for bond_t, bond_flows in bond_t_flows_lst])
            return sum((nss_pv - mkt_pvs)**2)
        
        if initial_guess is None:
            self.b0 = sum([irr.rate_value for _, irr in bonds_irr_list])/len(bonds_irr_list)
            short_term_yields = [irr.rate_value for _, irr in bonds_irr_list[:min(3, len(bonds_irr_list))]]
            self.b1 = sum(short_term_yields)/len(short_term_yields) - self.b0
            self.b2, self.b3, self.lambda_, self.mu_ = 0, 0.01, 1/2, 1/5
            initial_guess = [self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_]

        from scipy.optimize import minimize
        with np.errstate(over='ignore'):
            result = minimize(get_valuation_error, initial_guess, method='Powell', options={'maxiter': 1000})

        if result.success is False:
            raise ValueError(result.message)

        self.b0, self.b1, self.b2, self.b3, self.lambda_, self.mu_ = result.x

    def get_curve(self, calibration_date: date, bonds_irr_list: list[tuple[Bond, Rate]], initial_guess: list[float] | None = None) -> ZeroCouponCurve:
        self.calibrate(calibration_date, bonds_irr_list, initial_guess)
        mat_dates = [bond.get_maturity_date() for bond, _ in bonds_irr_list]
        dfs = self.get_df(np.array([(mat - calibration_date).days / 365 for mat in mat_dates]))
        curve = ZeroCouponCurve(date_dfs=list(zip(mat_dates, dfs)))
        return curve