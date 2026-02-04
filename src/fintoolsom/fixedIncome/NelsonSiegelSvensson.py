import numpy as np
from numba import njit
from scipy.optimize import minimize
from dataclasses import dataclass, field
from datetime import date
from dateutil.relativedelta import relativedelta

# Assuming these relative imports exist in your project
from .Bonds import Bond
from ..rates.Rates import Rate
from ..rates.ZeroCouponCurve import ZeroCouponCurve

# --- JIT-Compiled Rate Functions ---

@njit
def _ns_rate(t: np.ndarray, b0: float, b1: float, b2: float, l1: float) -> np.ndarray:
    """Core Nelson-Siegel (3-factor) logic."""
    t_l1 = t / l1
    e_l1 = np.exp(-t_l1)
    aux_l1 = (1 - e_l1) / t_l1
    return b0 + b1 * aux_l1 + b2 * (aux_l1 - e_l1)

@njit
def _nss_rate(t: np.ndarray, b0: float, b1: float, b2: float, b3: float, l1: float, l2: float) -> np.ndarray:
    """Nelson-Siegel-Svensson leveraging the NS logic."""
    base_ns = _ns_rate(t, b0, b1, b2, l1)
    t_l2 = t / l2
    e_l2 = np.exp(-t_l2)
    aux_l2 = (1 - e_l2) / t_l2
    return base_ns + b3 * (aux_l2 - e_l2)

# --- Model Classes ---

@dataclass
class NelsonSiegel:
    b0: float = field(init=False)
    b1: float = field(init=False)
    b2: float = field(init=False)
    l1: float = field(init=False)

    def __post_init__(self):
        self.b0, self.b1, self.b2, self.l1 = 0.03, -0.01, 0.0, 2.0

    def get_params(self) -> tuple:
        return (self.b0, self.b1, self.b2, self.l1)

    def get_rate(self, t: np.ndarray) -> np.ndarray:
        return _ns_rate(t, *self.get_params())

    def get_df(self, t: np.ndarray) -> np.ndarray:
        return np.exp(-self.get_rate(t) * t)

    @staticmethod
    @njit
    def _calculate_rates_flatted(flat_times, starts, lengths, b0, b1, b2, l1) -> np.ndarray:
        flat_rates = np.zeros_like(flat_times)
        for i in range(len(starts)):
            s, e = starts[i], starts[i] + lengths[i]
            flat_rates[s:e] = _ns_rate(flat_times[s:e], b0, b1, b2, l1)
        return flat_rates

    def _prepare_bond_structures(self, calibration_date: date, bonds_irr_list: list[tuple[Bond, Rate]]):
        bonds_irr_list.sort(key=lambda x: (x[0].get_maturity_date() - calibration_date).days)
        mkt_pvs = np.array([bond.get_present_value(calibration_date, irr) for bond, irr in bonds_irr_list])
        bond_data = [(np.array([(ed - calibration_date).days / 365 for ed in b.coupons.get_remaining_end_dates(calibration_date)]),
                      b.coupons.get_remaining_flows(calibration_date)) for b, _ in bonds_irr_list]
        flat_ts = np.concatenate([x[0] for x in bond_data])
        flat_flows = np.concatenate([x[1] for x in bond_data])
        lengths = np.array([len(x[0]) for x in bond_data])
        starts = np.insert(np.cumsum(lengths)[:-1], 0, 0)
        return mkt_pvs, flat_ts, flat_flows, starts, lengths

    def calibrate(self, calibration_date: date, bonds_irr_list: list, method='L-BFGS-B', min_l1=0.5, max_l1=3.0):
        mkt_pvs, flat_ts, flat_flows, starts, lengths = self._prepare_bond_structures(calibration_date, bonds_irr_list)
        
        def objective(params):
            rates = self._calculate_rates_flatted(flat_ts, starts, lengths, *params)
            dfs = np.exp(-rates * flat_ts)
            pv_flows = dfs * flat_flows
            model_pvs = np.zeros(len(mkt_pvs))
            for i in range(len(mkt_pvs)):
                model_pvs[i] = np.sum(pv_flows[starts[i]:starts[i]+lengths[i]])
            return np.sum((model_pvs - mkt_pvs)**2)

        bounds = [(None, None), (None, None), (None, None), (min_l1, max_l1)]
        res = minimize(objective, self.get_params(), method=method, bounds=bounds, tol=0.01)
        self.b0, self.b1, self.b2, self.l1 = res.x

    def get_curve(self, calibration_date: date, bonds_irr_list: list, method='powell') -> ZeroCouponCurve:
        self.calibrate(calibration_date, bonds_irr_list, method=method)
        dates = [calibration_date + relativedelta(months=i) for i in range(1, 241)]
        ts = np.array([(d - calibration_date).days / 365 for d in dates])
        return ZeroCouponCurve(calibration_date, date_dfs=list(zip(dates, self.get_df(ts))))


@dataclass
class NelsonSiegelSvensson(NelsonSiegel):
    b3: float = field(init=False)
    l2: float = field(init=False)

    def __post_init__(self):
        self.b0, self.b1, self.b2, self.b3, self.l1, self.l2 = 0.03, -0.01, 0.0, 0.01, 2.0, 5.0

    def get_params(self) -> tuple:
        return (self.b0, self.b1, self.b2, self.b3, self.l1, self.l2)

    def get_rate(self, t: np.ndarray) -> np.ndarray:
        return _nss_rate(t, *self.get_params())

    @staticmethod
    @njit
    def _calculate_rates_flatted(flat_times, starts, lengths, b0, b1, b2, b3, l1, l2) -> np.ndarray:
        flat_rates = np.zeros_like(flat_times)
        for i in range(len(starts)):
            s, e = starts[i], starts[i] + lengths[i]
            flat_rates[s:e] = _nss_rate(flat_times[s:e], b0, b1, b2, b3, l1, l2)
        return flat_rates

    def calibrate(self, calibration_date: date, bonds_irr_list: list, method='L-BFGS-B', 
                  min_l1=0.3, max_l1=3.0, max_l2=15.0):
        
        mkt_pvs, flat_ts, flat_flows, starts, lengths = self._prepare_bond_structures(calibration_date, bonds_irr_list)

        def objective(params):
            l1_val, l2_val = params[4], params[5]
            gap = l2_val - l1_val
            penalty = 1.0
            if gap < 0.1:
                penalty = 1000*(0.1 - gap)**2
            
            rates = self._calculate_rates_flatted(flat_ts, starts, lengths, *params)
            dfs = np.exp(-rates * flat_ts)
            pv_flows = dfs * flat_flows
            
            model_pvs = np.zeros(len(mkt_pvs))
            for i in range(len(mkt_pvs)):
                model_pvs[i] = np.sum(pv_flows[starts[i]:starts[i]+lengths[i]])
                
            pricing_error = np.sum((model_pvs - mkt_pvs)**2)
            return pricing_error * penalty

        bounds = [(None, None), (None, None), (None, None), (None, None), (min_l1, max_l1), (min_l1 + 0.1, max_l2)]
        res = minimize(objective, self.get_params(), method=method, bounds=bounds, tol=0.01)
        self.b0, self.b1, self.b2, self.b3, self.l1, self.l2 = res.x