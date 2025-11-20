from datetime import date
import numpy as np


def get_irr(
    cash_flows: list[float],
    dates: list[date],
    valuation_date: date,
    pv: float,
    initial_guess: float = 0.04,
    target_error: float = 1e-10,
    max_iterations: int = 100,
    base: int = 365,
    ) -> float:
    """
    Compute the Internal Rate of Return (IRR) — or yield to maturity (YTM) — 
    for a set of dated cash flows using the Newton–Raphson method.

    The IRR is the discount rate `r` that satisfies:

        PV = Σ [ CF_i / (1 + r)^(t_i) ]

    where:
      - PV is the present value at `valuation_date`
      - CF_i are the future cash flows occurring on `dates[i]`
      - t_i is the year fraction between `valuation_date` and `dates[i]` with ACT/`base` convention

    The method iteratively refines `r` by applying:

        r_{n+1} = r_n - f(r_n) / f'(r_n)

    where:
        f(r)  = Σ [ CF_i / (1 + r)^(t_i) ] - PV
        f'(r) = -Σ [ t_i * CF_i / (1 + r)^(t_i + 1) ]

    Parameters
    ----------
    cash_flows : list[float]
        Cash flows of the instrument (must exclude any initial outlay).
    dates : list[date]
        Dates corresponding to each cash flow.
    valuation_date : date
        Date on which the present value is evaluated.
    pv : float
        Objective present value of all future cash flows as of the valuation date.
    initial_guess : float, optional
        Starting estimate for the IRR (default is 0.04).
    target_error : float, optional
        Tolerance threshold for the NPV residual (default is 1e-10).
    max_iterations : int, optional
        Maximum number of Newton–Raphson iterations before aborting (default is 100).
    base : int, optional
        Day count basis for year fraction calculation (default is 365 for ACT/365).

    Returns
    -------
    float
        The computed internal rate of return.

    Raises
    ------
    ValueError
        If no cash flows occur after the valuation date.
    RuntimeError
        If the iteration fails to converge within `max_iterations`.
    DivisionByZeroError
        Rare, but it could occur if new irr is estimated to be -1.0.
    """

    dates, cash_flows = zip(*[(dt, cf) for dt, cf in zip(dates, cash_flows) if dt > valuation_date])
    if len(cash_flows) == 0:
        raise ValueError("No cash flows after valuation date")

    cash_flows = np.array(cash_flows, dtype=np.float64)
    terms = np.array([(dt - valuation_date).days / base for dt in dates], dtype=np.float64)

    irr = initial_guess
    for _ in range(max_iterations):
        df_no_exp = 1.0 / (1.0 + irr)
        pvs = cash_flows * np.power(df_no_exp, terms)
        npv = pvs.sum() - pv # Error in valuation
        if abs(npv) < target_error:
            return float(irr)

        delta = - np.dot(terms, pvs * df_no_exp) # Derivative of NPV w.r.t. irr
        irr -= npv / delta # Newton-Raphson step: -f(x)/f'(x)
    
    raise RuntimeError(f"IRR calculation did not converge after {max_iterations} iterations. Final NPV error: {npv}")



# Matrix NS
def get_nelson_siegel_loadings(
    yfs: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """
    Calculates the Nelson-Siegel factor loadings matrix.
    
    Args:
        yfs: Array of maturities (M,)
        lambda_: The decay factor (scalar)
        
    Returns:
        Loadings matrix X of shape (M, 3)
    """
    yfs = np.asarray(yfs, dtype=float)
    
    lt = yfs / lambda_
    exp_lt = np.exp(-lt)

    L1 = np.ones_like(yfs)
    L2 = (1 - exp_lt) / lt
    L3 = L2 - exp_lt

    # Stack columns to create (M, 3) matrix
    return np.column_stack([L1, L2, L3])

def get_nelson_siegel_betas(
    yfs: np.ndarray,
    dfs: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """
    Calculates betas for all dates simultaneously using OLS matrix algebra.
    
    Args:
        yfs: Maturities (M,)
        dfs: Discount factors (N_dates, M)
        lambda_: Scalar decay factor
        
    Returns:
        Betas matrix of shape (N_dates, 3)
    """
    dfs = np.atleast_2d(dfs) 
    yfs = np.asarray(yfs)
    
    rates = -np.log(dfs) / (yfs[None, :] + 1e-10)
    X = get_nelson_siegel_loadings(yfs, lambda_)
    
    # 4. Solve OLS: Beta = Rates * X * (X.T * X)^-1
    # We use Pseudo-Inverse for numerical stability: X_pinv = (X.T @ X)^-1 @ X.T
    # X_pinv Shape: (3, M)
    X_pinv = np.linalg.pinv(X)
    
    # 5. Matrix Multiplication to get Betas
    # (N, M) @ (3, M).T  -> (N, M) @ (M, 3) -> (N, 3)
    betas = rates @ X_pinv.T
    
    return betas

def get_nelson_siegel_rates(
    yfs: np.ndarray,
    betas: np.ndarray,
    lambda_: np.ndarray
) -> np.ndarray:
    X = get_nelson_siegel_loadings(yfs, lambda_)
    rates = betas @ X.T
    return rates

from scipy.optimize import minimize
def get_nelson_siegel_params(
    yfs: np.ndarray, 
    dfs: np.ndarray, 
    lambda_: float = None,
    lambda_x0: float = 1.5,
) -> tuple[np.ndarray, float]:
    """
    Optimizes a global lambda and returns the corresponding betas.
    
    Args:
        yfs: Maturities (M,)
        dfs: Discount factors (N_dates, M)
        lambda_: If provided, skips optimization.
        lambda_x0: If provided, gives an initial guess for lambda optimization. Only used if lambda_ is None.
        
    Returns:
        tuple(betas_matrix, lambda_scalar)
        - betas_matrix: (N_dates, 3)
        - lambda_scalar: float
    """
    dfs = np.atleast_2d(dfs)
    yfs = np.asarray(yfs)

    # Case A: Lambda is provided
    if lambda_ is not None:
        betas = get_nelson_siegel_betas(yfs, dfs, lambda_)
        return betas, lambda_

    # Case B: Optimize Global Lambda
    def sse(x):
        lam = x
        betas = get_nelson_siegel_betas(yfs, dfs, lam)
        
        X = get_nelson_siegel_loadings(yfs, lam)
        model_rates = betas @ X.T
        
        model_dfs = np.exp(-model_rates * yfs[None, :])
        
        return np.sum((model_dfs - dfs)**2)

    res = minimize(
        sse, 
        x0=lambda_x0, 
        bounds=[(0.01, 5.0)], 
        method='L-BFGS-B'
    )
    
    if not res.success:
        print(f"Warning: Optimization did not converge completely: {res.message}")

    opt_lambda = float(res.x[0])
    opt_betas = get_nelson_siegel_betas(yfs, dfs, opt_lambda)
    
    return opt_betas, opt_lambda