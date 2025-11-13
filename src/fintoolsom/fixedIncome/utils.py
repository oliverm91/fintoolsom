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