# fintoolsom

`fintoolsom` is a Python library designed for financial calculations, specializing in fixed income instruments, interest rate modeling, and derivatives. It provides a robust set of tools for valuation, analysis, and curve construction, with specific features tailored for the Chilean financial market.

## Core Features

### Fixed Income

-   **Bond Valuation (`Bond`)**:
    -   A comprehensive class for valuing fixed income securities.
    -   Calculate Present Value (PV) from a given Internal Rate of Return (IRR) or a Zero-Coupon Curve.
    -   Determine the IRR (Yield-to-Maturity) from a given price or settlement amount.
    -   Compute key risk metrics like `get_duration` (Macaulay duration) and `get_dv01`.
    -   Handle complex coupon structures and amortizations.
    -   **Chilean Market Features**: Includes specialized methods for the Chilean market, such as calculating the *Tasa de Emisión de Renta Anual* (TERA) and determining settlement amounts based on local conventions.

-   **Deposits (`Deposit`)**:
    -   Simple valuation for fixed-term deposits.
    -   Calculates PV, duration, and DV01.

-   **Yield Curve Modeling (`NelsonSiegel`, `NelsonSiegelSvensson`)**:
    -   Implements the **Nelson-Siegel** and **Nelson-Siegel-Svensson** parametric models for yield curve fitting.
    -   Uses `numba` for JIT-compiled calculations for high performance.
    -   **Calibration**: Calibrates model parameters to fit a set of market bond prices or IRRs using optimization methods from `scipy`.
    -   **Curve Generation**: Generates a smooth, continuous `ZeroCouponCurve` object from the calibrated model, ideal for pricing and risk management.
    -   Structured into a base `NelsonSiegel` class and an inheriting `NelsonSiegelSvensson` class for the four-factor model.

### Interest Rates

-   **Rate Engine (`Rate`, `RateConvention`)**:
    -   A flexible `Rate` object that encapsulates both a rate value and its `RateConvention`.
    -   Supports various interest calculation methods:
        -   `LinearInterestConvention`
        -   `CompoundedInterestConvention`
        -   `ExponentialInterestConvention` (Continuously compounded)
    -   Handles conversions between different rate conventions.

-   **Zero-Coupon Curves (`ZeroCouponCurve`)**:
    -   Constructs yield curves from a set of dates and discount factors or zero-coupon rates.
    -   Calculates discount factors (`get_df`), wealth factors, and forward rates between any two dates.
    -   Provides interpolation methods for points between curve nodes, including:
        -   `LogLinear`
        -   `HermiteCubicSpline` (PCHIP)
    -   Supports curve operations like parallel bumping and aging the curve to a future date (`get_aged_curve`).

## Installation

Install directly from the GitHub repository to get the latest version:

```bash
pip install git+https://github.com/oliverm91/fintoolsom.git --upgrade
```

## Requirements

-   Python: >=3.11
-   Packages: `numpy`, `scipy`, `numba`