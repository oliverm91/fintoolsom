# fintoolsom

`fintoolsom` is a Python library designed for financial calculations, specializing in fixed income instruments, interest rate modeling, and derivatives. It provides a robust set of tools for valuation, analysis, and curve construction, with specific features tailored for the Chilean financial market.

## Core Features

### Fixed Income

-   **Generic Bond Valuation (`Bond`)**:
    -   Calculate Present Value (PV) from a given Internal Rate of Return (IRR) or a Zero-Coupon Curve.
    -   Determine the IRR (Yield-to-Maturity) from a given price/PV.
    -   Compute key risk metrics like `get_duration` (Macaulay duration) and `get_dv01`.
    -   Handle complex coupon structures and amortizations.
    -   Calculate accrued interest and par value for accurate pricing.

-   **Chilean Bonds (`CLBond`)**:
    -   Extends the generic `Bond` class with features specific to the Chilean market.
    -   Calculates the *Tasa de Emisión de Renta Anual* (TERA).
    -   Determines bond price and settlement amounts based on local conventions.
    -   Solves for IRR based on the traded amount.

-   **Deposits (`Deposit`)**:
    -   Simple valuation for fixed-term deposits.
    -   Calculates PV, duration, and DV01.

-   **Yield Curve Modeling (`NelsonSiegelSvensson`)**:
    -   Implements the Nelson-Siegel-Svensson model for yield curve fitting.
    -   Calibrates the model parameters (β₀, β₁, β₂, β₃, λ, μ) to a set of market bond prices or IRRs.
    -   Generates a `ZeroCouponCurve` object from the calibrated model, allowing for smooth and continuous yield curve representation.

### Interest Rates

-   **Rate Engine (`Rate`, `RateConvention`)**:
    -   A flexible `Rate` object that encapsulates both a rate value and its `RateConvention`.
    -   Supports various interest calculation methods:
        -   `LinearInterestConvention`
        -   `CompoundedInterestConvention` (Annual, semi-annual, etc.)
        -   `ExponentialInterestConvention` (Continuously compounded)
    -   Handles conversions between different rate conventions.

-   **Zero-Coupon Curves (`ZeroCouponCurve`)**:
    -   Constructs yield curves from a set of dates and discount factors or zero-coupon rates.
    -   Calculates discount factors (`get_df`), wealth factors, and forward rates between any two dates.
    -   Provides interpolation methods for points between curve nodes, including:
        -   `LogLinear`
        -   `HermiteCubicSpline` (PCHIP)
    -   Supports curve operations like parallel bumping and aging the curve to a future date (`get_aged_curve`).

### Dates and Calendars

-   Add standard tenors (1D, 1M, 1W, 2Y, ...).
-   Multiple day count conventions (Actual, 30/360, etc.).
-   Advanced holiday and calendar management, including rules for specific dates (e.g., 4th of July) and recurring events (e.g., third Monday of a month).
-   Pre-built calendars for New York (NY) and Chile (CL).
-   Date adjustment methods (Following, Modified Following, Preceding).

### Derivatives

-   **FX Forwards**: Valuation and analysis.
-   **Options**:
    -   Mark-to-Market (MtM) valuation and Greeks (Delta, Gamma, Vega, Theta).
    -   Volatility surface modeling and interpolation (DoubleLinear, DoubleCubicSpline, eSSVI).

## Installation

Install directly from the GitHub repository to get the latest version:

```bash
pip install git+https://github.com/oliverm91/fintoolsom.git --upgrade
```

## Requirements

-   Python: >=3.11
-   Packages: `numpy`, `scipy`
