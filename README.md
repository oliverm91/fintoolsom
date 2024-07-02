# fintoolsom

Finantial calculations related library, specialized in Chilean market, meant for personal use.

Fully implemented in Python.

## Contents:
- Fixed Income:
  - Bonds (`get_irr`, `get_pv`, `get_dv01`)
  - Chilean Bonds (`get_amount`, `get_tera`)
  - Chilean Deposits
- Rates:
  - Rate object: interest calculation, rate convention convertions
  - Zero Coupon Curve object
- Dates:
  - Add standard tenors (1D, 1M, 1W, 2Y, ...)
  - Day Count conventions (Actual, Days30, Days30E, Days30U, Days30ISDA, etc...)
  - Generic Holidays: `MonthDayRule` (4th of july), `OrdinalWeekWeekdayRule` (Third monday of february), `easter` and others...
  - Date Adjustment methods (Following, Modified Following, Preceding, Modified Preceding)
  - Calendars
    - NY and Chilean Calendar pre-built

- Derivatives:
  - FX Forwards
  - Options
    - MtM and greeks
    - Volatility Surface interpolation models (DoubleLinear, DoubleCubicSpline, eSSVI)


## Install instructions:
`pip install git+https://github.com/oliverm91/fintoolsom.git --upgrade`

`pip install fintoolsom`  NOT UPDATED

## Requirements:
- Python: >=3.11
- Packages: numpy scipy multimethod