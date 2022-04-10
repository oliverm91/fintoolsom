# fintoolsom

Finantial calculations related library meant for personal use.

## Contents:
- Fixed Income (function can contain more than one input): 
  - Bonds
  - Chilean Bonds
  - Chilean Deposits (under development)
- Rates (function can contain more than one input):
  - Rate objecte: interest calculation, rate convention convertions
  - Zero Coupon Curve object
- Dates (uses external libraries for performance, but handles different kind of inputs/outputs):
  - Add standard tenors (1D, 1M, 1W, 2Y, ...)
  - Day Count conventions (Actual, Days30, Days30E, Days30U, Days30ISDA, etc...)
  - etc...

## Install instructions:
Easly install with **pip**. See https://pypi.org/project/fintoolsom/

`pip install fintoolsom`

## Requirements:
- Python: >=3.8
- Packages: numpy python-dateutil mathsom
