# fintoolsom

[![PyPI version](https://img.shields.io/pypi/v/fintoolsom.svg)](https://pypi.org/project/fintoolsom/)
[![CI](https://github.com/oliverm91/fintoolsom/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverm91/fintoolsom/actions/workflows/ci.yml)
[![Lint](https://github.com/oliverm91/fintoolsom/actions/workflows/lint.yml/badge.svg)](https://github.com/oliverm91/fintoolsom/actions/workflows/lint.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/fintoolsom.svg)](https://pypi.org/project/fintoolsom/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Finantial calculations related library, specialized in Chilean market, meant for personal use.

Fully implemented in Python.

## Contents:
- Fixed Income:
  - Bonds (`get_irr`, `get_pv`, `get_dv01`, ...)
  - Chilean Bonds (`get_amount`, `get_tera`, ...)
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

- Derivatives: work in progress.

- Models:
  - Nelson-Siegel-Svensson curve calibration


## Install instructions:
`pip install fintoolsom`

Or directly from source: `pip install git+https://github.com/oliverm91/fintoolsom.git --upgrade`

## Requirements:
- Python: >=3.11
- Packages: numpy pandas scipy holidays python-dateutil

## Development
This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

- Install dependencies (including dev tools): `uv sync`
- Run the test suite: `uv run pytest`
- Lint and format check: `uv run ruff check .` and `uv run ruff format --check .`