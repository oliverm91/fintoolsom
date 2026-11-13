# fintoolsom

[![PyPI version](https://img.shields.io/pypi/v/fintoolsom.svg)](https://pypi.org/project/fintoolsom/)
[![CI](https://github.com/oliverm91/fintoolsom/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverm91/fintoolsom/actions/workflows/ci.yml)
[![Lint](https://github.com/oliverm91/fintoolsom/actions/workflows/lint.yml/badge.svg)](https://github.com/oliverm91/fintoolsom/actions/workflows/lint.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/fintoolsom.svg)](https://pypi.org/project/fintoolsom/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Financial calculations in pure Python, specialised in the Chilean market (ICP, UF,
CLP/USD cross-currency): dates and calendars, rates and curves, curve
bootstrapping, CSA-aware derivatives valuation, Chilean bonds. Meant for personal use.

## Packages

| Package | Contents |
|---|---|
| [`dates`](src/fintoolsom/dates/README.md) | Calendars, holiday rules, business-day adjustments, day counts, tenors, schedules |
| [`rates`](src/fintoolsom/rates/README.md) | `Rate` and conventions; `IndexCurve` (forwards) and `DiscountCurve` (discount factors) |
| [`market`](src/fintoolsom/market/README.md) | Currencies, FX, indexes and fixings, the UF, quotes, volatility surfaces, `Market` |
| [`curve_builder`](src/fintoolsom/curve_builder/README.md) | `build_curves`: bootstraps every index and discount curve from quotes |
| [`derivatives`](src/fintoolsom/derivatives/README.md) | FX forwards and NDFs, swaps (IRS, OIS, basis, cross-currency, UF), FX options; `Calculator` |
| [`fixedIncome`](src/fintoolsom/fixedIncome/README.md) | Chilean bonds (`CLBond`), deposits, `get_irr` |
| [`models`](src/fintoolsom/models/README.md) | Nelson-Siegel-Svensson |

## Quick start

### Chilean bonds

A `CLBond` is its coupon table on base 100 ([`fixedIncome`](src/fintoolsom/fixedIncome/README.md)).
BTP0470930: CLP bullet issued 2018-03-01, 25 semi-annual coupons of 2.35.

```python
from datetime import date
from dateutil.relativedelta import relativedelta
from fintoolsom.dates import Days30EDayCountConvention
from fintoolsom.fixedIncome import CLBond, Coupon, Coupons
from fintoolsom.rates import LinearInterestConvention, Rate, RateConvention

accrual = RateConvention(LinearInterestConvention, Days30EDayCountConvention, 360)
issue = date(2018, 3, 1)
ends = [issue + relativedelta(months=6 * (i + 1)) for i in range(25)]
coupons = Coupons([
    Coupon(100.0 if end == ends[-1] else 0.0, 2.35, 100.0, start, end, accrual.copy())
    for start, end in zip([issue, *ends[:-1]], ends)
])
bond = CLBond(coupons, currency="clp", notional=380_000_000)

t, irr = date(2024, 7, 8), Rate(RateConvention(), 0.06)  # IRR: compounded Act/365
bond.get_price(t, irr)                     # (93.8945, 101.6537): price and par value, in %
bond.get_amount_value(t, irr)              # 362_699_638 CLP to pay
bond.get_duration(t, irr), bond.get_dv01(t, irr)    # 5.29 years, -192_012 CLP per bp
bond.get_irr_from_amount(t, 360_000_000)   # Rate of 6.1498%
```

### Derivatives

A market with given curves (curve building is also included, detailed [below](#curve-building)).
ICP is the risk-free index: CLP discounts on `"ICP"`, USD on `"USD_ICP"`
([`market`](src/fintoolsom/market/README.md), [`rates`](src/fintoolsom/rates/README.md)).

```python
from datetime import date
from fintoolsom.market import Currency, CurrencyPair, Market, OvernightInterestPriceIndex, OvernightRateIndex
from fintoolsom.market.currencies import FX_Rate, FX_RateData
from fintoolsom.rates import DiscountCurve, IndexCurve, Rate, RateConvention

t, y1, y5, y30 = date(2024, 7, 8), date(2025, 7, 8), date(2029, 7, 9), date(2054, 7, 8)
icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
sofr = OvernightRateIndex("SOFR", currency=Currency.USD)
usdclp = CurrencyPair(Currency.USD, Currency.CLP)
market = Market(
    t,
    risk_free_index=icp,
    fx_history={usdclp: FX_RateData(usdclp, {t: FX_Rate(usdclp, 930.0)})},
    curves=[  # index curves: one forward per segment; discount curve: one df per pillar
        IndexCurve(t, icp, [t, y1, y5, y30], [0.055, 0.050, 0.052]),   # "ICP": projects ICP, discounts CLP
        IndexCurve(t, sofr, [t, y1, y5, y30], [0.050, 0.042, 0.044]),  # "SOFR"
        DiscountCurve(t, Currency.USD, icp, [y1, y5, y30], [0.9522, 0.8065, 0.2592]),  # "USD_ICP"
    ],
)
```

#### Swaps

Legs come from `swaps.builders`; each swap carries its CSA ([`derivatives`](src/fintoolsom/derivatives/README.md)).

```python
from fintoolsom.dates import ModifiedFollowingConvention, Term, TermUnit
from fintoolsom.derivatives import Calculator
from fintoolsom.derivatives.swaps import LegSide, Swap, builders

cl, us = ModifiedFollowingConvention("CL"), ModifiedFollowingConvention("US")
clp_leg_descriptor = dict(notional=1e9, start_date=icp.calendar.add_business_days(t, 2),
                          term=Term(5, TermUnit.Y, cl), frequency="SEMIANNUAL", adj_convention=cl)
icp_irs = Swap(  # receive 5.3% vs ICP, ICP CSA
    receive_leg=builders.fixed_leg(rate=Rate(RateConvention(), 0.053), currency=Currency.CLP, **clp_leg_descriptor),
    pay_leg=builders.overnight_leg(index=icp, **clp_leg_descriptor),
    collateral_index=icp,
)
Calculator.valuate(icp_irs, market, Currency.CLP)                            # 2_428_464 CLP
Calculator.get_leg_dv01(icp_irs, LegSide.RECEIVE, market, Currency.CLP)      # -448_305 CLP per bp

usd_leg_descriptor = dict(notional=1e6, start_date=sofr.calendar.add_business_days(t, 2),
                          term=Term(2, TermUnit.Y, us), frequency="ANNUAL", adj_convention=us)
sofr_ois = Swap(  # receive SOFR vs 4.4%, SOFR CSA: discounted on the SOFR curve
    receive_leg=builders.overnight_leg(index=sofr, **usd_leg_descriptor),
    pay_leg=builders.fixed_leg(rate=Rate(RateConvention(), 0.044), currency=Currency.USD, **usd_leg_descriptor),
    collateral_index=sofr,
)
Calculator.valuate(sofr_ois, market, Currency.USD)   # 5_863 USD
Calculator.valuate(sofr_ois, market, Currency.CLP)   # 5_452_539 CLP, converted at spot
```

#### Forwards

```python
from fintoolsom.derivatives import NDF, Forward
fwd = Forward(1e6, 935.0, date(2025, 1, 8), is_buy=True, currency_pair=usdclp)  # buy 1M USD at 935
ndf = NDF(1e6, 935.0, date(2025, 1, 10), is_buy=True, currency_pair=usdclp, fixing_date=date(2025, 1, 8))
Calculator.valuate(fwd, market, Currency.CLP)   # -2_114_056 CLP: the fair forward is 932.83
Calculator.valuate(ndf, market, Currency.USD)   # -2_272 USD
```

## Curve building

`build_curves` fills `market.curves` so every quote that sets a pillar reprices to zero: IRS/OIS,
basis, cross-currency swaps and FX forwards ([`curve_builder`](src/fintoolsom/curve_builder/README.md)).

```python
from fintoolsom.curve_builder import build_curves
from fintoolsom.dates import TF_Actual360
from fintoolsom.market import FixedLegSpec, FloatingLegSpec, IRSQuote, PaymentFrequency, QuotedSide

def clp_ois(years: int, rate: float) -> IRSQuote:
    leg = dict(currency=Currency.CLP, payment_frequency=PaymentFrequency.SEMIANNUAL, adj_convention=cl,
               time_fraction=TF_Actual360())
    return IRSQuote(
        quoted_side=QuotedSide.RECEIVE, term=Term(years, TermUnit.Y, cl), quote_date=t, collateral_index=icp,
        fixed_leg=FixedLegSpec(**leg, rate=Rate(RateConvention(), rate)),
        floating_leg=FloatingLegSpec(**leg, index=icp, spread=Rate(RateConvention(), 0.0)),
    )

quotes = [clp_ois(y, r) for y, r in [(1, 0.055), (2, 0.054), (5, 0.053), (10, 0.054)]]
built = Market(t, risk_free_index=icp)
build_curves(quotes, built)        # built.curves == {"ICP": IndexCurve with 4 forward segments}
max(abs(Calculator.valuate(q.get_instrument(), built, Currency.CLP)) for q in quotes)  # ~1e-13
built.get_index_curve(icp).get_forward_rate(date(2029, 7, 10), date(2034, 7, 10))    # 5y5y forward
```

Quotes that add no pillar are reported in a `RedundantQuoteWarning`. `notebooks/main.py`
builds ICP, SOFR and USD curves from a full set of real 2026-06-29 quotes.

## Design notes

- **Curves.** `Market.curves` holds curves by id. Every curve discounts; index curves also project.
- **CSA collateral.** Swaps carry their CSA index, applied when discounting (`get_dfs(..., collateral=)`).
- **Legacy.** `ZeroCouponCurve` is kept for `CLBond` and Nelson-Siegel-Svensson only; it warns on creation.

## Install

`pip install fintoolsom`, or from source: `pip install git+https://github.com/oliverm91/fintoolsom.git --upgrade`.
Python 3.11 or later. Dependencies: numpy, scipy, holidays, python-dateutil.

## Development

- Install dependencies with [uv](https://docs.astral.sh/uv/), dev tools included: `uv sync`
- Run the test suite: `uv run pytest`
- Lint and format check: `uv run ruff check .` and `uv run ruff format --check .`
- Commits follow Conventional Commits; `CHANGELOG.md` is generated by commitizen.
