# fintoolsom.rates

Interest rates and the curves used for pricing.

## Rates (`Rates.py`)

- `RateConvention(interest_convention, day_count_convention, time_fraction_base)`.
  The default is compounded Act/365.
- Interest conventions: `LinearInterestConvention`, `CompoundedInterestConvention`
  and `ExponentialInterestConvention` (continuous).
- `Rate(rate_convention, value)` gives wealth factors, discount factors, accrued
  interest and conversion to another convention. Every method takes a date or a
  list of dates.

## Curves

`Curve` (`Curve.py`) is the base of everything a `Market` stores. Every curve can
discount: `get_df(s)`, `get_df_fwd`, `get_wealth_factor`, `get_zero_rate`,
`get_forward_rate`. `parallel_bump_bps` returns a bumped copy that shifts every
continuously compounded zero rate. `get_id()` is the key the curve is stored under
in `Market.curves`.

| Class | `get_id()` | Stores | Interpolation |
|---|---|---|---|
| `IndexCurve` | `index.name` (`"SOFR"`) | knot dates, one forward per segment, optional spot rate | `PiecewiseConstant` |
| `DiscountCurve` | `"<ccy>_<risk-free index>"` (`"USD_ICP"`) | pillar dates and discount factors | `LogLinear` (default), `HermiteCubicSpline` |

- **IndexCurve.** `df(d) = df_spot · exp(−∫ f)`, with Act/365 continuous forwards
  starting at the effective date. Before the effective date (the next business day
  for an overnight index) the spot `Rate` discounts in its own convention. It
  projects its index (`get_accrual`). When its index is the risk-free one, it also
  discounts that currency.
- **DiscountCurve.** A currency seen from the risk-free index of another currency,
  such as USD under ICP. Its first pillar is the implicit `(curve_date, 1)`. A
  currency has no DiscountCurve under its own risk-free index: the index's
  IndexCurve is used instead.
- Both curves extend the last forward flat past their last date.
- `InterpolationMethod` is one enum for all curves. Each curve class rejects the
  methods it does not support.

## Example

```python
from datetime import date

from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.market import Currency, OvernightInterestPriceIndex
from fintoolsom.rates import DiscountCurve, IndexCurve, LinearInterestConvention, Rate, RateConvention

t, one_year, ten_years = date(2024, 7, 8), date(2025, 7, 8), date(2034, 7, 8)
r = Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 360), 0.05)
r.get_wealth_factor(t, one_year)  # 1 + 0.05 * 365 / 360

icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
icp_curve = IndexCurve(t, icp, [t, one_year, ten_years], [0.05, 0.045])
usd_curve = DiscountCurve(t, Currency.USD, icp, [one_year, ten_years], [0.96, 0.70])
icp_curve.get_id(), usd_curve.get_id()  # ('ICP', 'USD_ICP')
icp_curve.get_forward_rate(date(2025, 1, 8), one_year)  # Rate, compounded Act/365 by default
usd_curve.parallel_bump_bps(1).get_df(date(2030, 1, 8))
```

## Legacy

`ZeroCouponCurve` (built on `LegacyCurve`) is kept only for `fixedIncome.CLBond`
and `models.NelsonSiegelSvensson`. It emits a `DeprecationWarning` when created,
and `Market` refuses to store it. To convert one, use
`IndexCurve.from_zero_coupon_curve(zcc, index)` or
`DiscountCurve.from_zero_coupon_curve(zcc, currency, risk_free_index)`.
