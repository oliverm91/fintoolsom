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

`ModelCurve` is the family for parametric curves and has no implementation yet.
`PillarCurve` (`PillarCurve.py`) is every curve defined at dates. Its parameters are
one forward rate per segment between knots, and discount factors are only a result of
them. It also holds the `InterpolationMethod`. `IndexCurve` and `DiscountCurve` are its
two roles.

| Class | `get_id()` | Stores | Interpolation |
|---|---|---|---|
| `IndexCurve` | `index.name` (`"SOFR"`) | knot dates, one average forward per segment, optional spot rate | any, `PiecewiseConstant` by default |
| `DiscountCurve` | `"<ccy>_<risk-free index>"` (`"USD_ICP"`) | knot dates from the curve date, one average forward per segment | any, `PiecewiseConstant` by default |

- **IndexCurve.** `df(d) = df_spot · exp(−∫ f)`, with Act/365 continuous forwards
  starting at the effective date. Before the effective date (the next business day
  for an overnight index) the spot `Rate` discounts in its own convention. It
  projects its index (`get_accrual`). When its index is the risk-free one, it also
  discounts that currency.
- **DiscountCurve.** A currency seen from the risk-free index of another currency,
  such as USD under ICP. Its first knot is the curve date. A currency has no
  DiscountCurve under its own risk-free index: the index's IndexCurve is used
  instead.
- Both curves extend the last forward flat past their last date.
- Every pillar curve needs at least 2 segments (3 knots).
- `InterpolationMethod` only chooses the function integrated between knots, and every
  pillar curve accepts every method. `LogLinear` exists only for the legacy
  `ZeroCouponCurve`.
- `PiecewiseLinear` forwards zig-zag: each knot forward is `2 * segment average -
  previous knot forward`, so a change in one segment reaches every later knot without
  decaying. Hagan & West (2006), "Interpolation Methods for Curve Construction",
  p. 97, reject the method for this reason. The first curve created with it in a
  process issues a `UserWarning`.
- `ConvexMonotone` keeps forwards continuous without that zig-zag: a knot moves only
  the segments next to it. It is the monotone convex method of Patrick S. Hagan and
  Graeme West, with Simon Ibbotson's QuantLib enhancements at quadraticity 0.3 and
  monotonicity 0.7. Forwards can go negative between knots, because QuantLib's
  `forcePositive` floor is not included.

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
usd_curve = DiscountCurve(t, Currency.USD, icp, [t, one_year, ten_years], [0.04, 0.035])
icp_curve.get_id(), usd_curve.get_id()  # ('ICP', 'USD_ICP')
icp_curve.get_forward_rate(date(2025, 1, 8), one_year)  # Rate, compounded Act/365 by default
usd_curve.parallel_bump_bps(1).get_df(date(2030, 1, 8))
```

## Legacy

`ZeroCouponCurve` (built on `LegacyCurve`) is kept only for `fixedIncome.CLBond`
and `models.NelsonSiegelSvensson`. It emits a `DeprecationWarning` when created,
and `Market` refuses to store it.
