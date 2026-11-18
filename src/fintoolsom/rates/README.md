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
`get_forward_rate`, and projects: `get_accrual` (wealth factor - 1).
`parallel_bump_bps` returns a bumped copy that shifts every continuously compounded
zero rate.

A curve has no role of its own. The `Market` key it is stored under gives it one:
under an `Index` it projects that index (and discounts the index's currency when the
index is the risk-free one), under a `Currency` it discounts that currency under the
market's risk-free index.

`ModelCurve` is the family for parametric curves and has no implementation yet.
`PillarCurve` (`PillarCurve.py`) is the curve defined at dates:
`PillarCurve(curve_date, knot_dates, forward_rates, interpolation_method, spot_rate=None)`.
Its parameters are one forward rate per segment between knots, and discount factors
are only a result of them.

- `df(d) = df(first knot) · exp(−∫ f)`, with Act/365 continuous forwards from the
  first knot. When the first knot is after the curve date (the next business day for
  an overnight index, the spot lag of a term index), `spot_rate` discounts up to it
  in its own convention; without one, the first forward is extended back.
- `PillarCurve.from_dfs(curve_date, dates, dfs, method)` and
  `PillarCurve.from_zero_rates(curve_date, dates, rates, method)` build one from
  pillars; each zero `Rate` is read in its own convention.
- The last forward is extended flat past the last knot.
- Every pillar curve needs at least 2 segments (3 knots).
- `InterpolationMethod` only chooses the function integrated between knots, and every
  pillar curve accepts every method. `PiecewiseConstant` is log-linear in the dfs.
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
from fintoolsom.market import Currency, Market, OvernightInterestPriceIndex
from fintoolsom.rates import InterpolationMethod, LinearInterestConvention, PillarCurve, Rate, RateConvention

t, one_year, ten_years = date(2024, 7, 8), date(2025, 7, 8), date(2034, 7, 8)
r = Rate(RateConvention(LinearInterestConvention, ActualDayCountConvention, 360), 0.05)
r.get_wealth_factor(t, one_year)  # 1 + 0.05 * 365 / 360

pc = InterpolationMethod.PiecewiseConstant
icp_curve = PillarCurve(t, [t, one_year, ten_years], [0.05, 0.045], pc)
usd_curve = PillarCurve.from_dfs(t, [one_year, ten_years], [0.961, 0.69], pc)
icp_curve.get_forward_rate(date(2025, 1, 8), one_year)  # Rate, compounded Act/365 by default
icp_curve.get_accrual(date(2025, 1, 8), one_year)       # wealth factor - 1
usd_curve.parallel_bump_bps(1).get_df(date(2030, 1, 8))

icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
market = Market(t, risk_free_index=icp, curves={icp: icp_curve, Currency.USD: usd_curve})
```
