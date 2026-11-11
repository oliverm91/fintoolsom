# fintoolsom.curve_builder

Bootstraps the curves of a `Market` from instrument quotes, so that every quote
that pins a pillar reprices to zero MTM.

```python
from fintoolsom.curve_builder import build_curves
from fintoolsom.rates import InterpolationMethod

build_curves(quotes, market)  # fills market.curves in place
build_curves(quotes, market, interp_method=InterpolationMethod.HermiteCubicSpline)
build_curves(
    quotes,
    market,
    interp_method=InterpolationMethod.PiecewiseConstant,
)
```

## What it builds

- **Index curves.** Every index a quote reads (collateral, floating legs) gets an
  `IndexCurve` under its name, such as `"ICP"` or `"SOFR"`.
- **Discount curves.** Every currency other than the risk-free index's gets a
  `DiscountCurve` under `"<ccy>_<risk-free index>"`, such as `"USD_ICP"`.
- **Interpolation.** New curves, discount and index alike, use `interp_method`:
  `PiecewiseConstant` forwards by default, or any other pillar-curve method. A
  curve already in the market keeps its own method when the build extends it past
  its last date.
- **Minimum quotes.** A curve needs at least 2 pillars. A solve that would leave a
  curve with one segment raises `InsufficientQuotesError`.

## What the market needs

- `market.t` equal to the quotes' `quote_date`. A mismatch raises `ValueError`.
- `market.risk_free_index`: the numeraire every discount curve is built under.
- FX spot for every pair a quote converts, and index histories for:
  - fixings that started coupons read;
  - the short end of overnight curves: today's fixing, or the last level ratio for
    ICP.

## Example: feeding quotes to the builder

A quote is a dataclass holding one instrument's conventions and its market level.
Fill the market with fixings and spot, list the quotes in any order, and call
`build_curves`. The builder works out which pillar each quote pins, so you don't
assign them. Besides the two types below there are `IRBasisQuote`,
`CrossCurrencyFixedFloatQuote`, `CrossCurrencyFloatFloatQuote` and
`ForwardPriceQuote`.

```python
from datetime import date

from fintoolsom.curve_builder import build_curves
from fintoolsom.dates import ActualDayCountConvention, ModifiedFollowingConvention
from fintoolsom.dates.time_fractions import TF_Actual360
from fintoolsom.market import (
    Currency, CurrencyPair, FixedLegSpec, FloatingLegSpec, ForwardPointsQuote, FX_Rate, IRSQuote, Market,
    OvernightInterestPriceHistory, OvernightInterestPriceIndex, OvernightRateHistory, OvernightRateIndex,
    PaymentFrequency, QuotedSide, Term, TermUnit,
)
from fintoolsom.rates import InterpolationMethod, LinearInterestConvention, Rate, RateConvention

t = date(2026, 11, 6)
icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
sofr = OvernightRateIndex("SOFR", currency=Currency.USD)
usdclp = CurrencyPair(Currency.USD, Currency.CLP)
convention = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)
cl, us = ModifiedFollowingConvention("CL"), ModifiedFollowingConvention("US")

market = Market(t, risk_free_index=icp)
market.add_index(OvernightInterestPriceHistory(icp, {date(2026, 11, 5): 100.0, t: 100.0125}))
market.add_index(OvernightRateHistory(sofr, {t: Rate(convention, 0.0365)}))
market.add_fx_rate(t, FX_Rate(usdclp, 945.0))

def ois(years, level, currency, adj, index, frequency, collateral):
    specs = (currency, frequency, adj, TF_Actual360())
    return IRSQuote(
        QuotedSide.RECEIVE, Term(years, TermUnit.Y, adj), t, adj, collateral_index=collateral,
        fixed_leg=FixedLegSpec(*specs, Rate(convention, level)),
        floating_leg=FloatingLegSpec(*specs, index, Rate(convention, 0.0)),
    )

quotes = [ois(y, r, Currency.CLP, cl, icp, PaymentFrequency.SEMIANNUAL, None) for y, r in [(1, 0.046), (2, 0.047), (3, 0.048)]]
quotes += [ois(y, r, Currency.USD, us, sofr, PaymentFrequency.ANNUAL, sofr) for y, r in [(1, 0.036), (2, 0.035), (3, 0.0355)]]
quotes += [
    ForwardPointsQuote(currency_pair=usdclp, value=points, is_buy=True, term=Term(months, TermUnit.M, cl), quote_date=t)
    for months, points in [(3, 2.0), (6, 3.5), (12, 6.0)]
]

build_curves(
    quotes, market,
    interp_method=InterpolationMethod.PiecewiseConstant,
)
list(market.curves)  # ['ICP', 'SOFR', 'USD_ICP']
```

## How it works

The build is split in private modules, driven by `builder._CurveBuilder`.

1. **Grouping.** Quotes that read the same set of curves form a group.
2. **Pillars.** Each quote pins at most one pillar, its maturity, on the first of
   its candidate curves where that date is still open. A quote with no open pillar
   is left out and reported in a `RedundantQuoteWarning`.
3. **Order.** A group solves once no other group still owns a pillar on a curve it
   reads. Groups waiting on each other are merged into one joint solve.
4. **Solving.** The unknowns are log discount factors on discount curves and segment
   forwards on index curves. With local interpolation a group is bootstrapped date
   by date. A spline curve is solved jointly with `scipy.optimize.root`, and with a
   `CubicSpline` every pillar is re-solved jointly at the end.

## Errors and warnings

| Name | When |
|---|---|
| `InsufficientQuotesError` | A group has more free pillars than quotes, reads a curve nobody builds, or a term index has no rate history |
| `RepricingError` | A pinning quote no longer reprices after the whole build |
| `RedundantQuoteWarning` | Quotes owned no pillar. `.ditched` holds `(quote, mtm, mtm_pct, mtm_bp)` |

## Notes

- **Quote order.** The order of the quotes changes neither ownership nor the
  resulting curves. Tests check this to `1e-9`.
- **Pillar dates.** Pillar dates are quote maturities, not a fixed grid. Two
  quotes of one group with the same maturity and no other open candidate both keep
  that pillar: the system is over-determined and the build fails. Drop one of them.
- **Bumping.** Bump a built curve with `market.set_curve(curve.parallel_bump_bps(1))`.
  Rebuilding from bumped quotes is the alternative for a quote-level risk view.
- **Not built.** No synthetic numeraire curves (such as `CLP_SOFR`) and no
  discounting on term indexes.
