# fintoolsom.derivatives.swaps

A `Swap` is two legs, one received and one paid. A leg holds per-coupon arrays:
accrual start and end dates, payment dates and residual notionals. The builders
generate them from a start date, a term and a frequency. For anything they can't
generate, such as an amortizing notional or irregular dates, pass the arrays
yourself.

Instruments only hold data. Value them with `fintoolsom.derivatives.Calculator`, see
the [derivatives README](../README.md).

## Building a swap with the builders

Every builder makes a bullet leg: constant notional, repaid with the final coupon.

| Builder | Leg | Use for |
|---|---|---|
| `fixed_leg` | `FixedLeg` | Fixed rate. Takes `rate` and `currency` |
| `overnight_leg` | `OvernightLeg` | Overnight index compounded daily (ICP, SOFR) |
| `term_rate_leg` | `TermRateLeg` | Term-rate index, one fixing per coupon. Takes `fixing_lag` |
| `xccy_floating_leg` | `XCCYFloatingLeg` | Overnight leg of a cross-currency swap |
| `floating_leg` | Chosen by index | Overnight index gives `OvernightLeg`, or `XCCYFloatingLeg` with `cross_currency=True`. Any other index gives `TermRateLeg` |

All of them take these parameters:

| Parameter | Meaning |
|---|---|
| `notional` | Constant notional |
| `start_date` | First accrual start |
| `term` | Leg tenor, a `Term` |
| `frequency` | `"MONTHLY"`, `"QUARTERLY"`, `"SEMIANNUAL"` or `"ANNUAL"` |
| `adj_convention` | Business-day adjustment convention |
| `stub_first` | Keyword-only, default `True`. The irregular period goes at the start, or at the end when `False` |
| `long_stub` | Keyword-only, default `False`. Merge the irregular period into its neighbour |
| `maturity_date` | Keyword-only. Explicit maturity that overrides `term` |
| `fx_fixing_lag` | Keyword-only, default `0`. Business days between each FX fixing and its payment, for a non-deliverable leg in another currency |

Floating builders also take `index` and an optional `spread`, which is zero when
omitted.

A 2Y CLP swap, receiving fixed against ICP:

```python
from datetime import date

from fintoolsom.dates import ModifiedFollowingConvention, Term, TermUnit
from fintoolsom.derivatives import Calculator
from fintoolsom.derivatives.swaps import LegSide, Swap, builders
from fintoolsom.market import Currency, Market, OvernightInterestPriceIndex, OvernightRateIndex
from fintoolsom.rates import IndexCurve, Rate, RateConvention

t = date(2024, 7, 8)
icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
market = Market(t, risk_free_index=icp, curves=[IndexCurve(t, icp, [t, date(2034, 7, 8)], [0.05])])

mf = ModifiedFollowingConvention("CL")
start = icp.calendar.add_business_days(t, 2)
common = dict(notional=1e9, start_date=start, term=Term(2, TermUnit.Y, mf), frequency="SEMIANNUAL", adj_convention=mf)

fixed = builders.fixed_leg(rate=Rate(RateConvention(), 0.052), currency=Currency.CLP, **common)
floating = builders.overnight_leg(index=icp, **common)
swap = Swap(receive_leg=fixed, pay_leg=floating, collateral_index=icp)

Calculator.get_swap_mtm(swap, market, Currency.CLP)
Calculator.get_leg_pv(swap, LegSide.RECEIVE, market, Currency.CLP)
```

For a cross-currency swap, use one leg per currency and say how it settles:

```python
sofr = OvernightRateIndex("SOFR", currency=Currency.USD)
xccy = dict(start_date=start, term=Term(5, TermUnit.Y, mf), frequency="SEMIANNUAL", adj_convention=mf)
clp_leg = builders.xccy_floating_leg(notional=9e9, index=icp, **xccy)
usd_leg = builders.xccy_floating_leg(notional=1e7, index=sofr, **xccy)

non_deliverable = Swap(receive_leg=clp_leg, pay_leg=usd_leg, collateral_index=sofr, payment_currency=Currency.USD)
deliverable = Swap(receive_leg=clp_leg, pay_leg=usd_leg, collateral_index=sofr, is_deliverable=True)
```

## Building a custom swap

`Swap` accepts any legs, so you can build them from their dates and notionals. Every
leg takes these keyword arguments, one value per coupon:

| Field | Meaning |
|---|---|
| `start_dates`, `end_dates` | Accrual period, each start before its end |
| `residuals` | Outstanding notional each coupon accrues on |
| `payment_dates` | Optional. Date each flow is paid, on or after its end date. Defaults to `end_dates` |
| `fx_fixing_dates` | Optional. On or before each payment date. Defaults to `payment_dates` |

Each leg type adds its own fields:

| Leg | Adds |
|---|---|
| `FixedLeg` | `rate` (one `Rate` for every coupon), `currency` |
| `OvernightLeg`, `XCCYFloatingLeg` | `index`, optional `spread` (one `Rate`, zero by default) |
| `TermRateLeg` | as above, and optional `fixing_dates`, on or before each start date. Defaults to `start_dates` |

The amortizations come from the residuals: the notional is fully repaid by the last
coupon. Coupon `i` repays `residuals[i] - residuals[i + 1]`, and the last coupon
repays its residual. A negative amortization is a drawdown, so an accreting notional
works too. There is no initial notional exchange.

An amortizing swap on irregular dates, continuing the first example:

```python
from fintoolsom.derivatives.swaps import FixedLeg, OvernightLeg

dates = [start, date(2025, 1, 10), date(2025, 7, 10), date(2026, 1, 12)]
periods = {
    "start_dates": dates[:-1],
    "end_dates": dates[1:],
    "residuals": [1e9, 7.5e8, 5e8],  # repays 2.5e8, 2.5e8, then 5e8
}

custom_fixed = FixedLeg(**periods, rate=Rate(RateConvention(), 0.052), currency=Currency.CLP)
custom_floating = OvernightLeg(**periods, index=icp)
amortizing = Swap(receive_leg=custom_fixed, pay_leg=custom_floating, collateral_index=icp)

Calculator.get_swap_mtm(amortizing, market, Currency.CLP)
```

A leg is immutable and its arrays are read-only. Use `dataclasses.replace` for a
changed copy, for example `replace(custom_fixed, payment_dates=lagged_dates)`.

### Coupons

A leg doesn't store coupon objects. `leg.get_coupons()` and `leg.get_coupon(i)` build
them on request (`FixedCoupon`, `OvernightCoupon`, `XCCYCoupon` or `TermRateCoupon`),
so changing one doesn't change the leg. Code written against coupon lists can build a
leg with `from_coupons`:

```python
leg = FixedLeg.from_coupons(coupons, currency=Currency.CLP)
floating = OvernightLeg.from_coupons(floating_coupons, index=icp, spread=spread)
```

`from_coupons` raises `ValueError` when the coupons' amortizations don't repay the
notional as described above, or when a fixed leg's coupons don't share one rate. The
rate defaults to the coupons' rate, and a `TermRateLeg`'s `fixing_dates` default to
its coupons' fixing dates.

Keep these in mind when building legs by hand:

- **Dates are used exactly as given.** They aren't adjusted for business days, so pass
  dates that are already adjusted. A `TermRateLeg` warns on creation if a payment date
  is a holiday of its index's calendar.
- **Coupons must be in schedule order.** A leg raises `ValueError` if the start or
  payment dates go backwards, or if the dates of a coupon are inconsistent (a start
  not before its end, an end after its payment, an FX fixing after its payment).
- **One rate per leg.** A fixed leg has one rate and a floating leg one spread, for
  every coupon.
- **Floating flows always come from the index.** Only fixed legs pay known amounts.

## Simplified swaps for curve building

`SimplifiedSwap` and `NonStartedSimplifiedSwap` take the same legs as `Swap` and are
valued on a shorter path (`Calculator.valuate` routes them). They drop what a
curve-building quote does not need:

- **No settlement currency.** Every leg settles in its own currency, as in a
  deliverable swap. There is no `payment_currency` or `is_deliverable`, and a UF
  (`CLF`) leg is accepted: it is valued in UF on the UF curve and converted at spot,
  not paid in CLP at its fixing.
- **No payment lag.** Every coupon pays, and fixes FX, on its accrual end date. Legs
  built with a lag are copied with those dates moved; the legs you pass are not
  changed.
- **`NonStartedSimplifiedSwap` also ignores fixings.** Every coupon counts as unpaid,
  and every floating coupon is projected over its whole period, including a
  term-rate coupon whose fixing date has passed. Valuing it after its
  `effective_date` (the swap's first accrual start, not a curve's) raises
  `ValueError`. A swap starting before its index curve's effective date is fine:
  those days are projected on the curve's spot segment.

No builder returns them yet. Wrap the legs of a built swap:

```python
from fintoolsom.derivatives.swaps import NonStartedSimplifiedSwap

fast = NonStartedSimplifiedSwap(
    receive_leg=swap.receive_leg, pay_leg=swap.pay_leg, collateral_index=swap.collateral_index
)
Calculator.valuate(fast, market, Currency.CLP)
```
