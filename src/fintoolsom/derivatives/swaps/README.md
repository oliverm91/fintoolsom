# fintoolsom.derivatives.swaps

A `Swap` is two legs, one received and one paid. A leg is a list of coupons. The
builders generate that list from a start date, a term and a frequency. For anything
they can't generate, such as an amortizing notional or irregular dates, build the
coupons yourself.

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

`Swap` accepts any legs, so you can write the coupons directly. Every coupon takes
these fields, the first five in this order:

| Field | Meaning |
|---|---|
| `residual` | Outstanding notional the coupon accrues on |
| `amortization` | Notional repaid with this coupon |
| `start_date`, `end_date` | Accrual period |
| `payment_date` | Date the flow is paid |
| `fx_fixing_date` | Keyword-only. Defaults to `payment_date` |

Each coupon type adds its own field, and each leg is built from its own coupon type:

| Leg | Coupon | Coupon adds | Leg takes |
|---|---|---|---|
| `FixedLeg` | `FixedCoupon` | `rate` | `coupons`, `currency` |
| `OvernightLeg` | `OvernightCoupon` | `spread` | `coupons`, `index`, `spread` |
| `XCCYFloatingLeg` | `XCCYCoupon` | `spread` | `coupons`, `index`, `spread` |
| `TermRateLeg` | `TermRateCoupon` | `spread`, optional `fixing_date` | `coupons`, `index`, `spread`, `fixing_dates` |

An amortizing swap on irregular dates, continuing the first example:

```python
from fintoolsom.derivatives.swaps import FixedCoupon, FixedLeg, OvernightCoupon, OvernightLeg

dates = [start, date(2025, 1, 10), date(2025, 7, 10), date(2026, 1, 12)]
residuals = [1e9, 7.5e8, 5e8]
amortizations = [2.5e8, 2.5e8, 5e8]
periods = list(zip(dates[:-1], dates[1:], residuals, amortizations))

fixed_rate = Rate(RateConvention(), 0.052)
zero = Rate(RateConvention(), 0.0)

custom_fixed = FixedLeg(
    coupons=[FixedCoupon(res, amort, s, e, e, rate=fixed_rate) for s, e, res, amort in periods],
    currency=Currency.CLP,
)
custom_floating = OvernightLeg(
    coupons=[OvernightCoupon(res, amort, s, e, e, spread=zero) for s, e, res, amort in periods],
    index=icp,
    spread=zero,
)
amortizing = Swap(receive_leg=custom_fixed, pay_leg=custom_floating, collateral_index=icp)

Calculator.get_swap_mtm(amortizing, market, Currency.CLP)
```

A fixed leg has no field for an explicit amount, because a coupon's flow is its
interest plus its amortization. To pay exact amounts, set `residual` to zero and put
the amount in `amortization`. The `rate` is still required, but it has no effect:

```python
payments = [(date(2025, 1, 10), 3e7), (date(2026, 1, 12), 1.03e9)]
accrual_starts = [start, payments[0][0]]

explicit_leg = FixedLeg(
    coupons=[FixedCoupon(0.0, amount, s, d, d, rate=fixed_rate) for s, (d, amount) in zip(accrual_starts, payments)],
    currency=Currency.CLP,
)
```

Keep these in mind when building coupons by hand:

- **Nothing checks the notional.** Each amortization is paid as given. If the
  repayments don't add up to the notional, or `residual` doesn't follow them, the
  swap is still valued.
- **Dates are used exactly as given.** Coupons aren't adjusted for business days, so
  pass dates that are already adjusted.
- **There is no initial notional exchange.** Only the repayments in `amortization`
  are paid.
- **A floating leg uses the leg's `spread`.** The `spread` on each coupon is not used
  in valuation, so one floating leg can't vary its spread by coupon. A fixed leg can
  vary `rate` by coupon.
- **A `TermRateLeg` reads `fixing_dates` from the leg.** Pass one date per coupon. The
  `fixing_date` on the coupon is not used in valuation.
- **Floating flows always come from the index.** Only fixed legs can pay explicit
  amounts.
