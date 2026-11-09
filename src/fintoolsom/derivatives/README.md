# fintoolsom.derivatives

FX forwards, swaps and FX options, valued against a `fintoolsom.market.Market`.

Instruments only hold data. `Calculator` values them: every method is a
staticmethod that takes the instrument and the market, and nothing is cached. The
same instrument can be valued on any market, for example a bumped copy.

## Layout

| Path | Contents |
|---|---|
| `forwards/` | `Forward` (deliverable) and `NDF` (cash-settled at the FX rate of `fixing_date`) |
| `swaps/coupons.py` | `FixedCoupon`, `TermRateCoupon`, `OvernightCoupon`, `XCCYCoupon` |
| `swaps/legs.py` | `FixedLeg`, `TermRateLeg`, `OvernightLeg`, `XCCYFloatingLeg`: coupons plus cached per-coupon arrays |
| `swaps/builders.py` | `fixed_leg`, `overnight_leg`, `term_rate_leg`, `xccy_floating_leg`, `floating_leg` (chooses by index type) |
| `swaps/swaps.py` | `Swap(receive_leg=, pay_leg=, collateral_index=, payment_currency=, is_deliverable=)` |
| `options/` | `Call`, `Put` on an FX pair (Garman-Kohlhagen; work in progress) |
| `calculator.py` | `Calculator`: MTM, leg PV, FX fixings, duration and DV01, option greeks |

## Valuation

`Calculator.valuate(instrument, market, currency)` sends each instrument to its
pricer and reports the result in `currency`, converted at spot.

**Discounting.** Flows paid in currency `c` under collateral `k` (an index, or
`None`) are discounted with `market.get_dfs(c, dates, collateral=k)`:

- `k is None`: the uncollateralised discount curve of `c`. That is the risk-free
  index's own curve when `c` is its currency, and `"<c>_<risk-free>"` otherwise.
- `k` in currency `c`: the curve of `k`, as with a SOFR swap under a SOFR CSA.
- `k` in another currency: `disc(c) · curve(k) / disc(k.currency)`.

**Projection.** Floating coupons are projected on `market.get_index_curve(index)`:

- Term-rate coupon fixed on or before `t`: the published fixing over the whole
  coupon. Otherwise: the forward accrual of the index curve.
- Overnight coupon already running: the realised part `[start, t]` (compounded
  daily fixings, or the ICP level ratio) times the projected part `[t, end]`.
  Coupons not yet started are fully projected.
- Realised fixings come from `market.get_index(name)`. A missing fixing raises a
  `KeyError`.

**Settlement.** `Swap.leg_settlement_currency(leg)` is the leg's own currency for
a deliverable swap and `payment_currency` for a non-deliverable one. A UF (`CLF`)
leg always settles in CLP, at the UF of each flow's FX fixing date. Every flow
paid in another currency is converted at its FX fixing:

```
FX(d) = FX(L) · [df_base(d) / df_base(L)] / [df_quote(d) / df_quote(L)]
```

where `L = market.get_fx_known_until(pair)`: `t` for a traded pair, or the last
published UF date for CLF/CLP. Fixings on or before `L` are read from the history.

Only the final notional exchange is valued (the last coupon's amortization). There
is no initial exchange, and coupons paid on or before `t` are dropped.

## Risk

- `get_leg_pv(swap, side, market, currency)` gives the PV of the unpaid flows of
  one leg, with `side` being `LegSide.RECEIVE` or `LegSide.PAY`. `get_swap_mtm` is
  the receive leg minus the pay leg.
- `get_leg_duration(swap, side, market)` is the Macaulay duration in Act/365 years.
- `get_leg_dv01(swap, side, market, currency)` is `−PV · duration / 10 000`, for
  fixed legs only (a floating leg raises `NotImplementedError`). It is a
  first-order estimate: a full revaluation with
  `market.set_curve(curve.parallel_bump_bps(1))` matches it to first order.
- Forwards: `get_forward_mtm` and `get_ndf_mtm`. Options: `get_option_mtm`,
  `get_option_delta`, `get_option_gamma`, `get_option_vega` and the strikes from
  delta.

## Example: a 2Y CLP OIS on ICP

```python
from datetime import date

from fintoolsom.dates import ModifiedFollowingConvention, Term, TermUnit
from fintoolsom.derivatives import Calculator
from fintoolsom.derivatives.swaps import LegSide, Swap, builders
from fintoolsom.market import Currency, Market, OvernightInterestPriceIndex
from fintoolsom.rates import IndexCurve, Rate, RateConvention

t = date(2024, 7, 8)
icp = OvernightInterestPriceIndex("ICP", currency=Currency.CLP)
market = Market(t, risk_free_index=icp, curves=[IndexCurve(t, icp, [t, date(2034, 7, 8)], [0.05])])

mf = ModifiedFollowingConvention("CL")
start = icp.calendar.add_business_days(t, 2)  # spot start: a coupon running at t would read ICP fixings
common = dict(notional=1e9, start_date=start, term=Term(2, TermUnit.Y, mf), frequency="SEMIANNUAL", adj_convention=mf)
fixed = builders.fixed_leg(rate=Rate(RateConvention(), 0.052), currency=Currency.CLP, **common)
floating = builders.overnight_leg(index=icp, **common)
swap = Swap(receive_leg=fixed, pay_leg=floating, collateral_index=icp)

Calculator.valuate(swap, market, Currency.CLP)  # > 0: 5.2% annual beats the curve's ~5.13% (5% continuous)
Calculator.get_leg_dv01(swap, LegSide.RECEIVE, market, Currency.CLP)  # about -190k CLP per bp
```

## Notes

- **Options.** `Option` still names its curves through `domestic_index` and
  `foreign_index`. Its pricing has not been reviewed against the curve redesign:
  treat it as work in progress.
- **Report currency.** Reporting in a currency other than the settlement one
  converts at spot, so the market needs that pair's FX history.
- **Curve builder.** `fintoolsom.curve_builder` solves curves with
  `Calculator.valuate`: a quote reprices exactly when its MTM is zero on the built
  market.
