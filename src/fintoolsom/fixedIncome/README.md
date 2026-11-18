# fintoolsom.fixedIncome

Chilean bonds, deposits and a generic IRR solver.

## Contents

| Name | What it is |
|---|---|
| `CLBond` | Chilean bond defined by its coupon table. Valued on base 100, then scaled to `notional`. |
| `Coupons`, `Coupon` | Coupon table: amortization, interest, residual, dates and the accrual `RateConvention`. |
| `Deposit` | Chilean time deposit (single payment): value, DV01 and duration at a given rate. |
| `get_irr` | IRR (YTM) of dated cash flows by Newton-Raphson, compounded Act/`base`. |

## CLBond

- `get_accrued_interest`, `get_par_value` and `calculate_tera` (the TERA, used for
  the par value).
- `get_present_value(date, irr)` and `get_price(date, irr)` give the PV and the
  Chilean price (percent of par value, rounded as the market rounds it).
- `get_irr_from_present_value(date, pv, rate_convention)` inverts the price.
- `get_amount_value` and `get_irr_from_amount` work on the traded amount.
- `get_duration_from_rate(date, irr)` (Macaulay, flows discounted at the IRR) and
  `get_dv01(date, irr)` are risk measures at an IRR.
- `get_duration_from_curve(date, curve)` is the duration on any `rates.Curve`:
  Σ t·flow·df / Σ flow·df (Fisher-Weil). On a flat continuous curve it equals
  `get_duration_from_rate` at that rate; `date` need not be the curve date. Both
  durations are in Act/365 years by default (`day_count_convention`,
  `time_fraction_base`), and a matured bond raises `ZeroDivisionError`.
- `get_present_value_zc`, `get_z_spread` and `get_irr_with_zcc` value the bond on
  any `rates.Curve`, e.g. a `PillarCurve` or `market.get_discount_curve(ccy)`. The
  z-spread is a parallel shift of the curve's continuously compounded Act/365 zero
  rates (`Curve.parallel_bump_bps`).

## Example

```python
from datetime import date

from fintoolsom.fixedIncome import get_irr

flows = [5.0, 5.0, 105.0]
dates = [date(2025, 1, 2), date(2026, 1, 2), date(2027, 1, 2)]
get_irr(flows, dates, valuation_date=date(2024, 1, 2), pv=100.0)  # about 0.05
```

## Notes

- **Strings, not `Currency`.** Currencies are plain strings here (stored
  lowercased), not `fintoolsom.market.Currency` objects.
- **Calibration input.** `models.NelsonSiegel` and `models.NelsonSiegelSvensson`
  `calibrate_from_bonds` take a list of `(CLBond, Rate)` pairs, and return per bond
  its `get_duration_from_curve` on the fitted curve and its IRR error in bp.
