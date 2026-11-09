## v1.0.0b3 (2026-11-09)

### BREAKING CHANGE

- **derivatives**: `Calculator.get_leg_pv`, `get_leg_duration` and `get_leg_dv01` take a `LegSide` (`LegSide.RECEIVE` or `LegSide.PAY`) instead of a leg, so they can only value the swap's own legs.
- **derivatives**: `Calculator.get_leg_dv01` raises `NotImplementedError` for a floating leg.

### Feat

- **derivatives**: add `LegSide` and `Swap.leg(side)`.
- **rates**: `InterpolationMethod` adds `CubicSpline` (natural spline on `log(df)`, for `DiscountCurve` and `IndexCurve`) and `Linear` (forwards linear between knots, `IndexCurve` only).
- **curve_builder**: `build_curves` accepts the new methods. With a `CubicSpline` curve every pillar is finally solved jointly.
- **derivatives**: `Swap` and the swap legs print as a compact summary. `str(swap)` shows maturity, settlement mode and collateral, then one line per leg with its kind, currency, notional and rate or spread.
- **dates**: `Calendar` has a short `__repr__`, so indexes and legs no longer print every holiday.

### Docs

- **scripts**: add a sample swap valuation from quotes in `scripts/swap_valuation_from_quotes`.

## v1.0.0b2 (2026-11-09)

### Fix

- **deps**: raise the `holidays` floor to 0.88 so the holiday calendars are up to date.

## v1.0.0b1 (2026-11-08)

### BREAKING CHANGE

- **derivatives**: instruments only hold data and valuation moved to `Calculator`. `Forward.get_mtm`, `NDF.get_mtm`, `Option.get_mtm`, `get_delta`, `get_gamma`, `get_vega`, `get_log_moneyness`, `get_strike_from_delta` and `SwapBase.valuate`, `get_pay_flows_value`, `get_receive_flows_value` are removed. Use `Calculator.valuate(instrument, market, ...)` or the matching `Calculator` method.
- **derivatives**: swaps are redesigned around typed legs and coupons. `DefaultSwap` and `FXCompensatedSwap` are replaced by `Swap`. The `*SwapLeg` classes are replaced by `SwapLeg`, `FixedLeg`, `FloatingLeg`, `OvernightLeg`, `TermRateLeg` and `XCCYFloatingLeg`. The `*SwapCoupon` classes are replaced by `SwapCoupon`, `FixedCoupon`, `FloatingCoupon`, `OvernightCoupon`, `TermRateCoupon` and `XCCYCoupon`.
- **derivatives**: `IndexedNDF` is removed. A UF forward is now a CLF/CLP `NDF`.
- **market**: index definitions are split from their data. `IborRateIndex`, `IborRateIndexData`, `IndexData` and `OvernightIndexData` are replaced by the index classes (`OvernightRateIndex`, `OvernightInterestPriceIndex`, `TermRateIndex`, ...) and the history classes (`OvernightRateHistory`, `TermRateHistory`, ...). "Ibor" is now "Term" throughout.
- **market**: `Market.add_index` takes a history and `Market.get_index` no longer takes a date.
- **market**: `Market.zero_coupon_curve_mapper` and `get_zero_coupon_curve` are replaced by `Market.curves` (keyed by curve id), `set_curve`, `get_index_curve`, `get_df` and `get_dfs`. `get_discount_curve` now takes `(currency, risk_free_index=...)`. `Market.add_currency_pair` is removed.
- **market**: `Locality` is now an enum instead of a dataclass, and `Locality.EU` uses the German calendar.
- **market**: `VolatilitySurface` and its interpolation classes move from `derivatives.options.volatility_surface` to `market.volatility_surface`.
- **models**: `NelsonSiegelSvensson.get_curve` is removed. Use `get_rate` or `get_df`.
- **fixedIncome**: `Coupon.validate_inputs` and `Coupons.validate_residuals` are now private.
- **dates**: `ScheduleGenerator.generate_schedule` has a new `end_date` parameter before `stub_first`, so positional callers must be updated.
- **rates**: rate conventions print in a readable form, for example `lin/act/360` instead of `lact360`, and interest conventions print as `lin`, `comp` and `exp` instead of `l`, `c` and `e`.

### Feat

- **curve_builder**: new `build_curves` bootstraps discount and index curves from market quotes, solving curves in dependency order. It raises `InsufficientQuotesError` and `RepricingError`, and warns with `RedundantQuoteWarning`.
- **rates**: new `Curve` interface with `IndexCurve` (forward-rate curve projecting an index) and `DiscountCurve` (pillar discount factors), including `parallel_bump_bps`. `InterpolationMethod` adds `PiecewiseConstant`.
- **market**: new quote family with `get_instrument()`: `IRSQuote`, `IRBasisQuote`, `CrossCurrencyFixedFloatQuote`, `CrossCurrencyFloatFloatQuote`, `ForwardPointsQuote` and `ForwardPriceQuote`.
- **market**: indexes carry currency, calendar, spot lag and `get_maturity`.
- **market**: the UF is a currency (`Currency.CLF`) quoted against CLP like any FX pair, with `Market.get_fx_known_until` and UF/CPI helpers in `market.uf`.
- **market**: add PEN and `Locality.PE`.
- **derivatives**: `Calculator` adds `get_swap_mtm`, `get_leg_pv`, `get_leg_duration` and `get_leg_dv01`.
- **derivatives**: started swaps are valued exactly, combining realised fixings with projected ones.
- **derivatives**: swaps support deliverable and non-deliverable settlement and cross-currency notional exchange.
- **derivatives**: new leg builders `fixed_leg`, `floating_leg`, `overnight_leg`, `term_rate_leg` and `xccy_floating_leg`.
- **dates**: add `Term`, `TermUnit` and `get_calendar`. Adjustment conventions accept a country code as well as a `Calendar`.

### Deprecated

- **rates**: `ZeroCouponCurve` is legacy, kept for `CLBond` and Nelson-Siegel-Svensson fitting, and emits a `DeprecationWarning`.
- **rates**: `Rate.rate_value` is an alias of `Rate.rate`.
- **models**: `NelsonSiegelSvensson.calibrate` is an alias of `calibrate_from_bonds`.

### Fix

- **market**: `Market(t=...)` could not be instantiated.
- **market**: `accrue_rates_custom_reset_days` did not accumulate periods.
- **dates**: correct Act/Act ISDA (infinite loop across year ends), Act/Act ICMA, Act/365L, NL/360, NL/365 (29 February boundary) and the 30U/360 end-of-month rule.
- **dates**: `ScheduleGenerator` dropped a date when `stub_first=False`.
- **dates**: correct the `MonthDayRule` Monday adjustment (Chilean Law 19.668).
- **dates**: countries without a bank holiday category no longer fail, and bank holidays are added only for CL and JP.
- **rates**: `ZeroCouponCurve.copy` and `get_aged_curve` dropped the interpolation method.
- **fixedIncome**: `CLBond.get_z_spread` ignored `maxiter`.

### Perf

- **dates**: legs share one calendar per country instead of rebuilding it.

## v0.7.1 (2026-06-24)

### Fix

- **fixedIncome**: scale tera_value by notional in get_irr_from_amount

### Refactor

- **fixedIncome**: merge Bond into CLBond with explicit constructor

## v0.7.0 (2026-06-23)

### Fix

- **dates**: correct day adjustment order in 30/360 US convention
- **rates**: fix curve point filtering and scalar dates in ZeroCouponCurve
- **fixed-income**: correct return type annotation of Bond.get_price
- **fixed-income**: raise documented error when get_irr has no flows after valuation date

## v0.6.10 (2026-06-23)

