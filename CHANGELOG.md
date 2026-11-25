## v1.0.0 (2026-11-25)

First stable release since v0.7.1. It consolidates the changes of v1.0.0b1 to v1.0.0b21, which keep their detailed entries below. This section lists the net result: items a later beta replaced or removed are not repeated. **v1.0.0 is a major release with many breaking changes; read the migration notes under BREAKING CHANGE.**

### BREAKING CHANGE
#### Curves
- **rates**: `ZeroCouponCurve`, `ZeroCouponCurvePoint`, `LegacyCurve`, `InterpolationMethod.LogLinear`, `Market.zero_coupon_curve_mapper` and `get_zero_coupon_curve` are removed. Use `PillarCurve` (`PiecewiseConstant` is the log-linear shape; `PillarCurve.from_dfs` builds a curve from discount factors).
- **rates**: `IndexCurve` and `DiscountCurve` are one class, `PillarCurve(curve_date, knot_dates, forward_rates, interpolation_method, spot_rate=None)`. A curve carries no identity. `forward_rates[i]` is the average forward over segment `i` for every method. A curve needs at least 3 knots; `get_aged_curve` raises `ValueError` on or after the second-to-last knot. `Curve` declares the abstract `get_aged_curve`.
- **rates**: `InterpolationMethod.Linear` is renamed `PiecewiseLinear`.
- **market**: `Market.curves` is keyed by `Index` or `Currency` instead of id strings (`"ICP"`, `"USD_ICP"`), with `set_curve`, `get_index_curve`, `get_df` and `get_dfs`.
- **curve_builder**: `build_curves` takes one `interp_method` instead of `discount_interp_method` and `index_interp_method`, and raises `InsufficientQuotesError` when a build would leave a curve with a single segment.
#### Instruments, swaps and valuation
- **derivatives**: instruments only hold data; valuation moved to `Calculator` (`Forward.get_mtm`, `NDF.get_mtm`, `Option.get_mtm`, greeks, strikes from delta, ...). The FX option functions take the market and currency explicitly.
- **derivatives**: swaps are redesigned around typed legs and coupons. `DefaultSwap` and `FXCompensatedSwap` are replaced by `Swap`, and the `*SwapLeg` classes by `SwapLeg`, `FixedLeg`, `FloatingLeg`, `OvernightLeg`, `TermRateLeg`, ... `IndexedNDF` is removed: a UF forward is a CLF/CLP `NDF`.
- **derivatives**: `Calculator.get_leg_pv`, `get_leg_duration` and `get_leg_dv01` take a `LegSide` instead of a leg; `get_leg_dv01` raises `NotImplementedError` for a floating leg.
- **swaps**: a `SwapLeg` raises `ValueError` at construction when its coupons are out of order, a start date is not before its end date, an end date is after its payment date, or an FX fixing date is after its payment date. A deliverable `Swap` with a `CLF` leg raises `ValueError` (use `is_deliverable=False, payment_currency=...`).
- **market**: `LegSpec.time_fraction` is removed (it was never read).
#### Market, quotes and indexes
- **market**: index definitions are split from their data. `IborRateIndex`, `IborRateIndexData`, `IndexData` and `OvernightIndexData` are replaced by `OvernightRateIndex`, `OvernightInterestPriceIndex`, `TermRateIndex`, ... and their histories. `Market.add_index` takes a history; `Market.get_index` no longer takes a date. `RateIndex` takes a keyword-only `rate_convention`.
- **market**: `quote_date` is a required keyword-only field of every `InstrumentQuote`. `ForwardPriceQuote.to_outright` is removed. FX forward quotes default `spot_lag` to `CurrencyPair.spot_lag` (T+1 for USD/CLP and USD/CAD) instead of 2. Quotes on a pair holding `CLF` must be NDFs.
- **market**: `Locality` is an enum, and `Locality.EU` is the TARGET2 calendar (`"XECB"`) instead of German holidays, so EUR dates roll on TARGET2.
- **market**: `VolatilitySurface` and its interpolation classes move to `market.volatility_surface`, take `(days, deltas, vols, foreign_curve, interpolation_method)`, and are renamed `VolatilitySurfaceInterpolationMethod` / `VolatilitySurfaceInterpolationModel`. `DoubleCubicSpline` is replaced by `DoubleMonotoneCubic`; the two-step methods interpolate total variance, so values between tenors change.
- **market**: `uf.last_published_date` no longer takes a `calendar` (the CPI release calendar is the Chilean one).
#### Fixed income and models
- **fixedIncome**: `CLBond.get_present_value_zc`, `get_z_spread` and `get_irr_with_zcc` take any `Curve`; `get_duration` is renamed `get_duration_from_rate`. `Coupon.validate_inputs` and `Coupons.validate_residuals` are private.
- **models**: `NelsonSiegelSvensson.get_curve` and `calibrate` are removed (use `get_rate` / `get_df` and `calibrate_from_bonds`). `calibrate_from_bonds` and `calibrate_from_curve` are bounded least squares with analytic Jacobians and drop `method=`; `calibrate_from_bonds` weighs errors by duration by default (`duration_weighted=True`). `calibrate_from_curve` takes a `PillarCurve`. `lambda_` and `mu_` are the years to each hump, and the first hump calibrates down to 0.25 years.
#### Dates, rates and other
- **dates**: `ScheduleGenerator.generate_schedule` has a new `end_date` parameter before `stub_first`. `Days30EISDADayCountConvention` follows 2006 ISDA 4.16(h).
- **rates**: rate conventions print as `lin/act/360`, `lin`, `comp`, `exp` instead of `lact360`, `l`, `c`, `e`.
- **deps**: pandas is no longer a dependency; `holidays` floor raised to 0.88.

### Feat
- **curve_builder**: new `build_curves` bootstraps discount and index curves from market quotes in dependency order, with `report_currency`, `reprice_tol`, `sweep_tol` and `group_tol`. It raises `InsufficientQuotesError` and `RepricingError` and warns with `RedundantQuoteWarning`. New `build_curve_from_instruments` bootstraps a `PillarCurve` from instruments.
- **rates**: new `Curve` interface with `PillarCurve`, `ModelCurve` (a curve backed by any term-structure model), `parallel_bump_bps` and `get_aged_curve`. Interpolation: `PiecewiseConstant`, `PiecewiseLinear`, `CubicSpline`, `HermiteCubicSpline` and `ConvexMonotone`.
- **market**: quote family with `get_instrument()`: `IRSQuote`, `IRBasisQuote`, `CrossCurrencyFixedFloatQuote`, `CrossCurrencyFloatFloatQuote`, `ForwardPointsQuote`, `ForwardPriceQuote` and `UFForward`.
- **market**: the UF is a currency (`Currency.CLF`) quoted against CLP; UF/CPI helpers in `market.uf` and `last_known_uf_date`. New currencies `PEN` and `CAD`, `Locality.PE` and `Locality.CA`, `CurrencyPair.spot_lag`, `Market.get_accrual`, `Market.get_future_fx_rate`.
- **market**: volatility surfaces gain `VannaVolga` and `SVI` smiles and grid validation.
- **derivatives**: `Calculator.get_swap_mtm`, `get_leg_pv`, `get_leg_duration`, `get_leg_dv01` and `valuate_many`; `valuate(option, market, currency)` reports in any currency. Started swaps are valued exactly (realised plus projected fixings).
- **derivatives**: swaps support deliverable and non-deliverable settlement, cross-currency notional exchange (including `initial_exchange`), payment lags, `collateral_index` (also on `Forward` and `NDF`), `SimplifiedSwap`, and leg builders `fixed_leg`, `floating_leg`, `overnight_leg`, `term_rate_leg`, `xccy_floating_leg`. `LegSide` and `Swap.leg(side)`; compact `str(swap)`.
- **models**: `NelsonSiegel` (one hump) next to `NelsonSiegelSvensson`; `calibrate_from_instruments`, `calibrate_many_from_bonds`, `calibrate_many_from_curves`, `humps` / `estimate_humps`, and per-point fit errors in bps.
- **fixedIncome**: `CLBond.get_duration_from_curve` (Fisher-Weil).
- **dates**: `Term`, `TermUnit`, `get_calendar`, `get_joint_calendar`; calendars accept market codes such as `"XECB"` and `"XNYS"`; adjustment conventions accept a country code; `termination_date` for 30E/360 ISDA.

### Fix
- **dates**: Act/Act ISDA (infinite loop across year ends), Act/Act ICMA, Act/365L, NL/360, NL/365 (29 February), 30U/360 end-of-month and the Chilean Monday-holiday rule (Law 19.668); `ScheduleGenerator` dropped a date with `stub_first=False` and shifted dates after a short month; countries without a bank holiday category no longer fail.
- **market**: `Market(t=...)` could not be instantiated; `accrue_rates_custom_reset_days` did not accumulate periods; overnight fixings no longer depend on arrival order; `UFForward` settlement date; `Currency(CurrencyName.CLF)` default settlement currency.
- **fixedIncome**: `CLBond.get_z_spread` ignored `maxiter`; `tera_value` scaled by notional in `get_irr_from_amount`; `get_irr` tolerance scales with notional (>= 1e6); `get_dv01` is per bp of IRR; `Coupons.adjust_to_notional` also scales interest and flows.
- **curve_builder**: deterministic builds regardless of `PYTHONHASHSEED`; every non-local method gets the final joint solve; groups reading past a curve's built end are ordered correctly.
- **derivatives**: floating legs with payment lag value every started, unpaid coupon; unfixed term-rate coupons project over their fixing window; CLF/NDF and cross-currency leg sizing.
- **models**: Nelson-Siegel fits use a rank-truncated SVD and recover the curves bonds were priced on; empty or matured bond lists raise a clear `ValueError`; the caller's bond list is no longer sorted in place.

### Perf
- **models**: calibrations are 4-18x faster (least squares with analytic Jacobians, variable projection).
- **dates**: `Calendar` business-day tables make `is_business_day` and `add_business_days` lookups (`add_business_days(250)` 222 -> 0.50 us); legs share one calendar per country.
- **market**: faster overnight history updates and closest-past-rate lookups (bisection); FX history tracks its last date.
- **rates / curve_builder**: cached interpolators, curve discount factors precomputed, and the repricing check reuses each group's compiled `ValuationBatch`.

### Docs
- READMEs for `derivatives.swaps` and `curve_builder`; notebooks for curve methods.

### Tests
- Suite runs on 4 workers by default (`pytest-xdist`; `-n 0` for serial).
