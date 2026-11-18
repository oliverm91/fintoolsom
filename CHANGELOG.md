## v1.0.0b17 (Unreleased)

### BREAKING CHANGE

- **rates**: `IndexCurve` and `DiscountCurve` are replaced by one class, `PillarCurve(curve_date, knot_dates, forward_rates, interpolation_method, spot_rate=None)`. A curve no longer carries an identity (`index`, `currency`, `risk_free_index`) and `get_id()` is removed: the `Market` key gives it its role. `interpolation_method` has no default. The optional `spot_rate` (from `IndexCurve`) discounts from the curve date to a later first knot, so the builder's index curves keep their effective-date start; without it the first forward is extended back, and the first knot no longer has to be the curve date. `PillarCurve.from_dfs` and `from_zero_rates` build a curve from pillars. `get_accrual` moves to `Curve`.
- **rates**: `ZeroCouponCurve`, `ZeroCouponCurvePoint`, `LegacyCurve` and `InterpolationMethod.LogLinear` are removed; use `PillarCurve` (`PiecewiseConstant` is log-linear in the dfs, so `PillarCurve.from_dfs(t, dates, dfs, InterpolationMethod.PiecewiseConstant)` reproduces a log-linear `ZeroCouponCurve` up to its last pillar). Past the last pillar it extends the last forward flat, not the last zero rate.
- **fixedIncome**: `CLBond.get_present_value_zc`, `get_z_spread` and `get_irr_with_zcc` take any `Curve`. The z-spread is a parallel shift of the curve's continuously compounded Act/365 zero rates (`parallel_bump_bps`, on a copy) instead of each zero rate in its own convention, bumped and restored in place.
- **models**: `NelsonSiegelSvensson.calibrate_from_curve` takes a `PillarCurve` and fits its continuously compounded Act/365 zero rates at the knots after the curve date.
- **market**: `Market.curves` is keyed by `Index` (projects that index; the risk-free index's curve also discounts its currency) or `Currency` (discounts it under `risk_free_index`) instead of id strings (`"ICP"`, `"USD_ICP"`). `Market(curves=...)` takes that dict, no longer a list; `set_curve(key, curve)` takes the key and rejects other key types and the risk-free index's currency. `get_discount_curve`, `get_dfs` and `get_df` drop `risk_free_index=`: a market has one numeraire.

### Feat

- **market**: `Market.get_accrual(index, start, end)` projects an index over one period or aligned periods (its curve's wealth factor - 1); the calculator projects floating legs through it instead of fetching the index curve.

## v1.0.0b16

### BREAKING CHANGE

- **market**: `LegSpec.time_fraction` is removed. It was never read: fixed coupons and spreads accrue in the day count of their `Rate`, so a `time_fraction` other than the rate's was silently ignored. `FixedLegSpec` and `FloatingLegSpec` take `(currency, payment_frequency, adj_convention, rate | index, spread)`; a call that still passes a time fraction positionally raises `TypeError`.
- **market**: `Locality.EU` is `"XECB"` (the TARGET2 calendar of `holidays.financial_holidays`) instead of `"DE"` (German public holidays). EUR dates (ESTR, EURIBOR, EUR legs) now roll on TARGET2: 2025-05-29 (Ascension), 2025-06-09 (Whit Monday) and 2025-10-03 are business days.
- **dates**: `Days30EISDADayCountConvention` follows 2006 ISDA 4.16(h): an end date on the last day of February counts as day 30 unless it is the termination date. Before, it never adjusted it, while `TF_30E_360ISDA` always did; both now share one implementation. Pass `termination_date` to `get_day_count` / `get_time_fraction` (or `TF_30E_360ISDA(termination_date=...)`) to keep the actual day on a final February end. 31 Jan -> 29 Feb 2024 is 29 days on the termination date and 30 otherwise.
- **models**: `NelsonSiegelSvensson.lambda_` and `mu_` are the years to the first and second hump: each curvature loading is h(x) = (1 - e^-x)/x - e^-x at x = `HUMP_PEAK` * t / hump, which peaks at t = hump (`HUMP_PEAK` = 1.7933 solves x^2 + x + 1 = e^x). Before, the first hump used x = t / `lambda_` (peak at 1.79 * `lambda_` years) and the second x = `mu_` * t, a decay rate: with the bounds (2, 20) the second hump could only sit at 0.09-0.9 years, never at the long end. The bounds (0.5-3 and 2-20 years) and the starting humps (2 and 5 years) now mean what their comments say; a fresh model starts at `lambda_` = 2, `mu_` = 5 instead of 0.5 and 0.2. Fitted or stored `lambda_` / `mu_` values change meaning: refit them. A curve with humps at 1.5 and 12 years is now recovered (`lambda_` 1.499, `mu_` 11.998, rates within 2e-7) instead of ending on the bounds 8bp off.

### Feat

- **market**: `RateIndex` (`OvernightRateIndex`, `TermRateIndex`) takes a keyword-only `rate_convention`, the convention its rates are quoted in (linear Act/360 by default, as `InterestPriceIndex` already had), so every interest index carries one. It is not part of the index identity: indexes stay equal by name.
- **dates**: `Calendar` accepts the market codes of `holidays.financial_holidays` (`"XECB"`, `"XNYS"`, ...) besides country codes, and combines them with country calendars. `get_joint_calendar(*codes)` returns one cached combined calendar per tuple of codes.
- **rates**: day counts, `Rate.get_wealth_factor`, `get_discount_factor` and `get_accrued_interest` take an optional `termination_date`, which only 30E/360 ISDA reads. `FixedCoupon.termination_date` (keyword-only) passes it; `FixedLeg.from_term` and the floating legs' spreads set it to the last end date of the leg.

### Fix

- **market**: IRS, basis and cross-currency quotes (`_SwapQuote`) default `adj_convention` to modified following on the calendar of `locality` when given, else of the legs' currencies (joint for cross-currency), instead of a weekends-only calendar. `locality` was stored but never used. A forward quote's payment lag without a term moves on the pair's joint calendar. T+2 from 2026-09-17 for a CLP IRS is now 2026-09-22 (Fiestas Patrias), not 09-21; an explicit `adj_convention` is kept as given.
- **swaps**: a floating leg built without a spread (the builders and `from_term` with `spread=None`) gets a zero spread that is linear in the index's day count and base (`index.rate_convention`), instead of compounded Act/365. Values do not change, since the spread is zero, but the convention is what anything moving the spread uses: a 1bp bump of a CLP Cámara leg's default spread now accrues `1e-4 * days / 360`, not `1.0001 ** (days / 365) - 1`. An index without a `rate_convention` gets linear Act/360.
- **fixedIncome**: `Coupons.adjust_to_notional` also scales `interest` and recomputes each `flow` and `Coupons.flows`, and `CLBond` rescales a copy of the coupons instead of the caller's. A bond on coupons in a base other than 100 was valued with unscaled interests: TERA 87934.7 instead of 0.05063 on a 1,000,000 base. `Coupons.copy` keeps `check_residuals`.
- **fixedIncome**: `CLBond.get_dv01` is the value change for one basis point of the IRR: `-sum(tau * PV) / (1 + y)` for compounded IRRs (the modified duration), `-sum(tau * PV)` for exponential and `-sum(tau * PV / (1 + y * tau))` for linear, scaled to the notional. It used the Macaulay duration for every convention, which overstates a compounded DV01 by a factor 1 + y (5% at y = 5%). A matured bond has a DV01 of 0; other conventions raise `NotImplementedError`.
- **rates**: `ZeroCouponCurve.get_dfs` writes the results back through the masks of the dates before, inside and after the pillars, so unsorted dates get their own discount factors (`get_dfs([long, short])` gave `[0.888, 0.996]` instead of `[0.823, 0.997]`). It returns a copy of its cached array, so writing into a result no longer changes the curve.
- **market**: `UFForward.settlement_date` is the 9th of the month `next_n_cpi_settlements` months after `last_known_uf_date(quote_date)`, moved to the preceding business day (`CL` calendar) when it is not one. It was the CPI release date (one business day before the 9th) of `quote_date + n` months, which for a quote on a release day (2026-11-06, n=1) gave 2026-12-07 instead of 2027-01-08. `next_cpi_publication_date` is unchanged; `UFForward` no longer uses it.
- **dates**: `Calendar.add_custom_holiday` and `add_custom_holidays` record the date in `Calendar.custom_holidays` (once), so `combine` / `+` carries holidays added after construction and the repr counts them. Before, they only reached the underlying holiday calendar and were lost on combine.

### Perf

- **dates**: `Calendar` keeps per-year business-day tables (a business-day flag and a cumulative count per day), so `is_business_day` and `add_business_days` are lookups instead of `holidays` queries stepping one day at a time: `add_business_days(250)` 222 -> 0.50 us, `(2)` 1.69 -> 0.37 us. The tables are rebuilt after `add_custom_holiday`. Loading 743 ICP levels 1.08 -> 0.47 ms, 750 SOFR fixings 2.38 -> 1.18 ms.
- **market**: `OvernightInterestPriceHistory.add_value` fills only the gap between the new date's neighbours instead of rescanning the history: 298 adds 62.3 -> 1.33 ms, appending one day to 742 1.04 -> 0.02 ms.
- **market**: `get_rate(use_closest_past_rate=True)` finds the closest past date by bisection on cached sorted keys instead of scanning every date. With the calendar tables, one year of accrual on the closest past rate takes 1.39 -> 0.47 ms on daily fixings and 1.84 -> 0.73 ms on monthly ones.
- **market**: `FX_RateData` tracks its last date, so `get_fx_known_until` no longer runs `max` over the history on every non-deliverable flow: 73 instruments on 1,101 days of FX history 1.84 -> 1.63 ms.
- **rates**: a Hermite `ZeroCouponCurve` builds its `PchipInterpolator` once per set of pillars instead of on every `get_dfs`: pricing 39 bonds 4.86 -> 0.63 ms.
- **curve_builder**: the repricing check after a build reuses the `ValuationBatch` each group compiled for its solve instead of valuing the pinning quotes one by one.
- **curve_builder**: building the notebook curves (mean of 8 warm builds, the working tree against `HEAD` on the same machine): piecewise-constant 41.3 -> 38.1 ms, cubic spline 69.7 -> 67.6 ms, convex monotone 121.9 -> 119.6 ms; the test quote set 42.6 -> 39.6, 71.0 -> 67.2 and 124.7 -> 121.2 ms. The curves and the quotes' MTMs are bit-identical. A cold first build in a new process is about 5 ms slower (104.9 -> 109.7 ms).

## v1.0.0b15 (2026-11-16)

### BREAKING CHANGE

- **swaps**: a `SwapLeg` raises `ValueError` at construction when a coupon's start date is not before its end date, its end date is after its payment date, or its FX fixing date is after its payment date. The message names the first such coupon. Legs from the builders always met these conditions.

### Feat

- **curve_builder**: `build_curves` takes `report_currency`, `reprice_tol`, `sweep_tol` and `group_tol` as keyword arguments.
  - `report_currency` (CLP by default) is the currency the quotes' MTMs are valued in: the solve residuals, the repricing checks and the MTMs of `RedundantQuoteWarning`. The curves do not depend on it beyond rounding (1.3e-14 on the forwards, CLP against USD).
  - `reprice_tol` (`1e-2`; it was a fixed `1e-4`) is the largest |MTM| a fitted quote may keep after the build before `RepricingError`.
  - `sweep_tol` (`1e-4`; it was a fixed `1e-6`) is the largest |MTM| a date-by-date sweep may leave before its group is solved again jointly. Sweeps converge to about 1e-11 CLP on notional 100, so the new default changes no curve.
  - `group_tol` (`1`) is new: the largest sum of |MTM| of the fitted quotes of one solve group after the build, above which `build_curves` raises `RepricingError` naming the group's curves. Quotes that owned no pillar are not counted; they are only reported in `RedundantQuoteWarning`.
  - All three tolerances are absolute, in `report_currency`, and must be positive and finite (`ValueError` otherwise).
- **calculator**: `Calculator.valuate_many(instruments, market, currency)` values a list of instruments at once and returns one MTM per instrument, bit-identical to `Calculator.valuate` on each, errors included. `ValuationBatch(instruments, market, currency)` compiles the list once, so `values()` can be called again as the curves move. Swaps (`Swap` itself) and NDFs are compiled: each curve lookup (a discounting, an index curve, a projection or an FX pair) is evaluated once per call on every date the batch needs. Anything else, including swaps with a coupon already running or fixed, is valued by `Calculator.valuate`. `ValuationBatch.gradient(values)` gives the derivative of each MTM with respect to the values the batch reads off the market.

### Fix

- **dates**: the repr of a combined `Calendar` names its calendars in order (`Calendar(CL + US)`) instead of `Calendar(weekends only)`, and it counts the custom holidays of its inputs, which `combine` now carries in `custom_holidays`. A calendar with a subdivision shows its ISO 3166-2 code: `Calendar(US-NY)` instead of `Calendar(US, subdiv=NY)`.
- **curve_builder**: the unknowns of a group solve are laid out in curve-id order instead of the order of a frozenset, so a build no longer depends on `PYTHONHASHSEED`. Before, two runs could take different solver paths and give forwards that differed at rounding level.

### Perf

- **curve_builder**: the solver values its residuals through one `ValuationBatch` per instrument list, and updates the curves it built in place on each iteration (`PillarCurve._set_forward_rates`) instead of constructing and checking new ones.
- **curve_builder**: joint solves hand scipy's `hybr` the analytic Jacobian of the residuals instead of letting it difference them. It is the batch's gradient chained with the derivative of each curve's `log df` with respect to its forwards: constant over a solve on piecewise-constant, piecewise-linear and cubic-spline curves, and a forward difference per call on the monotone methods.
- **curve_builder**: the date blocks of a piecewise-constant or piecewise-linear sweep (1 to 3 unknowns) are solved by Newton steps with Broyden updates, starting from a differenced Jacobian or the block's own from the previous sweep. The solver falls back to `hybr` when the steps stop lowering the residuals, a residual is not finite or the valuation raises.
- **curve_builder**: the planner reads each instrument's pillar dates and last date once per build instead of on every query.
- **calculator**: a floating leg discounted on its own index curve (CSA on the leg index, or uncollateralised on the risk-free index), with no coupon started or fixed, is valued at par: `residuals[0] * df(first start)` plus the spreads discounted from the end dates, instead of projecting and discounting every coupon. It applies to legs with no payment lag, a contiguous schedule and the notional repaid through the amortizations, checked once at construction (`FloatingLeg._par_eligible`); `SimplifiedSwap` and `NonStartedSimplifiedSwap` assume it. The PV is the same up to rounding (about 1e-16 relative).
- **curve_builder**: building the notebook curves (`notebooks/quotes.json`; mean of 8 builds after 3 warm-ups, the 2 slowest of 10 dropped): piecewise-constant 60.5 -> 38.5 ms (1.6x), piecewise-linear 118.4 -> 65.8 ms (1.8x), Hermite cubic spline 361.8 -> 87.1 ms (4.2x), cubic spline 348.3 -> 63.9 ms (5.5x), convex monotone 376.2 -> 111.1 ms (3.4x). The discount factors move by at most 6.4e-11 relative on piecewise curves and 6e-15 on the others.

### Refactor

- **calculator**: leg PVs sum with `math.fsum` (correctly rounded) instead of `np.dot` / `np.sum`, whose BLAS summation order depends on the array's memory layout. A batch can then reproduce `Calculator.valuate` exactly. PVs move by at most an ulp.
- **tests**: the suite runs on 4 workers by default (`pytest-xdist`, `addopts = "-n 4"`); pass `-n 0` for a serial run. CI and the publish workflow run it on 2 workers, and publishing now runs the tests before building. The full suite takes 10.3 s instead of 19.0 s serially.

## v1.0.0b14 (2026-11-15)

### BREAKING CHANGE

- **swaps**: a `SwapLeg` whose coupons' start or payment dates decrease along `coupons` raises `ValueError` at construction. Coupons must be in schedule order; equal dates are allowed. Legs from the builders were always in order.

### Feat

- **swaps**: `FloatingLeg.fixed_flows` holds each coupon's flow known at construction: spread interest plus amortization. Valuation adds it to the projected or realised index interest.
- **swaps**: `SimplifiedSwap(receive_leg=, pay_leg=, collateral_index=)` is a `Swap` valued on a shorter path, for curve building. Every leg settles in its own currency (no `payment_currency` or `is_deliverable`; a UF leg is valued in UF and converted at spot) and pays on its accrual end dates: legs with a payment lag are copied with payment and FX fixing dates on their end dates. `NonStartedSimplifiedSwap` also values every coupon as unpaid and projected, reading no fixings, and raises `ValueError` when valued after its `effective_date` (the swap's first accrual start). `Calculator.valuate` routes both, through `Calculator.get_simplified_swap_mtm` and `Calculator.get_non_started_swap_mtm`. No builder returns them: wrap the legs of a built swap. Building the notebook curves with `NonStartedSimplifiedSwap` is about 10% faster and gives the same curves.

### Fix

- **calculator**: an overnight or cross-currency floating leg with a payment lag values every started, unpaid coupon. When the valuation date fell between a coupon's end and its lagged payment date, the next coupon, already accruing, was given the previous coupon's flow.
- **market**: the `Currency` constants (`Currency.USD`, `Currency.CLF`, ...) are declared as `ClassVar`s on the class, so type checkers know them. They are assigned after the class, and Pyright reported every use as an unknown attribute. Nothing changes at runtime.

### Perf

- **calculator**: `_leg_flows` returns the projected flows directly when no coupon of an overnight leg has started, or none of a term-rate leg has fixed, which covers the curve builder's instruments: about 15% faster there.

## v1.0.0b13 (2026-11-13)

### BREAKING CHANGE

- **market**: `quote_date` is a required, keyword-only field of `InstrumentQuote`, inherited by every quote. Forward quotes no longer accept `quote_date=None`, and swap quotes (`IRSQuote`, `IRBasisQuote`, `CrossCurrencyFixedFloatQuote`, `CrossCurrencyFloatFloatQuote`) no longer take it as their third positional argument: `adj_convention` is now third. Pass `quote_date=...` by keyword.
- **market**: `ForwardPriceQuote.to_outright` is removed: it ignored the spot and returned `value`. Only `ForwardPointsQuote.to_outright(spot)` remains.
- **market**: `uf.last_published_date` no longer takes a `calendar` argument. The CPI release calendar is always the Chilean one.

### Feat

- **dates**: `last_known_uf_date(t)` returns the last date whose UF is published as of `t`: the 9th of the current month before this month's CPI release, the 9th of next month from the release day on. `next_cpi_publication_date(t)` returns the next CPI release, one Chilean business day before `last_known_uf_date(t)`. `uf.last_published_date` calls `last_known_uf_date`.
- **market**: `UFForward(value, is_buy, next_n_cpi_settlements, quote_date=...)` quotes a UF forward as the outright UF level. It builds a `CLF/CLP` NDF that fixes and pays, with no payment lag, on `next_cpi_publication_date(quote_date + relativedelta(months=n))`, where `n` is from 1 to 24. It takes no `term`, `payment_date`, `fixing_date` or currency pair.

### Fix

- **curve_builder**: a group of quotes that reads a curve past its built end is solved after the ready group that extends that curve there. Before, UF forwards past the last uncollateralised ICP pillar were fitted on the extrapolated ICP curve, and the SOFR-collateralised CLP swaps solved after them moved them, so `build_curves` raised `RepricingError`.

## v1.0.0b12 (2026-11-13)

### BREAKING CHANGE

- **market**: `ForwardPriceQuote` and `ForwardPointsQuote` on a pair holding `CLF` raise `ValueError` at construction unless the quote is an NDF, meaning `fixing_date` is set or derivable from `term` and `quote_date`. They no longer become an NDF fixing on the payment date by default: pass `fixing_date` (the old default was `fixing_date=payment_date`).

### Fix

- **market**: `CrossCurrencyFixedFloatQuote.get_instrument(market=None)` sizes its legs at the market FX on `quote_date`: the pay notional is the base notional and the receive notional is the pay notional converted from the pay to the receive currency, as `CrossCurrencyFloatFloatQuote` does. Both legs were given the same notional before. Without a market, both legs keep the base notional. The curve builder passes the market to this quote.

## v1.0.0b11 (2026-11-12)

### BREAKING CHANGE

- **market**: `VolatilitySurface` takes `(days, deltas, vols, foreign_curve, interpolation_method)` instead of `(vol_surface_df, spot, domestic_curve, foreign_curve)`. `days` are integer days to maturity, `deltas` are spot put deltas in percent (`25`, not `0.25`) or `"atm"`, and `vols` is a `(len(days), len(deltas))` matrix. The `vol_surface_df`, `vol_surface_np`, `spot` and curve attributes are removed: spot and the domestic rate cancel in the pillar log-moneyness, and no curve is kept.
- **market**: `InterpolationMethod` and `InterpolationModel` are renamed `VolatilitySurfaceInterpolationMethod` and `VolatilitySurfaceInterpolationModel`. `DoubleCubicSpline` is replaced by `DoubleMonotoneCubic` (PCHIP), and the default method is `VannaVolga`.
- **market**: the two-step methods interpolate total variance in time at constant delta, with flat vol outside the tenors, and recompute each strike from its delta at the queried tenor. Values between tenors change, and so do eSSVI values, which now use ACT/365 like the option pricer instead of days/360.
- **swaps**: a deliverable `Swap` with a `CLF` leg raises `ValueError`, and so does a cross-currency quote with a `CLF` leg left deliverable (the default). The UF cannot be delivered: use `is_deliverable=False, payment_currency=Currency.CLP`.
- **deps**: pandas is no longer a dependency.

### Feat

- **market**: `VolatilitySurfaceInterpolationMethod.VannaVolga` is the Castagna-Mercurio second-order smile around the ATM quote, with no calibration. It goes through every quote. With more than three, the segments next to ATM use the 25-delta pair and each outer segment uses its own two quotes and ATM. The smile is flat beyond the outer quotes. `vanna_volga_volatility` is public.
- **market**: `VolatilitySurfaceInterpolationMethod.SVI` fits a raw SVI smile at each queried tenor, cached by days. Total variance is never negative and the wing slopes stay within Lee's moment bound. It needs five or more quotes and matches them exactly only when they are SVI-shaped, otherwise in the least-squares sense. `SVIParameters`, `svi_volatility`, `svi_total_variance` and `calibrate_svi` are public.
- **market**: `VolatilitySurface` validates its grid (integer, positive and unique days; deltas in `[1, 100)` or `"atm"`; finite, positive vols) and sorts its rows on a copy. A put delta above the foreign discount factor raises `ValueError` instead of giving NaN.
- **market**: `Market.get_future_fx_rate(t, currency_pair)` returns the FX fixing at `t`. It is the loaded rate up to `get_fx_known_until`, published UF levels included, and is projected on the funding curves after that. It raises `ValueError` unless `t` is after the market date. `Market.convert_value_currency` converts an amount at spot and `Market.convert_future_value_currency` at the fixing of a date, without building a `CurrencyPair`.
- **forwards**: `Forward` and `NDF` take a keyword-only `collateral_index` (default `None`), which discounts their flows under that CSA as swaps do. The NDF's FX projection does not depend on the CSA, so forward points are unchanged.

### Refactor

- **derivatives**: `derivatives/calculator.py` is now the `derivatives.calculator` package (`_fx`, `_forwards`, `_options`, `_swap_pv`, `_swap_risk`). `Calculator` keeps every method as a staticmethod alias, so `from fintoolsom.derivatives.calculator import Calculator` keeps working. Patching a `Calculator` method no longer changes the internal calls of the other functions.
- **derivatives**: NDFs, non-deliverable and UF swap legs and the curve builder project and convert FX through `Market`. `Calculator._convert`, `get_fx_fixings` and `_fx_fixings_at_ordinals` delegate to it, and valuations are unchanged.
- **scripts**: the `swap_valuation_from_quotes` scripts, their SQLite snapshot and the curve methods notebook are removed.

## v1.0.0b10 (2026-11-11)

### Feat

- **swaps**: every leg builder (`fixed_leg`, `term_rate_leg`, `overnight_leg`, `xccy_floating_leg`, `floating_leg`) and `from_term` takes `payment_lag`, the business days between each period end date and its payment date. It defaults to 0, so existing legs are unchanged. Accrual and projection stay on the period dates, and `fx_fixing_lag` is now measured back from the lagged payment date. A negative lag raises `ValueError`.
- **market**: `FixedLegSpec` and `FloatingLegSpec` take a keyword-only `payment_lag`, which the IRS, basis and cross-currency quotes pass to their legs. `build_curves` pins each funding curve at the lagged payment date, with no change to the builder itself.

## v1.0.0b9 (2026-11-11)

### BREAKING CHANGE

- **curve_builder**: `build_curves` takes one `interp_method`, used for discount and index curves alike, in place of `discount_interp_method` and `index_interp_method`. Non-local methods now make every group solve jointly, since index curves no longer sweep date by date next to them.

## v1.0.0b8 (2026-11-10)

### BREAKING CHANGE

- **rates**: `DiscountCurve` is created from `knot_dates` and `forward_rates` instead of pillars and discount factors, and its first knot must be the curve date. `DiscountCurve.from_dfs`, `dfs`, `pillar_dates` and `pillars` are removed, and so is `from_zero_coupon_curve` on both `DiscountCurve` and `IndexCurve`.
- **rates**: `forward_rates[i]` is the average forward over segment `i` for every interpolation method. An `IndexCurve` using the former `Linear` method read them as forwards at the knots, so the same inputs now give a different curve.
- **rates**: `InterpolationMethod.Linear` is renamed `PiecewiseLinear`.
- **rates**: `LogLinear` is accepted only by the legacy `ZeroCouponCurve`. `DiscountCurve` and `build_curves` default to `PiecewiseConstant`, which is the same curve shape.
- **rates**: `IndexCurve` and `DiscountCurve` need at least 2 segments (3 knots). `get_aged_curve` raises `ValueError` on or after the second-to-last knot.
- **curve_builder**: a build that would leave a curve with a single segment raises `InsufficientQuotesError`.

### Feat

- **rates**: `PillarCurve` is the new base of `IndexCurve` and `DiscountCurve`. Forward rates are the only parameters of a curve and discount factors are a result of them. Both curves accept every method except `LogLinear`, and share `get_aged_curve`.
- **rates**: `InterpolationMethod.ConvexMonotone` gives continuous forwards, quadratic by pieces, where a knot moves only the segments next to it. It is the monotone convex method of Hagan & West (2006) with QuantLib's enhancements at quadraticity 0.3 and monotonicity 0.7. Discount factors match QuantLib 1.44 up to the last knot. QuantLib's `forcePositive` floor is not included, so forwards can go negative between knots.
- **rates**: the first curve created with `PiecewiseLinear` in a process issues a `UserWarning` about its zig-zag, citing Hagan & West (2006).
- **curve_builder**: the solver always solves forward rates, for index and discount curves alike.
- **scripts**: restore `scripts/swap_valuation_from_quotes`, without `tweak_valuation.py`, and add `curve_methods.ipynb` to compare build error, build time, valuation error and ICP forwards across interpolation methods. USDCLP forwards are used up to 18 months and cross-currency basis swaps from 2 years, as explicit tenor limits.

### Fix

- **curve_builder**: every non-local interpolation method now gets the final joint solve. `HermiteCubicSpline` did not, and could raise `RepricingError` when a later pillar moved a quote already fitted. Builds with it take about three times as long on the sample quotes.
- **dates**: schedule dates are rolled from one anchor date, so a short month no longer shifts the later ones, and holidays are adjusted only after rolling. A whole number of periods has no stub even when the maturity is clamped to a shorter month. An end date equal to the adjusted tenor maturity rolls from the unadjusted one. An inner date adjusted onto its neighbour or the maturity is dropped, so no zero-length coupon is created.

## v1.0.0b7 (2026-11-10)

### BREAKING CHANGE

- **rates**: `Curve` declares the abstract method `get_aged_curve`, so a custom `Curve` subclass must implement it.

### Feat

- **rates**: `IndexCurve.get_aged_curve(date)` and `DiscountCurve.get_aged_curve(date)` return the curve as seen from `date` (after the curve date and before its last knot or pillar), with `df_aged(t) = df(t) / df(date)`. The aged curve matches the original exactly for `PiecewiseConstant` and `LogLinear`; spline methods match at the pillars and are refitted between them, and `Linear` keeps each remaining knot's forward with a flat first segment. An `IndexCurve` drops its spot rate.

## v1.0.0b6 (2026-11-09)

### Fix

- **market**: `OvernightRateHistory.add_rate` no longer depends on the order fixings arrive in. A fixing added on a gap-filled day now applies until the next fixing that was actually set, and changing a set fixing also updates the repeated fixings after it, as when the history is created with all its fixings at once. Accruals across such a gap change for histories loaded out of order.

### Perf

- **market**: `OvernightRateHistory.add_rate` updates only what a new fixing changes instead of rebuilding the history. On 1,250 fixings, changing an existing fixing or adding one before the first is 87-98% faster. Later levels are scaled, so they can differ from a full rebuild by about 1e-15 in relative terms.
- **models**: Nelson-Siegel-Svensson calibration sums bond values with `numpy`, 20-58% faster for 5 to 40 bonds.
- **fixedIncome**: bond present value, duration and DV01 sum their flows with `numpy`, up to 42% faster on a 30-year bullet.
- **dates**: `ActualDayCountConvention` computes date-to-list and list-to-date day counts from integer ordinals.

## v1.0.0b5 (2026-11-09)

### Feat

- **rates**: `IndexCurve` and `DiscountCurve` store the discount factor of every day up to their last knot or pillar on creation. The keyword-only `precalculate` argument (default `True`) turns this off, and `precalculate_dfs()` builds the grid later. `copy()` and `parallel_bump_bps()` keep the setting. After changing a curve's data in place, call `precalculate_dfs()` again.

### Perf

- **derivatives**: swap legs store their dates as integer ordinals, so a valuation converts no dates. Swap valuation is 55-68% faster and an all-quotes portfolio 56% faster.
- **derivatives**: option valuation uses `scipy.special` and no longer imports `scipy.stats`.
- **curve_builder**: curves with local interpolation skip the daily grid while solving and get it when the build ends, and cubic-spline curves keep it throughout. Each quote's instrument is built once. Builds are 36-58% faster.
- **rates**: `IndexCurve` no longer clips segment indexes or recomputes its last forward on every call, and `ZeroCouponCurve.get_dfs` uses a tuple as its cache key.
- **fixedIncome**: `Coupons.get_remaining_flows` and `get_remaining_end_dates` bisect the stored flows and end dates.
- **dates**: `Tenor` reuses its `relativedelta` steps, making schedule generation about a third faster.

Creating or bumping a curve is slower because of the grid: about 120 µs instead of 11 µs for a piecewise-constant index curve. Pass `precalculate=False` for curves created many times and read little.

## v1.0.0b4 (2026-11-09)

### BREAKING CHANGE

- **derivatives**: `Calculator.get_leg_pv`, `get_leg_duration` and `get_leg_dv01` take a `LegSide` (`LegSide.RECEIVE` or `LegSide.PAY`) instead of a leg, so they can only value the swap's own legs.
- **derivatives**: `Calculator.get_leg_dv01` raises `NotImplementedError` for a floating leg.

### Feat

- **derivatives**: add `LegSide` and `Swap.leg(side)`.
- **rates**: `InterpolationMethod` adds `CubicSpline` (natural spline on `log(df)`, for `DiscountCurve` and `IndexCurve`) and `Linear` (forwards linear between knots, `IndexCurve` only).
- **curve_builder**: `build_curves` accepts the new methods. With a `CubicSpline` curve every pillar is finally solved jointly.
- **derivatives**: `TermRateLeg` warns on creation when a payment date is a holiday of its index's calendar.

### Docs

- **derivatives**: add a `derivatives.swaps` README on the leg builders and custom swaps, and correct how amortizations are described.
- **curve_builder**: the README shows both interpolation arguments and an example of feeding quotes to `build_curves`, and describes the build more briefly.
- **scripts**: add `tweak_valuation.py` to compare the sample swap's valuation across curve-building settings, and set the sample quote conventions (ICP coupons from 1Y, SOFR coupons from 18M, SOFR collateral on ICP swaps from 2Y).

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

