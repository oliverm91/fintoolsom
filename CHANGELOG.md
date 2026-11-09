## v1.0.0b1 (2026-11-08)

### BREAKING CHANGE

- ProjectionCurve and NSSCurve are removed, as are
NelsonSiegelSvensson.get_curve and Market.get_curve, get_funding_curve,
get_discount_dfs, get_projection, projection_curves and funding_key.
Market.riskless_index is now risk_free_index and Market.curves is keyed by
curve id. build_curves takes discount_interp_method / index_interp_method.

### Feat

- **market**: rate convention of interest price indexes
- **curve_builder**: ownership-based build order, one pillar per quote; split into private modules
- **curve_builder**: one scheduler for discount and projection curves; date-by-date bootstrap
- **derivatives**: leg duration and DV01; factories, integration tests, coverage
- **derivatives,market**: ND settlement, UF as currency, exact swap valuation
- **curve_builder**: anchor price-based overnight curves via spot_overnight_rate
- **market**: concrete OvernightInterestPriceIndex + UFIndex.get_maturity
- **curve_builder**: register/bootstrap projection curves after discount build
- **rates**: ProjectionCurve forward-rate curve + get_projection seam
- **market**: index spot_lag (fixing->value offset); projection-curve design doc
- **market**: Index.get_maturity + locality calendar on Index
- **market**: type indexes as Overnight/Term; dispatch OvernightLeg; fix OIS projection
- **market**: build ForwardPointsQuote outright from the market spot
- **derivatives**: XCCY notional exchange + FX-matched leg notionals
- **curve_builder**: dependency-scheduled, log-df root solve of curve groups
- **curve_builder**: bootstrap discount/projection curves from instrument quotes
- **market**: term-based date derivation on forward quotes
- **market**: swap-level adj_convention, Term class, explicit start/maturity on SwapQuote
- **market**: add InstrumentQuote ABC; get_instrument() on all quote types
- **derivatives**: redesign swap hierarchy with typed legs and coupons
- **market**: add UF forward quote family
- **market**: split index definitions from data-bearing histories
- **market**: add PEN/Locality.PE; fix Locality.EU value to use DE calendar
- **market**: add ForwardQuote and SwapQuote hierarchy with leg specs
- **market**: redesign index hierarchy with InterestIndex, OvernightRateIndex, OvernightPriceIndex, TermRateIndex
- **market**: add LocalityCode enum and locality attribute to Currency, add Spot

### Fix

- **fixedIncome**: CLBond curve methods take ZeroCouponCurve; z-spread honours maxiter
- **market**: UFIndex.get_maturity returns the reajuste-closing 9th
- **notebooks**: instantiate ActualDayCountConvention for ICP index
- **market**: rename implied_monthly_cpi_between params, enforce base on 9th
- **dates**: handle unsupported bank holiday category gracefully
- **market**: repair Market dataclass and make Locality hashable
- **fixedIncome**: scale tera_value by notional in get_irr_from_amount

### Refactor

- **models**: NelsonSiegelSvensson.calibrate_from_bonds
- **rates**: readable rate convention names
- **rates**: IndexCurve and DiscountCurve replace ZeroCouponCurve in market, builder and derivatives
- **rates,market**: Curve interface, NSSCurve backend, riskless-index discounting
- **market**: overnight histories defer next date to their Index
- **curve_builder**: extract _CurveBuilder / _GroupSolve, unify helpers
- **market**: move term from RateIndex to TermRateIndex
- **derivatives**: leg-level swap builders, TermRateLeg fixing-date fix, market updates
- **market**: remove Spot, use FX_Rate in ForwardQuote.to_outright
- **market**: replace Locality dataclass and LocalityCode enum with a single Locality enum
- **swaps**: rename Ibor to Term throughout
- **derivatives**: strip valuation logic out of instruments into Calculator
- **fixedIncome**: merge Bond into CLBond with explicit constructor

### Perf

- **dates**: share one calendar per country

## v0.7.0 (2026-06-23)

### Fix

- **dates**: correct day adjustment order in 30/360 US convention
- **rates**: fix curve point filtering and scalar dates in ZeroCouponCurve
- **fixed-income**: correct return type annotation of Bond.get_price
- **fixed-income**: raise documented error when get_irr has no flows after valuation date

## v0.6.10 (2026-06-23)
