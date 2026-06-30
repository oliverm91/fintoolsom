# Curve Building
Create curve_builder module with empty `build_curves(list[InstrumentQuotes], riskless_index: Index) -> tuple[dict[Index, ZeroCouponCurve]], dict[tuple[Index, currency], ZeroCouponCurve]]`: raise `NotImplementedError`. But add docstring explaining that uses solver to find curves that minimizes market valuation error (all instruments should have mtm == 0).

A recursive method will define curves needed. For this 2 funcs are required:
- One that tells which curve will an instrument need to be valuated.
- One that tells which quotes of a given lists use a given curve.

A recursive or iterative process occurs until it finds an instrument or list of instruments that only depend on one curve. It starts the process there and reiterrates.

At some points, multicurve bootstrapping will be required. i.e. 2 curves A en B are required for instruments X and Y. At that point, solver must solve both curves at the same time.

Algorithm should detect if it needs more quotes to solve the problem. Example. It has a quote that require curves A and B and a quote that reuqires B and C. Multicurve solver will not work. Multicurve works on same amount of curves and quote groups.

Curves will have same amount of points as different tenors are in quotes.

When it finds a group of instruments to perform single or multicurve bootstrap it should try to go in small steps from shorter to longer duration to simplify solver problem.
  - As multiquote could have different amount of quotes and some quotes might differ in leg construction or maturity date due to calendars, "groupification" should be by similar tenors. Examples:
    - 2 quote groups: One jumps from 3Y to 5Y and other 3, 4, 5. Solve for 3Y together, then 5Y together. The one with 4Y includes it to the solver and curves involved in the 4Y quote will have that pillar.
    - 2 quote groups with 5Y: One is biannual, the other annual. Does not matter, quotes add pillars at end of maturity (`payment_date` for discount curves, `end_date` for forecasting).
    - 2 quotes groups with 5Y: One ends `t1`, other at `t2`. Difference is by calendar, adjustment or both. Pillar for curve affecting both is set to `max(t1, t2)`.

Note: Discount curve (ABCIndex, ABCIndex.currency) should be the same Projection curve ABCIndex. Example discount (ICPIndex, CLP) same as IndexProjection. Possible difficulty is different dates and used in curve building (`payment_date` for discount curves, `end_date` for forecasting). To ensure consistency, this is enforced and curve will use pillars with `max(t1, t2)` so forecasting curve doesn't have to extrapolate. They should not enter the solver as different objects. Maybe dict should point to the same object?
- If 3 group of instruments and 4 curves are needed it should not work. But if one is equivalent to other as ICPIndex and (ICPIndex, CLP) it should be considered as 3 curves and should work as finally only 3 curve objects enter the solver with 3 instrument groups.

Educated guess are needed:
- IRS: Ignore collateral. Directly use it's quote fixed `Rate`. Should work perfectly for Zero Coupon, for others it's good enough for solver.

Result is like:
 -> 
(
  {
    SOFR: Curve, # Used in collateral valuation and flow projection
    ICPIndex: Curve # Used for ICP projection
  }
  {
    (ICPIndex, CLP): Curve, # Discounts CLP flows
    (ICPIndex, USD): Curve # Discounts USD flows (known as USD curve in Chile)
  },
)

Result should be fed on every solver as it might be used in next iteration/rercursion step.