# Plan: separate Projection curves from Discount curves

Status: **design / not implemented.** This documents the target design so the multi-curve
model can grow deliberately. No behaviour changes until each phase is picked up.

## 1. Why

Today a single `ZeroCouponCurve` is used for **both** discounting and projection, with the
same smooth DF interpolation (Pchip on DFs, or log-linear). That is fine for discounting but
wrong for forwarding:

- Smooth DF interpolation produces **oscillating / sawtooth forward rates** — bad for anything
  forward-sensitive (caps/floors, forward greeks, term-rate short ends).
- A projection curve built only from swaps ≥ 3M/6M has **no genuine information below its first
  pillar**, so a near-dated term coupon is projected off flat extrapolation of the first pillar
  (see the short-end discussion that motivated this doc).
- Discounting and forwarding are pinned by **different instruments** and want **different
  interpolation** and **different pillar placement**.

The plumbing for multi-curve already exists at the *curve-key* level (`(index, currency)` map,
and `_leg_pv` splits discount vs `leg.index` projection). What's missing is that both are the
**same class with the same interpolation**.

## 2. Two curve types

| | DiscountCurve | ProjectionCurve (a.k.a. ForwardCurve) |
|---|---|---|
| Purpose | discount a cashflow at its payment date | project the floating index's forward rate |
| Keyed by | `(riskless/collateral index, currency)` | **`Index` only** (currency is `index.currency`) |
| Pillars on | **payment / end dates** | **reset / start dates** |
| Parameterised by | zeros / DFs | **forward rates** (per reset) |
| Interpolation | smooth (current `ZeroCouponCurve`) | **forward-preserving** (see §4) |
| Pinned by | OIS/collateral instruments + FX | the index's swaps' floating legs (given a discount curve) |

`DiscountCurve` ≈ today's `ZeroCouponCurve` (keep it; possibly alias/rename). `ProjectionCurve`
is new.

### Keying: projection by `Index` only

A forward rate is intrinsic to the index — the index already carries its `currency`, tenor and
conventions — so there is exactly **one projection curve per index**:

```
market.projection_curves: dict[Index, ProjectionCurve]
market.curves:            dict[(Index, Currency), DiscountCurve]   # unchanged
```

Discount stays `(index, currency)` because the *same* collateral index discounts *multiple*
currencies (the collateral-adjustment construct in `Calculator._leg_pv`). Projection does not
have that degree of freedom.

## 3. Representation — piecewise-constant forward "steps"

A `ProjectionCurve` is a list of contiguous **segments**, each carrying a start date, an end
date and a forward rate (plus the rate convention / day count):

```
segment = (start_date, end_date, forward_rate)
curve   = [seg_0, seg_1, ...]   # contiguous, covering the modelled horizon
```

- For a **single-tenor term index** (e.g. TermSOFR 3M) the natural segments are the 3M forward
  periods aligned to the reset schedule: each step is "the 3M forward resetting at date X".
- For an **overnight index** the segments are the instantaneous-forward buckets — piecewise
  constant between knots (pillars, or FOMC/meeting dates, or turn-of-year points).

Other interpolation families (below) can be layered on the same segment idea; piecewise-constant
is the default because it keeps forwards stable and makes the accrual query trivial (§5).

### Knots — one per maturity, defined by the builder

Knots (segment boundaries) are chosen by `build_curves`, **not** derived one-per-coupon. A knot
per coupon breaks squareness: adding a 3Y quote to a 6M-frequency curve that already reaches 1Y
introduces coupons at 18M / 24M / 30M / 36M — **4 new coupons for 1 new quote** — so the forward
curve would gain 4 unknowns against 1 constraint and the solve goes under-determined
(`InsufficientQuotesError`, builder.py:329). Instead:

- **One knot per instrument maturity** → one new segment (hence one free forward) per new quote,
  keeping the projection solve **square** (N quotes → N segments). Intermediate coupon reset/end
  dates that fall inside an existing segment are read off by interpolation; they add no unknowns.
- **Knot placement (open question).** Put the knot at the instrument's **maturity (last end date)**
  so each new instrument extends the curve by exactly one segment `[prev_knot, maturity]` whose
  constant forward is the unknown it pins — the natural, sequential, square choice, and the working
  assumption. Reset/start-date knots are the alternative but complicate squareness when schedules
  don't line up, so we default to maturity-knots unless a reason to switch appears.
- The **first (spot) segment** starts at the short-end anchor (§6): `[t + spot_lag, spot maturity]`.

## 4. Interpolation methods (pluggable; piecewise-constant first)

A `ProjectionCurve` owns an **`interpolation_method`** — one of several — and the method is what
changes how the curve answers its two public queries between *any* two dates:

```
get_accrual(start, end) -> float                      # compounded factor the index earns over [start,end]
get_equivalent_forward_rate(start, end, convention)   # the single equivalent forward rate over [start,end]
```

The same knots, under different methods, answer these differently:

- **Piecewise-constant forward ("step-ladder")** — the **only method to implement now**. Also the
  natural choice for OIS (FOMC steps) and single-tenor term forwards. `get_accrual` integrates the
  step function over the window (§5); `get_equivalent_forward_rate` inverts that to one rate.
- **Monotone convex (Hagan–West)** — *later.* Forward-preserving, keeps forwards positive/stable,
  avoids the sawtooth of smooth-DF interpolation.
- **Piecewise-linear zero / log-linear DF** — *later*; for parity with the discount side if wanted.

Build the `interpolation_method` seam now (an enum + strategy, mirroring `InterpolationMethod` on
`ZeroCouponCurve`), so `get_accrual` / `get_equivalent_forward_rate` dispatch on it — but implement
**piecewise-constant only**.

## 5. The overnight worry: does a 1Y coupon interpolate 252 times?

**No.** Two independent reasons, and the step representation makes it explicit:

1. **Integrate the steps, don't simulate days.** The compounded WF over `[start, end]` is
   obtained by walking only the **segments that overlap the window** and accumulating
   `rate × day_count_in_segment` (then compounding). For a 1Y overnight coupon with, say,
   FOMC-dated or monthly steps that is **~8–12 segments**, not 252. Because piecewise-constant
   collapses runs of equal daily forward into one segment, `get_accrual(start, end)` is
   `O(#segments ∩ window)`. This is exactly why "steps with a rate + start/end date" is the right
   shape — asking the curve for an accrual/equivalent rate between two dates is just integrating
   a step function.
2. **The df-ratio shortcut is O(1).** Under the standard OIS result the daily-compounded WF
   telescopes to `pseudo_df(start) / pseudo_df(end)` — a single ratio — which is what today's
   `get_wfs_fwds(start, end)` already returns. A ProjectionCurve can keep a cumulative
   pseudo-DF at each knot so the query is `O(1)` regardless of tenor.

Only a *naïve daily simulation* would be 252 lookups; neither representation above does that.
`_leg_pv` keeps calling one method (`get_accrual(start, end)`) for both `TermRateLeg` and
`OvernightLeg` — the curve absorbs the tenor/step detail.

## 6. Short-end anchor (ties to `spot_lag`)

The spot fixing pins a **forward**, not a zero: `pseudo_df(t+spot_lag) / pseudo_df(t+spot_lag+term)`.
So the first segment of a projection curve is seeded from the index history's spot fixing over

```
[ t + index.spot_lag ,  index.get_maturity(t + index.spot_lag) ]
```

- Overnight: `spot_lag = 0` → `[t, t+1bd]` (the existing overnight anchor).
- Term (e.g. TermSOFR 3M, `spot_lag = 2`): `[t+2, t+2+3M]`, seeded from the live 3M print.
- The `[t, t+spot_lag]` front stub is *not* pinned by the term fixing — take it from the OIS
  curve (or treat a 2-day gap as negligible). This is why the projection bootstrap depends on the
  discount/OIS curve existing first (the fixpoint scheduler already enforces such ordering).

Crucially the spot fixing is a **published datum** (today's print), so the anchor segment is **fixed
data, not a free variable** — it removes one unknown and is precisely what turns the short end from
flat extrapolation into a determinate value. The first *solved* segment then runs from the anchor's
end (`spot maturity`) to the first quoted instrument's maturity, and anchor + solved segments tile
contiguously. A near-dated *unfixed* coupon whose accrual straddles that boundary reads the fixed
forward inside the anchor segment and the solved forward beyond — never an extrapolated one. A
coupon that has **already fixed** (`fixing_date ≤ t`) bypasses the curve entirely and uses the
index-history print, exactly as `_leg_pv` does today; the anchor only matters for the first
*unfixed* period.

`index.spot_lag` and `Index.get_maturity` (both now on the index) provide exactly what this needs.

## 7. How `_leg_pv` and the bootstrap consume it

- **Valuation** (`Calculator._leg_pv`): projection comes from `market.projection_curves[leg.index]`
  via `get_accrual(start, end)`; discounting stays on the `(riskless/collateral, currency)`
  discount curves. `TermRateLeg` and `OvernightLeg` call the same projection primitive:
  - `TermRateLeg` — each *unfixed* coupon (`fixing_date > t`) → `get_accrual(coupon.start, coupon.end)`;
    already-fixed coupons still read the index-history print (unchanged).
  - `OvernightLeg` — `get_accrual(coupon.start, coupon.end)` over the whole compounding period; the
    curve integrates its steps / uses the pseudo-DF ratio, so **no daily loop** (§5). A
    currently-accruing coupon splits into realised history `[start, t]` + `get_accrual(t, end)`.
  - Discounting is untouched: collateral-adjusted DFs from the `(riskless/collateral, currency)`
    curves — the projection change is orthogonal to it.
- **Bootstrap** (`build_curves`):
  1. Discount/OIS + FX curves as today.
  2. For each rate index, bootstrap its `ProjectionCurve` **given** the discount curve — each swap
     pins the one forward segment it extends the curve by; the short end is anchored from the spot
     fixing (§6). One residual (par / MTM = 0) per instrument keeps the solve square, reusing the
     existing `root` / least-squares machinery.
  3. Knots are one-per-maturity (§3); the scheduler already orders "discount before projection".
  4. Because forwards are localised to segments, a curve bump is a localised delta — risk buckets
     naturally by segment, which is usually what a desk wants.

## 8. Projection == discount when they coincide — one unknown, not two

The critical `build_curves` case: for an **overnight, self-discounted** index (OIS — e.g. SOFR
swaps collateralised in SOFR) the projection curve `projection_curves[SOFR]` and the discount curve
`curves[(SOFR, USD)]` are the **same curve** — SOFR both projects and discounts. There is no
forwarding-vs-discounting basis, so they must be **one unknown**. If the builder treats them as two
independent curves, each maturity yields two pillars for one quote, the group goes under-determined,
and `InsufficientQuotesError` (builder.py:329) fires constantly.

Rule:

- **Overnight index used as its own collateral/riskless (OIS self-discounting):**
  `projection_curves[I]` **is** the `(I, I.currency)` discount curve — a thin forward *view* over
  the same DFs (`get_accrual(s, e) = df(s)/df(e) − 1`), **not** a second set of pillars. Build once.
- **Term index, or an index discounted by a *different* collateral/currency (a real basis):**
  `projection_curves[I]` is an **independent** unknown — the forwarding curve differs from every
  discount curve — bootstrapped from the index swaps *given* the (already-built) discount curve.

Equivalently: a separate projection unknown exists **iff there is a forwarding/discounting basis**
for that index. Overnight-self-discounted → no basis → alias the discount curve. This is the same
distinction the collateral-cancellation logic in `_curves_needed` already encodes (it is exactly
why today's single `(SOFR,USD)` curve is one unknown, not two) — the projection split must reuse
that logic, not fight it.

**Detection & worked cases.** Reuse `_curves_needed`: an index aliases the discount curve exactly
when its instruments discount on that same index in its own currency (the cancellation already
collapses projection and discount to one key). Cases: (a) SOFR OIS collateralised in SOFR → aliased,
one curve; (b) TermSOFR-3M swaps collateralised in SOFR → the SOFR discount curve is already built
and the TermSOFR-3M *projection* is a genuine second unknown (a real basis) built on top of it;
(c) a USD-SOFR leg collateralised in CLP → discount `(ICP, USD)` ≠ projection SOFR → independent.
So the invariant the squareness check must enforce is: **total unknown curves = discount curves +
projection curves that carry a real basis.** `market.get_projection(index)` returns the wrapped
discount curve when aliased and a standalone `ProjectionCurve` otherwise, so callers never branch.
(ICP today is an overnight rate self-discounted in CLP, so its projection aliases `(ICP, CLP)` — no
separate projection unknown yet.)

## 9. Migration

1. Introduce `DiscountCurve` (keep `ZeroCouponCurve` as-is or alias) and add `ProjectionCurve`
   with the piecewise-constant strategy + `get_accrual`.
2. Add `market.projection_curves: dict[Index, ProjectionCurve]`; route `_leg_pv` projection there,
   discount unchanged. Curve-key equality for projection drops the currency dimension.
3. Bootstrap projection curves in `build_curves` (given discount), with the spot-fixing short-end
   anchor.
4. (Optional) pluggable forward interpolation (step vs monotone-convex).

Phases 1–3 can land incrementally; the current single-curve behaviour is preserved until each
index is given a real `ProjectionCurve`.

## 10. Open questions / risks

- **Interpolation choice** is product-dependent — design the strategy hook, don't hardcode.
- **Numeraire / normalisation** of the projection pseudo-DFs (self-consistent vs discount-tied).
- **Overnight step placement** (FOMC / turn-of-year) — where the knots go matters for the shape.
- For **par / plain-vanilla** valuation the single-smooth-curve answer is usually within ~1bp;
  this mostly earns its keep on forward-sensitive products, greeks/hedging, and the term short end.
- **Sensitivities**: step forwards give localised (bucketed) deltas — often a feature, not a bug.
