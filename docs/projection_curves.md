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

## 4. Interpolation math (pluggable)

The projection curve must expose **one** primitive:

```
get_forward_wf(start: date, end: date) -> float      # compounded wealth factor over [start,end]
# (+ a get_forward_rate(start, end, convention) wrapper)
```

The *interpolation strategy* decides how the internal representation answers that query:

- **Piecewise-constant forward ("step-ladder")** — default. Also the natural choice for OIS
  (FOMC steps) and single-tenor term forwards.
- **Monotone convex (Hagan–West)** — forward-preserving, keeps forwards positive/stable, avoids
  the sawtooth of smooth-DF interpolation.
- **Piecewise-linear zero / log-linear DF** — for parity with the discount side if wanted.

Design the *interface* for a pluggable strategy now; commit to piecewise-constant first.

## 5. The overnight worry: does a 1Y coupon interpolate 252 times?

**No.** Two independent reasons, and the step representation makes it explicit:

1. **Integrate the steps, don't simulate days.** The compounded WF over `[start, end]` is
   obtained by walking only the **segments that overlap the window** and accumulating
   `rate × day_count_in_segment` (then compounding). For a 1Y overnight coupon with, say,
   FOMC-dated or monthly steps that is **~8–12 segments**, not 252. Because piecewise-constant
   collapses runs of equal daily forward into one segment, `get_forward_wf(start, end)` is
   `O(#segments ∩ window)`. This is exactly why "steps with a rate + start/end date" is the right
   shape — asking the curve for an accrual/equivalent rate between two dates is just integrating
   a step function.
2. **The df-ratio shortcut is O(1).** Under the standard OIS result the daily-compounded WF
   telescopes to `pseudo_df(start) / pseudo_df(end)` — a single ratio — which is what today's
   `get_wfs_fwds(start, end)` already returns. A ProjectionCurve can keep a cumulative
   pseudo-DF at each knot so the query is `O(1)` regardless of tenor.

Only a *naïve daily simulation* would be 252 lookups; neither representation above does that.
`_leg_pv` keeps calling one method (`get_forward_wf(start, end)`) for both `TermRateLeg` and
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

`index.spot_lag` and `Index.get_maturity` (both now on the index) provide exactly what this needs.

## 7. How `_leg_pv` and the bootstrap consume it

- **Valuation** (`Calculator._leg_pv`): projection comes from `market.projection_curves[leg.index]`
  via `get_forward_wf(start, end)`; discounting stays on the `(riskless/collateral, currency)`
  discount curves. `TermRateLeg` and `OvernightLeg` call the same projection primitive.
- **Bootstrap** (`build_curves`):
  1. Discount/OIS + FX curves as today.
  2. For each rate index, bootstrap its `ProjectionCurve` **given** the discount curve — each swap
     pins the forward segment(s) it introduces; the short end is anchored from the spot fixing (§6).
  3. Segments align to reset dates; the scheduler already orders "discount before projection".

## 8. Migration

1. Introduce `DiscountCurve` (keep `ZeroCouponCurve` as-is or alias) and add `ProjectionCurve`
   with the piecewise-constant strategy + `get_forward_wf`.
2. Add `market.projection_curves: dict[Index, ProjectionCurve]`; route `_leg_pv` projection there,
   discount unchanged. Curve-key equality for projection drops the currency dimension.
3. Bootstrap projection curves in `build_curves` (given discount), with the spot-fixing short-end
   anchor.
4. (Optional) pluggable forward interpolation (step vs monotone-convex).

Phases 1–3 can land incrementally; the current single-curve behaviour is preserved until each
index is given a real `ProjectionCurve`.

## 9. Open questions / risks

- **Interpolation choice** is product-dependent — design the strategy hook, don't hardcode.
- **Numeraire / normalisation** of the projection pseudo-DFs (self-consistent vs discount-tied).
- **Overnight step placement** (FOMC / turn-of-year) — where the knots go matters for the shape.
- For **par / plain-vanilla** valuation the single-smooth-curve answer is usually within ~1bp;
  this mostly earns its keep on forward-sensitive products, greeks/hedging, and the term short end.
- **Sensitivities**: step forwards give localised (bucketed) deltas — often a feature, not a bug.
