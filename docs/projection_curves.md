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
| Knots on | **payment / end dates** | **instrument maturity dates** (§3 Knots) |
| State / unknown | zeros / DFs | **forward rates** (per segment) |
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

## 3. Representation — a forward-rate function; interest is an integral

A `ProjectionCurve` represents a **forward-rate function `f(u)`** over the curve horizon, defined by
**knot values** plus a chosen **interpolation method** (§4). It is *not* one fixed shape — the earlier
"steps" wording described only the piecewise-constant special case.

Its core query **integrates** that function:

```
get_accrual(s, e)                = compound the index over [s, e]         # e.g. wealth factor − 1
get_equivalent_forward_rate(s,e) = the single rate equivalent to that accrual
```

With continuous compounding `WF(s, e) = exp(∫ₛᵉ f(u) du) = pseudo_df(s) / pseudo_df(e)`, where the
pseudo-DF is `pseudo_df(u) = exp(−∫ f)`. **The interpolation method decides the shape of `f`, and hence
how that integral is solved** — each has its own closed form:

- **piecewise-constant** forward → `∫ = Σ rateᵢ · Δtᵢ` over the knot intervals the window spans (the
  "step-ladder"; a run of equal forward collapses to one term).
- **piecewise-linear** forward → integrate a linear piece per interval (trapezoidal).
- **cubic-spline / cubic-hermite (Pchip)** forward → integrate the spline piece analytically per interval.

`get_accrual` is the single interface; the **integrator behind it is swapped by method**. Only
piecewise-constant is implemented first.

### Pseudo-DFs are derived; the state is forward rates

`get_accrual` reduces to a pseudo-DF ratio (`pseudo_df(s)/pseudo_df(e)`), so a `ProjectionCurve` **also
exposes `pseudo_df(u)`** — needed by DF-consumers (the collateral adjustment and FX in `_leg_pv`
multiply discount factors). But that pseudo-DF is a **derived** quantity: the curve's **state and
bootstrap unknowns are forward rates** (per segment), not pseudo-DFs.

Why not just store pseudo-DFs / wrap a `ZeroCouponCurve`:

- For **piecewise-constant** the two are equivalent — a constant segment forward ⟺ log-linear
  pseudo-DFs ⟺ a `ZeroCouponCurve` under `LogLinear`. So the integrator and `pseudo_df` are trivial,
  and the **aliased OIS case** (§7) is literally a forward *view* over the discount `ZeroCouponCurve`.
- For **richer forward interpolation** (piecewise-linear / cubic *forward*) the shape is chosen in
  forward space and does **not** correspond to any single DF-interpolation of a plain `ZeroCouponCurve`.
  Storing forwards is the representation that generalises; pseudo-DFs are only the output for DF-consumers.

Recommendation: `ProjectionCurve` state = **forward rates**; expose `get_accrual` /
`get_equivalent_forward_rate` / `pseudo_df` as the seam; **solve forward rates** in the bootstrap.

### Knots — one per maturity (knot-based interpolation)

This applies to **knot-based** interpolation (piecewise-constant / -linear / cubic-hermite / spline /
monotone-convex): a discrete knot set + a between-knot rule, which can reprice exactly when knots =
one-per-instrument. It does **not** apply to parametric forms (Nelson–Siegel / Svensson): those are a
few global parameters least-squares-*fit* to the quotes, with no per-maturity knots (out of scope here).

Knots (segment boundaries) are chosen by `build_curves`, **not** derived one-per-coupon. A knot per
coupon breaks squareness: adding a 3Y quote to a 6M-frequency curve that already reaches 1Y introduces
coupons at 18M / 24M / 30M / 36M — **4 new coupons for 1 new quote** — so the forward curve would gain 4
unknowns against 1 constraint and the solve goes under-determined (`InsufficientQuotesError`,
builder.py:329). Instead:

- **One knot per instrument maturity** → one new segment (hence one free forward) per new quote,
  keeping the projection solve **square** (N quotes → N segments). Intermediate coupon reset/end dates
  that fall inside an existing segment are read off by interpolation; they add no unknowns.
- **Knot placement (open question).** Put the knot at the instrument's **maturity (last end date)** so
  each new instrument extends the curve by exactly one segment `[prev_knot, maturity]` whose forward is
  the unknown it pins — the natural, sequential, square choice, and the working assumption. Reset/start-
  date knots are the alternative (one per coupon start is the same trap as one-per-coupon →
  under-determined), so we default to maturity-knots unless a reason to switch appears.
- **Solved as forward rates.** Each knot's unknown is the **forward rate on its segment**
  `[prev_knot, maturity]`, pinned by the instrument's par / MTM = 0 residual — one per instrument, so the
  system is square. (For piecewise-constant this is equivalent to solving a pseudo-DF at the maturity,
  but the curve's *state* is the forward, per the subsection above.)
- **Locality caveat.** Piecewise-constant and piecewise-linear are *local* → the last forward is pinned
  by the last instrument (sequential/triangular bootstrap). Cubic/spline are *non-local* → a node value
  bleeds into neighbouring intervals, so the solve is global (still square, not triangular; watch for
  oscillation).
- The **first (spot) segment** starts at the short-end anchor (§5): `[t + spot_lag, spot maturity]`.

## 4. Interpolation methods (pluggable; piecewise-constant first)

A `ProjectionCurve` owns an **`interpolation_method`** — one of several — and the method is what
changes how the curve answers its two public queries between *any* two dates:

```
get_accrual(start, end) -> float                      # compounded factor the index earns over [start,end]
get_equivalent_forward_rate(start, end, convention)   # the single equivalent forward rate over [start,end]
```

The same knots, under different methods, answer these differently:

- **Piecewise-constant forward ("step-ladder")** — the **only method to implement now**. Also the
  natural choice for OIS (FOMC steps) and single-tenor term forwards. `get_accrual` integrates the step
  function over the window; `get_equivalent_forward_rate` inverts that to one rate.
- **Monotone convex (Hagan–West)** — *later.* Forward-preserving, keeps forwards positive/stable,
  avoids the sawtooth of smooth-DF interpolation.
- **Piecewise-linear forward / cubic** — *later*; the non-local ones need a global bootstrap solve.

All the above are **knot-based** (§3 Knots). Parametric forms (Nelson–Siegel / Svensson) are a separate,
out-of-scope paradigm (global params, least-squares fit, no per-maturity knots).

Build the `interpolation_method` seam now (an enum + strategy, mirroring `InterpolationMethod` on
`ZeroCouponCurve`), so `get_accrual` / `get_equivalent_forward_rate` dispatch on it — but implement
**piecewise-constant only**.

### Configuration — defaults on `Market`, overridable per call

`interpolation_method` is threaded through `build_curves` and the curve store, but to keep call sites
clean the **`Market` holds the defaults, one per curve role**:

- `discount_interpolation_method`   = **LogLinear**
- `projection_interpolation_method` = **PiecewiseConstant**

Usage:

- `build_curves(..., interpolation_method=None)` — optional; when omitted, each curve is bootstrapped
  with the market default for its role (discount vs projection). An explicit value overrides both; a
  `{role-or-index: method}` mapping overrides selectively.
- The `Market` curve accessors carry the same optional kwarg defaulting to the role default, so the
  common case passes nothing:
  - discount:   `market.get_curve(index, currency, *, interpolation_method=None)` / `set_curve(...)`
  - projection: `market.get_projection(index, *, interpolation_method=None)` / `set_projection(...)`

**Consistency** (from §3 and the "can't swap post-hoc" rule): the method is **baked at bootstrap** — it
is the curve's calibration identity. So on **set / build** the resolved method *is* how the curve is
solved and stored; on **get** the accessor returns the as-built curve, and passing a *different* method
yields a **resampled view** (representation / what-if / warm-start) that does **not** reprice the
calibration instruments. Changing the calibration method means re-solving that curve — per-curve, scoped
by the dependency DAG, never the whole market.

**Phase-1 note.** Piecewise-constant is the equivalence point: a constant segment forward ⟺ log-linear
pseudo-DFs ⟺ the `ZeroCouponCurve` `LogLinear` kernel. So `get_accrual` / `pseudo_df` are one-liners and
the OIS-aliased projection reuses the discount `ZeroCouponCurve` (§7). Storage is still forward rates
(§3); the equivalence just makes the integrator trivial. Later forward-space methods break the
equivalence and need a real integrator.

## 5. Short-end anchor (ties to `spot_lag`)

The spot fixing pins a **forward**, not a zero: `pseudo_df(t+spot_lag) / pseudo_df(t+spot_lag+term)`.
So the first segment of a projection curve is seeded from the index history's spot fixing over

```
[ t + index.spot_lag ,  index.get_maturity(t + index.spot_lag) ]
```

- Overnight: `spot_lag = 0` → `[t, t+1bd]` (the existing overnight anchor).
- Term (e.g. TermSOFR 3M, `spot_lag = 2`): `[t+2, t+2+3M]`, seeded from the live 3M print.
- **The `[t, t+spot_lag]` front stub (the t → t+2 gap).** No term fixing informs it — but the projection
  curve is (almost) never *asked* about it. A coupon accruing inside `[t, t+spot_lag]` has
  `fixing_date = start − fixing_lag ≤ t`, so it has **already fixed** and is read from the index-history
  print, not projected. The first *unfixed* (projected) period is the spot period at `t + spot_lag`, so
  the curve's natural domain begins there: anchor its pseudo-DF at the spot date
  (`pseudo_df(t+spot_lag) = 1`) rather than at `t`, and the gap falls outside the curve.
- Where the stub genuinely *is* needed it is a **discounting** question, not projection: the OIS/discount
  curve has `spot_lag = 0` and covers `[t, …]` in full, so any DF inside `[t, t+2]` comes from there —
  which is why the projection bootstrap needs the discount curve to exist first (the fixpoint scheduler
  enforces that order). If a caller still asks the projection curve for a sub-`spot_lag` forward,
  extrapolate flat from the first segment or borrow the OIS forward for the stub; for a term index that
  path shouldn't arise, and two business days of it is sub-bp anyway.

Crucially the spot fixing is a **published datum** (today's print), so the anchor segment is **fixed
data, not a free variable** — it removes one unknown and is precisely what turns the short end from flat
extrapolation into a determinate value. The first *solved* segment then runs from the anchor's end
(`spot maturity`) to the first quoted instrument's maturity, and anchor + solved segments tile
contiguously. A near-dated *unfixed* coupon whose accrual straddles that boundary reads the fixed forward
inside the anchor segment and the solved forward beyond — never an extrapolated one. A coupon that has
**already fixed** (`fixing_date ≤ t`) bypasses the curve entirely and uses the index-history print,
exactly as `_leg_pv` does today; the anchor only matters for the first *unfixed* period.

`index.spot_lag` and `Index.get_maturity` (both now on the index) provide exactly what this needs.

## 6. How `_leg_pv` and the bootstrap consume it

- **Valuation** (`Calculator._leg_pv`): projection comes from `market.projection_curves[leg.index]` via
  `get_accrual(start, end)`; discounting stays on the `(riskless/collateral, currency)` discount curves.
  `TermRateLeg` and `OvernightLeg` call the same projection primitive:
  - `TermRateLeg` — each *unfixed* coupon (`fixing_date > t`) → `get_accrual(coupon.start, coupon.end)`;
    already-fixed coupons still read the index-history print (unchanged).
  - `OvernightLeg` — `get_accrual(coupon.start, coupon.end)` over the whole compounding period; the curve
    integrates its step forwards / uses the pseudo-DF ratio (`O(#segments ∩ window)` or the `O(1)`
    cumulative-pseudo-DF shortcut), so **no per-day loop**. A currently-accruing coupon splits into
    realised history `[start, t]` + `get_accrual(t, end)`.
  - Discounting is untouched: collateral-adjusted DFs from the `(riskless/collateral, currency)` curves —
    the projection change is orthogonal to it.
- **Bootstrap** (`build_curves`, detailed in §8.1):
  1. Discount/OIS + FX curves as today.
  2. For each rate index, bootstrap its `ProjectionCurve` **given** the discount curve — each swap pins
     the one segment forward it extends the curve by; the short end is anchored from the spot fixing
     (§5). One residual (par / MTM = 0) per instrument keeps the solve square, reusing the existing `root`
     solver.
  3. Knots are one-per-maturity (§3); the scheduler already orders "discount before projection".
  4. Because forwards are localised to segments, a curve bump is a localised delta — risk buckets
     naturally by segment, which is usually what a desk wants.

## 7. Projection == discount when they coincide — one unknown, not two

The critical `build_curves` case: for an **overnight, self-discounted** index (OIS — e.g. SOFR swaps
collateralised in SOFR) the projection curve `projection_curves[SOFR]` and the discount curve
`curves[(SOFR, USD)]` are the **same curve** — SOFR both projects and discounts. There is no
forwarding-vs-discounting basis, so they must be **one unknown**. If the builder treats them as two
independent curves, each maturity yields two pillars for one quote, the group goes under-determined, and
`InsufficientQuotesError` (builder.py:329) fires constantly.

Rule:

- **Overnight index used as its own collateral/riskless (OIS self-discounting):** `projection_curves[I]`
  **is** the `(I, I.currency)` discount curve — a thin forward *view* over the same DFs
  (`get_accrual(s, e) = df(s)/df(e) − 1`), **not** a second set of pillars. Build once.
- **Term index, or an index discounted by a *different* collateral/currency (a real basis):**
  `projection_curves[I]` is an **independent** unknown — the forwarding curve differs from every discount
  curve — bootstrapped from the index swaps *given* the (already-built) discount curve.

Equivalently: a separate projection unknown exists **iff there is a forwarding/discounting basis** for
that index. Overnight-self-discounted → no basis → alias the discount curve. This is the same distinction
the collateral-cancellation logic in `_curves_needed` already encodes (it is exactly why today's single
`(SOFR,USD)` curve is one unknown, not two) — the projection split must reuse that logic, not fight it.

**Detection & worked cases.** Reuse `_curves_needed`: an index aliases the discount curve exactly when
its instruments discount on that same index in its own currency (the cancellation already collapses
projection and discount to one key). Cases: (a) SOFR OIS collateralised in SOFR → aliased, one curve;
(b) TermSOFR-3M swaps collateralised in SOFR → the SOFR discount curve is already built and the
TermSOFR-3M *projection* is a genuine second unknown (a real basis) built on top of it; (c) a USD-SOFR
leg collateralised in CLP → discount `(ICP, USD)` ≠ projection SOFR → independent. So the invariant the
squareness check must enforce is: **total unknown curves = discount curves + projection curves that carry
a real basis.** `market.get_projection(index)` returns the wrapped discount curve when aliased and a
standalone `ProjectionCurve` otherwise, so callers never branch. (ICP today is an overnight rate
self-discounted in CLP, so its projection aliases `(ICP, CLP)` — no separate projection unknown yet.)

## 8. Migration

Design-only until picked up. This section names the exact existing symbols and target changes so an
implementation (likely with cleared context) needs no re-discovery. Phases 1–4 land incrementally; the
current single-curve behaviour is preserved until an index is given a real `ProjectionCurve`.

**Existing symbols to build on (do not re-derive):**

- `src/fintoolsom/rates/ZeroCouponCurve.py` — `class InterpolationMethod(Enum) { LogLinear=1,
  HermiteCubicSpline=2 }`; `ZeroCouponCurve(curve_date, date_dfs=[(date,df), …],
  df_interpolation_method=HermiteCubicSpline)`; `.get_dfs(dates)->np.ndarray`, `.get_df_fwd(s,e)=df(e)/df(s)`,
  `.get_wf_fwd/.get_wfs_fwds` (wealth factors), `.date_dfs`. Flat-extrapolates outside the pillar range.
- `src/fintoolsom/market/index.py` — `Index.get_maturity(start)->date`, `Index.spot_lag` (0 on
  `RateIndex`/overnight, 2 on `TermRateIndex`), `Index.calendar`; markers `OvernightIndex`,
  `OvernightRateIndex`, `TermRateIndex`.
- `src/fintoolsom/market/index_history.py` — `OvernightRateHistory.rates: dict[date,Rate]` + `.calendar`;
  `TermRateHistory.rates` + `.get_accrued_interest(notional, start, end, fixing_date=None)`.
- `src/fintoolsom/market/market.py` — `Market.curves: dict[(Index,Currency), ZeroCouponCurve]`;
  `.get_curve(index,currency)`, `.get_discount_dfs(riskless,currency,dates)`,
  `.get_projection_dfs(index,dates) = curves[(index,index.currency)].get_dfs(dates)`, `.get_index(name)`,
  `.get_rate(t,name,use_closest_past_rate=…)`, `.get_fx_rate(t,cp)`.
- `src/fintoolsom/derivatives/calculator.py` — `Calculator._leg_pv(leg, collateral, market,
  riskless_index)`: sets `proj_curve = market.get_curve(leg.index, leg.index.currency)` then projects with
  `proj_curve.get_wfs_fwds(starts, ends)` (TermRateLeg `to_proj` + OvernightLeg `pure_future`) and
  `proj_curve.get_wf_fwd(t, end)` (currently-accruing OIS coupon). `get_swap_mtm(swap, market,
  riskless_index, currency)` = `_pv(receive) − _pv(pay)` with FX conversion.
- `src/fintoolsom/curve_builder/builder.py` — `build_curves(quotes, riskless_index, market)`,
  `_curves_needed` (pinned, collateral-cancellation), `_curves_touched` (must-exist), `_get_maturity`,
  `InsufficientQuotesError`, and the fixpoint scheduler (`_attempt(curve_group) -> 'solved'|'nothing'|
  'defer'`, per-curve `fixed_pillars`/`free_dates`, log-df `root(f, y0, method="hybr")`, atomic rollback).

**Phase 1 — the `ProjectionCurve` type.** New `src/fintoolsom/rates/ProjectionCurve.py`:
`ProjectionCurve(curve_date, knot_dates: list[date], forward_rates: np.ndarray, interpolation_method,
day_count)`. State is **forward rates per segment** (not pseudo-DFs). Public seam: `get_accrual(s,e)`,
`get_equivalent_forward_rate(s,e,convention)`, `pseudo_df(u)` (derived; cumulative segment wealth factors).
Implement piecewise-constant only — `get_accrual` integrates the step forwards over `[s,e]`. (Numerically
equal to a `LogLinear` pseudo-DF `ZeroCouponCurve`, but stored as forwards; §3.)

**Phase 2 — wire valuation.** `Market`: add `projection_curves: dict[Index, ProjectionCurve]`,
`get_projection(index, *, interpolation_method=None)` / `set_projection`. In `Calculator._leg_pv` replace
the `proj_curve = market.get_curve(leg.index, leg.index.currency)` + `.get_wfs_fwds(...)` calls with
`market.get_projection(leg.index).get_accrual(start, end)` per coupon. For an **aliased** (OIS) index
`get_projection` returns a forward view over the `(index, index.currency)` discount `ZeroCouponCurve`
(§7). Discounting / collateral adjustment untouched.

**Phase 3 — config.** Add `Market.discount_interpolation_method = LogLinear`,
`Market.projection_interpolation_method = PiecewiseConstant`, and the optional `interpolation_method`
kwarg on `build_curves` and the get/set accessors (§4 Configuration).

**Phase 4 — build projection curves** (§8.1).

**Phase 5 (optional) — forward-space methods** (piecewise-linear / cubic / monotone-convex) behind the
unchanged `get_accrual` seam. Only these change the integrator and the bootstrap Jacobian (local →
global).

### 8.1 Migration: Build curve

`build_curves` today bootstraps every `(Index,Currency)` **discount** curve as a `ZeroCouponCurve` of DFs
through the fixpoint scheduler. Extend it to also produce projection curves, reusing the scheduler and
the `_curves_needed` / `_curves_touched` machinery:

1. **Alias vs independent (per rate index `I`).** Using `_curves_needed`, decide whether `I` aliases its
   discount curve (OIS self-discounting → no separate unknown: register `projection_curves[I]` as a
   forward view over `curves[(I, I.currency)]`) or is an **independent basis** curve needing a real
   solve (§7). Aliased indices need no bootstrap.
2. **Knots & anchor (independent case).** One knot per instrument **maturity** (§3 Knots); the unknown
   per knot is the **segment forward rate**. Add the fixed spot anchor segment
   `[t + I.spot_lag, I.get_maturity(t + I.spot_lag)]` seeded from the index-history spot fixing
   (`OvernightRateHistory` / `TermRateHistory`) — anchor forward is fixed, not solved (§5).
3. **Residual & solve.** Solve the **forward rates** (not pseudo-DFs) so each swap reprices —
   `Calculator.get_swap_mtm(swap, market, riskless_index, reporting_ccy) == 0` — given the
   already-built discount curve; one residual per instrument → square. Reuse the existing `root` solver;
   seed forwards from a flat guess or the discount-curve forwards. Piecewise-constant / -linear solve
   sequentially (triangular); cubic/spline solve globally (§3 locality caveat).
4. **Scheduling.** A projection curve for `I` is *touched-but-not-pinned* by its discount curve, so the
   existing fixpoint order ("build a group once every curve it touches-but-doesn't-pin exists") already
   defers it until `curves[(I, I.currency)]` is present. Keep `InsufficientQuotesError` guarding
   under-determination (it now also catches "two curves for one quote" if the alias rule is skipped).
5. **Store.** `market.set_projection(I, ProjectionCurve(...))` for independent curves; register the alias
   view for OIS indices.

**Backward-compat.** Until an index actually has swaps that build a `ProjectionCurve`, `_leg_pv`'s
projection should fall back to the `(index, index.currency)` discount curve (today's behaviour), so the
migration is incremental and existing tests keep passing.
