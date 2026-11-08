# Curve redesign plan

Status: implemented on `feature/derivatives` (uncommitted). Deviations from the proposal: builder kwargs are
`discount_interp_method` / `index_interp_method`; `NSSCurve` and `NelsonSiegelSvensson.get_curve` were removed;
the legacy warning is a `DeprecationWarning`. Open: term-index discounting, synthetic `CLP_SOFR` curves.

## 1. Goal

One curve model for pricing and building:

- `Curve` (abstract), with two concrete roles:
  - `IndexCurve`: the curve of an interest index, stored as forward rates.
  - `DiscountCurve`: the discount curve of a currency that has no index of its own under the market's risk-free index (e.g. USD in Chile under ICP), stored as dfs.
- `Market` holds a single `dict[str, Curve]` keyed by `curve.get_id()`.
- `derivatives`, `curve_builder` and `market` stop using `ZeroCouponCurve`, `ProjectionCurve` and `DiscountProjectionView`.
- `ZeroCouponCurve` stays in `rates` for legacy use only (`CLBonds`, `models/NelsonSiegelSvensson`). It emits a warning when it is created.

Out of scope for now:

- the final name of `DiscountCurve`;
- synthetic cross-numeraire curves (e.g. CLP_SOFR built from CLP_ICP × SOFR / USD_ICP);
- options / volatility.

## 2. Classes (`rates`)

### 2.1 `Curve` (ABC)

Required by every curve:

| Member | Meaning |
|---|---|
| `curve_date: date` | t; df(t) = 1 |
| `interpolation_method` | Required. Each subclass validates it against its own `allowed_methods` |
| `get_id() -> str` | Key in `Market.curves` |
| `get_dfs(dates) -> ndarray`, `get_df(date)` | Discount factors from `curve_date` |
| `last_date` | Last pillar or knot; the builder uses it to decide whether a pillar is open |
| `parallel_bump_bps(bps) -> Self` | Returns a bumped copy: +bps on every continuously compounded zero rate. Copy rather than in place, to avoid +1/−1 round-off |
| `copy()` | |

Implemented once in the base class from `get_dfs`:

- `get_dfs_fwd(start, end)` and `get_wf_fwd(start, end)`;
- zero rates and forward rates expressed in any `RateConvention`.

Extrapolation is the same for both roles: **flat instantaneous forward past `last_date`**. This is a change from `ZeroCouponCurve`, which uses a flat zero rate; see §8.

### 2.2 `IndexCurve(Curve)`

- Fields:
  - `index: Index`
  - `effective_date: date`: the first knot. For an overnight index this is the next business day (today's `short_end_anchor`); for a term index it is `t + spot_lag`.
  - `knot_dates` (starting at `effective_date`) and `forward_rates: ndarray`. These are continuously compounded instantaneous forwards; the meaning follows the interpolation method.
  - `spot_rate: Rate | None`: covers `[curve_date, effective_date]` in its own convention.
- `get_id() = index.name`
- dfs:
  - df(d) = df_spot × exp(−∫_{effective}^{d} f), with df_spot = `spot_rate.get_discount_factor(t, effective_date)`.
  - For d inside the spot segment, the spot rate applies flat.
  - If `spot_rate is None`, the first forward is extended flat back to t, so df(t) = 1 always holds. A term index has no natural rate over the spot lag.
- Projection interface:
  - `get_wealth_factor(s, e)` and `get_accrual(s, e)`, computed as df(s)/df(e) (− 1 for the accrual);
  - `get_forward_rate(s, e, rate_convention)`.
- `allowed_methods`: forward-space methods only. Today that is `PiecewiseConstant`; later linear forward and monotone convex.
- No discount restriction: any `IndexCurve` can discount, including a term index. That is the user's call.

### 2.3 `DiscountCurve(Curve)`

- Fields: `currency`, `risk_free_index: Index`, `pillar_dates`, `dfs: ndarray`. Built from dfs only.
- `get_id() = f"{currency.name}_{risk_free_index.name}"`, e.g. `USD_ICP`.
- `allowed_methods`: df-space methods, `LogLinear` and `HermiteCubicSpline`.
- Invalid if `currency == risk_free_index.currency`, because that curve is the index's own `IndexCurve`. Both the constructor and `Market.set_curve` raise.

### 2.4 Interpolation enum

One `InterpolationMethod` enum covering both spaces (`LogLinear`, `HermiteCubicSpline`, `PiecewiseConstant`, …). Each class whitelists its members. `ProjectionInterpolationMethod` and the old enum in `ZeroCouponCurve` become legacy aliases.

### 2.5 Legacy

- `ZeroCouponCurve`, `ProjectionCurve` and `DiscountProjectionView` stay importable.
  - `ZeroCouponCurve.__post_init__` emits a warning (category to be decided, §9).
  - `ProjectionCurve` and `DiscountProjectionView` are removed once nothing uses them, or deprecated the same way.
- The old `rates.Curve` ABC (base of `ZeroCouponCurve` and `NSSCurve`) is renamed `LegacyCurve`. `Curve` is the new ABC.
- `CLBonds`, `models/NelsonSiegelSvensson.py` and `NSSCurve` keep `ZeroCouponCurve` / `LegacyCurve` unchanged.
- Converters for migration: `DiscountCurve.from_zero_coupon_curve(zcc, currency, risk_free_index)` and `IndexCurve.from_zero_coupon_curve(zcc, index, ...)`.

## 3. Market

- `curves: dict[str, Curve]` is the only curve store. `projection_curves`, `CurveKey` and `funding_key` are removed.
- Rename `riskless_index` to `risk_free_index`, matching the new API (§9).
- `set_curve(curve)`:
  - stores `curves[curve.get_id()] = curve`;
  - rejects a `ZeroCouponCurve` / `LegacyCurve` with an error that names the converter;
  - rejects a `DiscountCurve` whose currency is its risk-free index's currency;
  - an id clash between two `Index` objects with the same name is accepted as the user's responsibility.
- `get_index_curve(index) -> IndexCurve`: returns `curves[index.name]`, and raises a `KeyError` that names the index.
- `get_discount_curve(currency, *, risk_free_index=None) -> Curve`:
  - `rfi = risk_free_index or self.risk_free_index`;
  - if `currency == rfi.currency`, return `get_index_curve(rfi)`;
  - otherwise return `curves[f"{currency.name}_{rfi.name}"]`. A `KeyError` is expected when nothing was built under that numeraire (e.g. CLP under SOFR in an ICP market).
- `get_dfs(currency, dates, *, collateral=None, risk_free_index=None) -> ndarray`, plus `get_df` for a single date:
  - `collateral is None`: `get_discount_curve(currency, risk_free_index=...)`, one curve;
  - `collateral.currency == currency`: `get_index_curve(collateral)`, one curve. This is today's step-1 shortcut. Keep it as an explicit branch so the builder never needs the funding curve, and to avoid round-off;
  - otherwise: `disc(currency) × idx(collateral) / disc(collateral.currency)`, three curves, computed live so bumps carry through.
- Market holds no interpolation settings.
- Removed: `get_curve`, `get_funding_curve`, `get_discount_dfs`, `get_discount_df`, `get_projection`, `set_projection` and `funding_key`. Alternatively they stay for one release as thin deprecated wrappers (§9).

## 4. Curve builder

- **Keys.**
  - Internal keys become the curve kinds. An `Index` key means an `IndexCurve`. A `Currency` key means a `DiscountCurve` under `market.risk_free_index`.
  - `_Proj` disappears. Whether an index "discounts" stays a planning concept, inside `_curves_needed` (which curves a quote reads); it is no longer a storage concept.
  - Ids are derived only when reading from or writing to `Market`.
- **Unknowns.**
  - `IndexCurve`: the forwards of its new segments, as today for projection curves. SOFR and ICP move from log-df `ZeroCouponCurve`s to forward `IndexCurve`s.
  - `DiscountCurve`: log(df) at its new pillars, as today.
  - `_GroupSolve.apply` builds the right class per key, and the `_Proj` branches go.
- **Anchors.**
  - `short_end_anchor` becomes the `IndexCurve` `effective_date` and `spot_rate` (overnight fixing).
  - `projection_anchor` becomes the term index's first knot segment, with its forward from today's fixing and `spot_rate = None`.
  - Both are inputs, never unknowns.
- **Methods.**
  - `build_curves(interpolation_method=LogLinear, projection_method=PiecewiseConstant)` become the defaults for new `DiscountCurve`s and new `IndexCurve`s respectively. Consider renaming them `discount_method` / `index_method` (§9).
  - An extended curve keeps its own method, as now.
  - `local` checks each curve's method against the local set.
- **Planning.**
  - `_Planner.is_open` uses `curve.last_date`.
  - `partition_pillars` reads `(dates, dfs)` from a `DiscountCurve` and `(knots, forwards)` from an `IndexCurve`, via a small accessor on each class, not via `isinstance` on legacy types.
- **Seeds.** Unchanged in substance. Seeds in df space for an `IndexCurve` are converted to segment forwards.
- **Behaviour change.** `interpolation_method=HermiteCubicSpline` no longer affects SOFR/ICP, because `IndexCurve` only accepts forward-space methods. It applies to `DiscountCurve`s only. `test_unified_curve_build.py:170` must be adapted.

## 5. Derivatives (`derivatives/calculator.py`)

| Today | After |
|---|---|
| `market.get_funding_curve(ccy)` (lines 72-73, 119, 584-585, 798-799) | `market.get_discount_curve(ccy)` |
| `market.get_discount_dfs(ccy, dates, collateral)` (lines 531, 623, 746) | `market.get_dfs(ccy, dates, collateral=collateral)` |
| `market.get_projection(leg.index)` (line 440) | `market.get_index_curve(leg.index)`: `get_accrual` is unchanged |
| `options.py:19` comment on `market.curves` | Update the comment only |

Fixings and accrual logic do not change. The current period uses fixings up to t, then the `IndexCurve`, whose spot segment covers t to `effective_date`.

## 6. Tests

Today:

- 17 test files reference `ZeroCouponCurve`, `ProjectionCurve` or `DiscountProjectionView`;
- 21 `ZeroCouponCurve(...)` constructions;
- 29 `market.curves` / `projection_curves` references.

Plan:

1. **New unit tests for `IndexCurve` and `DiscountCurve`:**
   - ids and validation;
   - allowed methods;
   - spot-rate segment;
   - flat-forward extrapolation;
   - bump returns a copy, and +1bp on forwards equals +1bp on cc zeros.
2. **Parity tests** on the same pillars, to 1e-12 inside the curve:
   - LogLinear `DiscountCurve` ≡ LogLinear `ZeroCouponCurve`;
   - PiecewiseConstant `IndexCurve` ≡ LogLinear `ZeroCouponCurve` beyond `effective_date`;
   - `IndexCurve` ≡ `ProjectionCurve` on the same knots.
3. **Market tests:**
   - the three `get_dfs` branches;
   - `get_discount_curve` default and explicit numeraire, and the `KeyError` case;
   - `set_curve` rejections.
4. **Migrate the factories first** (`tests/factories/curve_factory.py`, `quote_factory.py`), then the builder and derivative tests.
5. **Legacy tests** (`CLBonds`, NSS) keep using `ZeroCouponCurve`, with the warning filtered in `pyproject` / `conftest`.
6. **End-to-end:**
   - the notebook and the B, C and partial integration cases still reprice to 1e-6 CLP;
   - timings are recorded before and after; the new classes hold no `Rate` objects, so the solver's `apply()` should get faster.

## 7. Phases

Each phase ends with the whole suite passing.

1. **`rates`.**
   - New `Curve` / `IndexCurve` / `DiscountCurve` and the merged enum.
   - Rename the old ABC to `LegacyCurve`.
   - Parity tests.
   - Nothing else changes.
2. **`Market`.**
   - Single dict, `set_curve`, the new getters, and `get_dfs`.
   - The old getters become temporary wrappers so that the builder and derivatives keep working.
3. **Builder.**
   - Keys, `_GroupSolve`, planner, anchors.
   - Writes go through `set_curve`; restore/rollback works on ids.
4. **Derivatives.** Switch to the new getters (§5).
5. **Clean-up.**
   - Remove the wrappers, `_Proj`, `funding_key` and `CurveKey`.
   - `ZeroCouponCurve` warns on creation.
   - Update the notebook, README and CHANGELOG.

## 8. Risks

- **Extrapolation change.** Flat forward instead of flat zero changes values past the last pillar. Tests with hard-coded long-end numbers will move.
- **Spot segment.** Moving SOFR/ICP to `IndexCurve` changes how t to `effective_date` is represented: the spot rate in its own convention, instead of a log-linear df pillar. Expect tiny differences before the first knot; the parity tests start at `effective_date`.
- **Warning noise.** `ZeroCouponCurve.copy()` and `get_aged_curve()` create new curves internally. Warn only at the user-facing constructor call, e.g. with a `_legacy_internal` flag or a once-per-process filter.
- **Same-named indexes.** A silent id clash is accepted (user responsibility). Document it in `set_curve`.

## 9. Decisions needed before coding

1. **Warning category for `ZeroCouponCurve`.**
   - `DeprecationWarning` is hidden by default outside `__main__` and test runs.
   - `FutureWarning` is always shown.
2. **Renames.** `riskless_index` → `risk_free_index` on `Market` (proposed: yes). Builder kwargs → `discount_method` / `index_method` (proposed: yes, while the API is being broken anyway).
3. **Old `Market` getters.** Remove in phase 5, or keep as deprecated wrappers for one release.
4. **`NSSCurve`.** Stays `LegacyCurve` (proposed), or becomes a `DiscountCurve` backend later.
5. **`ProjectionCurve` / `DiscountProjectionView`.** Delete in phase 5 (proposed), or keep as deprecated.
