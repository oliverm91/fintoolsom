- Correctly model sofr and icp as overnight indexes.
- Market overdefines indexes and fx history.
- Market call index history by name, should be enum or class.
- Market should have CurveKey or similar.
- Add UF products
- Bring back projection curves? Interpolate projection rate which has a term tied to its index. Thought for TermRateIndexes... what happens if need to project 3M rate starting in 1 W?
  - Maybe not, but ability to add a short term rate to a term curve. Example: SOFR O/N as short rate for building TERMSOFR 3M curve.
- FX spot lag in valuation (V4). The quoted spot `S` of a pair is the price for exchange on its spot date `s = pair.get_spot_date(t)` (USD/CLP T+2 offshore, T+1 by the current `FX_SPOT_LAGS` table), not on `t`. Today `Market` treats the FX of `t` as value `t`.
  - Current: CLP PV of a USD flow `N` on date `d` is `N * df_USD(d) * S`.
  - Correct, for any `d`: `PV_CLP = N * df_USD(d) / df_USD(s) * S * df_CLP(s)`. Move the USD flow from `d` to `s` on the USD curve, convert at `S` on `s`, discount the CLP from `s` to today. Equivalently convert at the forward anchored on spot, `F(d) = S * [df_USD(d) / df_CLP(d)] / [df_USD(s) / df_CLP(s)]`, and discount in CLP: `N * F(d) * df_CLP(d)`.
  - Flow after spot (e.g. `d = t+5`): `df_USD(d) / df_USD(s) < 1`, the USD is discounted back to spot.
  - Flow before spot (e.g. `d = t+1`): `df_USD(d) / df_USD(s) > 1`, the USD grows forward to spot; this is the tom rate (spot minus one day of points). With `d = t` it is the cash rate.
  - Current over correct is `df_USD(s) / df_CLP(s) ~ 1 + (r_CLP - r_USD) * lag / 365`, the same for every `d`. Example: N = 1M USD, S = 950, r_CLP 6%, r_USD 4.9% (continuous Act/365), T+2: correct 949,305,300 CLP at t+5 and 949,815,200 at t+1; current 949,362,600 and 949,872,500; about -57,200 CLP (0.6bp of notional) both. Forward-points calibration absorbs most of it into the USD curve level.
  - Also shows as one lag of carry on the notional of a spot-start deliverable swap with `initial_exchange=True`, which should be worth about 0 at inception.
  - To do: anchor `Market.convert_value_currency` and the future-FX projection (`get_future_fx_rate`, `_future_fx_rates_at_ordinals`) on `pair.get_spot_date(t)` with the formula above; check forwards, NDFs, options and `ValuationBatch` (values to the bit, gradient gains the `df(s)` terms); rebuild the curve-builder tests (USD curves calibrated from forward points will move slightly).
  - Curve impact, measured on the test quote set (`tests/factories/quote_factory.py`, `notebooks/quotes.json`: t = 2026-06-29, S = 900, short ICP 5.64%, short USD-ICP 5.33%, first USD forward pillar 2026-07-08). Method: the correct model is the current one with the market FX at t set to the value-today rate `X = S * df_CLP(s) / df_USD(s)`, quotes still trading at S (outrights S + points, xccy notionals sized at S); X solved by fixed point over full builds (7 iterations).
    - X / S - 1 = ln k: -0.097bp at T+1 (s = 2026-06-30), -0.222bp at T+2 (s = 2026-07-01).
    - ICP (CLP) and SOFR curves: unchanged (0 to the printed digits, every df and pillar forward).
    - USD discount curve (ICP collateral), df % difference (correct / current - 1):

      | lag | 1D | 1W | 1M | 3M | 6M | 1Y | 2Y | 5Y | 10Y | 20Y |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | T+1 | 0.00011 | 0.00076 | 0.00097 | 0.00097 | 0.00097 | 0.00097 | 0.00097 | 0.00097 | 0.00097 | 0.00097 |
      | T+2 | 0.00025 | 0.00173 | 0.00222 | 0.00222 | 0.00222 | 0.00222 | 0.00222 | 0.00222 | 0.00222 | 0.00222 |

    - USD discount curve forward differences, continuous Act/365, bp:

      | lag | 0-1D | 1D-1W | 1W-1M | 1M-3M | later |
      | --- | --- | --- | --- | --- | --- |
      | T+1 | -3.94 | -3.94 | -0.34 | 0 | 0 |
      | T+2 | -9.01 | -9.01 | -0.78 | 0 | 0 |

    - Solved pillar forwards: same knots; only the first USD segment (2026-06-29 to 2026-07-08) moves, by -3.943bp (T+1) and -9.012bp (T+2); every later pillar forward differs by less than 0.0001bp. The whole effect is `ln k` put into the first segment: shift = ln k / (first segment in years), 0.097bp * 365 / 9 = 3.9bp.
    - So past the first USD pillar every USD df moves by the same factor k (0.001% at T+1, 0.002% at T+2) and CLP values of USD flows are unchanged; flows before it, USD-reported values and USD forward rates in the first segment are what change.
    - Scaling: the df level is `(r_CLP - r_USD) * lag / 365` and the first-segment shift that divided by the segment length. With a 1.1% differential: 0.30bp / 0.60bp of df and about 12bp / 24bp on a 9-day first segment (T+1 / T+2).
  - CLF/CLP is not spot traded (the UF of t is the value of t): its spot lag should be 0 when this is done; `FX_SPOT_LAGS` gives it the default 2 today (unused in valuation).
