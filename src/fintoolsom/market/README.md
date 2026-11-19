# fintoolsom.market

Market data and its definitions: currencies, indexes and their fixings, FX, the UF,
curves, volatility surfaces and instrument quotes. `Market` is the one object that
pricing (`derivatives`) and curve building (`curve_builder`) read from.

## Modules

| Module | Contents |
|---|---|
| `currencies.py` | `Currency` (`Currency.USD`, `Currency.CLF`, ...), `CurrencyPair` (with its `spot_lag`), `FX_Rate`, `FX_RateData` (FX history of a pair), `FX_SPOT_LAGS` |
| `localities.py` | `Locality`: the country whose calendar a currency uses |
| `index.py` | Index definitions: `OvernightRateIndex` (SOFR), `TermRateIndex` (Term SOFR, TAB), `OvernightInterestPriceIndex` (ICP) |
| `index_history.py` | Fixings: `OvernightRateHistory`, `TermRateHistory`, `OvernightInterestPriceHistory` (levels, gaps filled geometrically) |
| `market.py` | `Market`: date `t`, FX (history, projected fixings `get_future_fx_rate`, currency conversion), index histories, `risk_free_index`, curves, volatility surfaces |
| `quotes.py` | Quotes that build their instrument: FX forwards (price or points), IRS/OIS, basis, cross-currency swaps |
| `conventions.py` | `FixedLegSpec`, `FloatingLegSpec`, `PaymentFrequency`: leg structure used by quotes |
| `uf.py` | UF (`CLF`) mechanics: publication calendar, CPI reajuste, `uf_rate_data(levels)` |
| `volatility_surface.py` | FX volatility surface: two-step interpolation (linear, monotone cubic, vanna-volga, SVI) or eSSVI |

## Indexes and histories

An index is a definition: name, currency, calendar and spot lag. Two indexes are
equal when their names are. Fixings live in a history, stored in
`Market.indexes_history` under the index name.

- Rate indexes publish `Rate`s. `spot_overnight_rate(t)` is today's fixing.
- Price indexes (ICP) publish levels, and accrual is the ratio of two levels.
  `rate_convention` (linear Act/360 by default) is the convention of any rate
  implied from two levels.

## Curves in the market

`Market.curves` maps a key to a curve (see `fintoolsom.rates`); the key gives the
curve its role. An `Index` key projects that index, and the risk-free index's curve
also discounts its currency. A `Currency` key discounts that currency under the
market's `risk_free_index`.

```python
market = Market(t, risk_free_index=icp, curves={icp: icp_curve, sofr: sofr_curve})
market.set_curve(Currency.USD, usd_curve)    # USD discounting under ICP
market.get_index_curve(sofr)                 # curves[sofr]
market.get_discount_curve(Currency.USD)      # curves[Currency.USD]
market.get_discount_curve(Currency.CLP)      # curves[icp]: the risk-free index's own currency
market.get_accrual(sofr, start, end)         # SOFR projected over [start, end]: wf - 1
market.get_dfs(Currency.CLP, dates, collateral=sofr)  # CLP_ICP * SOFR / USD_ICP
```

- Without collateral, `get_dfs` reads one discount curve. With collateral in the
  flows' currency, it reads the collateral index curve. With collateral in another
  currency, it reads three curves.
- A market has one numeraire, its `risk_free_index`. The risk-free index's currency
  is not a key: it discounts on the index's curve, so `set_curve(Currency.CLP, ...)`
  under ICP raises a `ValueError`.
- `set_curve` rejects values that are not a `Curve` and keys that are neither an
  `Index` nor a `Currency`.

## FX spot lag

A pair's `spot_lag` is its market convention: business days from the trade date to
the value date of a spot deal, on the joint calendar of both currencies.
`pair.get_spot_date(trade_date)` applies it.

| Pair | `spot_lag` |
| --- | --- |
| USD/CLP, USD/CAD | 1 |
| EUR/USD, GBP/USD, USD/JPY | 2 |
| any other pair (`DEFAULT_FX_SPOT_LAG`) | 2 |

The lag is looked up in `FX_SPOT_LAGS`, keyed by the two currencies in either order,
so `CLP/USD` is T+1 too; add or change an entry to set another convention. It is not
a dataclass field, so pairs stay equal and hash by their name. FX forward quotes
(`ForwardPriceQuote`, `ForwardPointsQuote`) default `spot_lag` to the pair's.
Valuation still reads the FX of date `t` as spot; the lag sets the dates quotes
build, not how FX is projected.

## UF

The UF is the currency `CLF`, quoted against CLP as FX. Levels published ahead of
time (up to the 9th of next month once CPI is out) are read as they are.
`Market.get_fx_known_until` reports how far they go, and later dates are projected
from the CLF and CLP curves.

## Quotes

Every quote builds its instrument with `get_instrument()`. FX-dependent quotes
(FX forwards, cross-currency swaps) take the market:
`get_instrument(market)`. Swap quotes carry `collateral_index`, the CSA under which
they are valued.
