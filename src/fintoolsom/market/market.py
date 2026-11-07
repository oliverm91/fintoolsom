from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from .currencies import Currency, CurrencyPair, FX_Rate, FX_RateData
from .index import Index
from .index_history import IndexHistory
from ..rates import (
    Rate,
    Curve,
    ProjectionCurve,
    DiscountProjectionView,
    InterpolationMethod,
    ProjectionInterpolationMethod,
)
from ..dates import Calendar

if TYPE_CHECKING:
    # Deferred: volatility_surface imports derivatives.calculator, which imports
    # Swap (derivatives.swaps -> market.currencies), so a runtime import here
    # would cycle back into this module before Market is defined.
    from .volatility_surface import VolatilitySurface


# A discount curve is identified by an Index (that index's own curve) or by a
# Currency (that currency's uncollateralised, cross-currency-implied curve).
CurveKey = Index | Currency


def funding_key(riskless_index: Index, currency: Currency) -> CurveKey:
    """Key of `currency`'s uncollateralised discount curve: the riskless index itself
    for its own currency, otherwise the currency (a cross-currency-implied curve)."""
    if currency == riskless_index.currency:
        return riskless_index
    return currency


@dataclass(slots=True)
class Market:
    t: date
    fx_history: dict[CurrencyPair, FX_RateData] = field(default_factory=dict)
    indexes_history: dict[str, IndexHistory] = field(default_factory=dict)
    interest_rates: dict[str, dict[date, Rate]] = field(default_factory=dict)
    currency_pairs_history: dict[str, dict[date, CurrencyPair]] = field(
        default_factory=dict
    )
    index_to_interest_rate_map: dict[str, str] = field(default_factory=dict)
    # UF is known until the 9th of the next month, then it must be projected with curves.
    uf_history: dict[date, float] = field(default_factory=dict)

    interest_rate_to_index_map: dict[str, str] = field(init=False, default_factory=dict)

    # The market's funding index (Chile: ICP). Uncollateralised flows in its own
    # currency discount on its curve; it says nothing about any other currency.
    riskless_index: Index | None = None

    # Discount-factor curves, each stored once under a CurveKey:
    #   Index    → that index's own curve (e.g. ICP, SOFR, Cámara). Used to discount
    #              (as funding or collateral) and, via get_projection, to project it.
    #   Currency → the uncollateralised curve of a currency other than the riskless
    #              index's, implied from cross-currency products (USD/UF/EUR in Chile).
    # Collateralised curves are not stored: get_discount_dfs derives them.
    curves: dict[CurveKey, Curve] = field(default_factory=dict)
    # Independent-basis forward curves (e.g. a term index such as TAB 3M), keyed by
    # Index. An index with a discount curve in `curves` needs no entry here:
    # get_projection falls back to a forward view over that curve.
    projection_curves: dict[Index, ProjectionCurve] = field(default_factory=dict)
    volatility_surfaces: dict[CurrencyPair, VolatilitySurface] = field(default_factory=dict)

    # Default interpolation per curve role, used when build_curves / the accessors are
    # not given an explicit override. Discount curves interpolate log-linearly in DF
    # (consistent with the log-df bootstrap); projection curves are piecewise-constant
    # in forward. The method is baked at bootstrap — it is the curve's calibration
    # identity — so changing it means re-solving that curve, not resampling in place.
    discount_interpolation_method: InterpolationMethod = InterpolationMethod.LogLinear
    projection_interpolation_method: ProjectionInterpolationMethod = (
        ProjectionInterpolationMethod.PiecewiseConstant
    )

    def __post_init__(self):
        if self.riskless_index is not None and self.riskless_index.currency is None:
            raise ValueError(
                f"riskless_index '{self.riskless_index.name}' needs a currency: it funds that currency."
            )
        fxs_to_add = {}
        for cp, fx_data in self.fx_history.items():
            # Check fx_history FX_RateData is correctly mapped to CurrencyPairs.
            if cp != fx_data.currency_pair:
                raise ValueError(
                    f"fx_history contained a key-value pair inconsistent in CurrencyPair. Key: {cp}, Value CurrencyPair: {fx_data.currency_pair}"
                )
            # Add inverted data
            if cp.invert() not in self.fx_history:
                fxs_to_add[cp.invert()] = fx_data.invert()
        for cp, fx_data in fxs_to_add.items():
            self.fx_history[cp] = fx_data

        for k in list(self.interest_rates.keys()):
            v = self.interest_rates.pop(k)
            self.interest_rates[k.upper()] = v

        for k in list(self.indexes_history.keys()):
            v = self.indexes_history.pop(k)
            self.indexes_history[k.upper()] = v

        self.interest_rate_to_index_map = {
            rate_name: index_name
            for index_name, rate_name in self.index_to_interest_rate_map.items()
        }

        # Invert currency pairs
        for cp_history_dict in list(self.currency_pairs_history.values()):
            for cp_date, cp in list(cp_history_dict.items()):
                inverted_cp = cp.invert()
                self.currency_pairs_history.setdefault(inverted_cp.name, {})[
                    cp_date
                ] = inverted_cp

    def get_curve(self, key: CurveKey) -> Curve:
        try:
            return self.curves[key]
        except KeyError:
            kind = "index" if isinstance(key, Index) else "currency"
            raise KeyError(f"No discount curve for {kind} '{key}' in market.") from None

    def funding_key(self, currency: Currency) -> CurveKey:
        """Key of `currency`'s uncollateralised curve. See :func:`funding_key`."""
        if self.riskless_index is None:
            raise ValueError("Market has no riskless_index; cannot resolve funding curves.")
        return funding_key(self.riskless_index, currency)

    def get_funding_curve(self, currency: Currency) -> Curve:
        """Uncollateralised discount curve of `currency`."""
        return self.get_curve(self.funding_key(currency))

    def get_discount_dfs(
        self, currency: Currency, dates: list[date], collateral: Index | None = None
    ) -> np.ndarray:
        """DFs for `currency` flows paid on `dates`, under `collateral` (None = uncollateralised):

            df = funding(ccy) · curve(collateral) / funding(collateral.ccy)

        When the collateral is the funding index of its own currency the ratio is 1, so
        e.g. CLP flows under ICP collateral in an ICP market discount on plain ICP."""
        dfs = self.get_funding_curve(currency).get_dfs(dates)
        if collateral is None:
            return dfs
        collateral_ccy = collateral.currency  # type: ignore[attr-defined]
        return (
            dfs
            * self.get_curve(collateral).get_dfs(dates)
            / self.get_funding_curve(collateral_ccy).get_dfs(dates)
        )

    def get_discount_df(self, currency: Currency, t: date, collateral: Index | None = None) -> float:
        return float(self.get_discount_dfs(currency, [t], collateral)[0])

    def get_projection(
        self, index: Index, *, interpolation_method=None
    ) -> ProjectionCurve | DiscountProjectionView:
        """Forward-projection curve for `index`, exposing the ``get_accrual`` seam.

        Returns the stored :class:`ProjectionCurve` when `index` has an independent
        forwarding basis. Otherwise it falls back to a :class:`DiscountProjectionView`
        over the index's own discount curve (``curves[index]``) — the same DFs, viewed
        as forwards — so an overnight index's curve is never stored twice.

        `interpolation_method` is accepted for forward-compatibility (a resampled
        what-if view); the stored curve is returned as-built for now."""
        if index in self.projection_curves:
            return self.projection_curves[index]
        if index not in self.curves:
            raise KeyError(f"No projection or discount curve for index '{index.name}' in market.")
        return DiscountProjectionView(self.curves[index])

    def set_projection(self, index: Index, curve: ProjectionCurve) -> None:
        """Store an independent-basis forward curve for a projection-only `index`.

        An index holds exactly one curve: if it already has a discount curve in
        ``curves``, that curve is its projection (via get_projection) and a second
        one would let the two drift apart."""
        if index in self.curves:
            raise ValueError(
                f"Index '{index.name}' already has a discount curve; it projects off that "
                "curve and cannot also hold an independent projection curve."
            )
        self.projection_curves[index] = curve

    def add_index(self, history: IndexHistory):
        self.indexes_history[history.name.upper()] = history

    def add_currency_pair(self, currency_pair: CurrencyPair):
        if currency_pair.name not in self.currency_pairs_history:
            self.currency_pairs_history[currency_pair.name] = {}
        self.currency_pairs_history[currency_pair.name][currency_pair.cp_date] = (
            currency_pair
        )
        inverted_pair = currency_pair.invert()
        self.currency_pairs_history[currency_pair.name][inverted_pair.cp_date] = (
            inverted_pair
        )

    def get_index(self, name: str) -> IndexHistory:
        name = name.upper()
        if name in self.indexes_history:
            return self.indexes_history[name]
        raise KeyError(f"Index {name} not found in market.")

    def get_rate(self, t: date, name: str, use_closest_past_rate: bool = False) -> Rate:
        if name in self.interest_rates:
            if t in self.interest_rates[name]:
                return self.interest_rates[name][t]
            if not use_closest_past_rate:
                raise ValueError(f"Rate {name} not found for date {t}.")
            else:
                past_dates = [
                    x
                    for x in self.interest_rates
                    if x < t and name in self.interest_rates[x]
                ]
                if len(past_dates) == 0:
                    raise ValueError(
                        f"Rate {name} not found for date {t} and there were no past dates."
                    )
                last_rate_date = max(past_dates)
                return self.interest_rates[name][last_rate_date]

        raise ValueError(f"Rate {name} not found.")

    def accrue_rates_reset_business_days(
        self,
        notional: float,
        rate_name: str,
        start_date: date,
        end_date: date,
        fixing_lag: int = 0,
        calendar: Calendar = None,
        use_closest_past_rate_for_fixing: bool = False,
    ) -> float:
        if calendar is None:
            calendar = Calendar()
        if fixing_lag < 0:
            raise ValueError(
                f"fixing_lag must be greater than or equal to 0. Got {fixing_lag}."
            )
        if start_date > end_date:
            raise ValueError(
                f"Start date must be earlier than end date. Start date: {start_date}, End date: {end_date}."
            )
        t = start_date
        acrrued_interest = 0
        while t < end_date:
            reset_date = calendar.add_business_days(t, -fixing_lag)
            rate = self.get_rate(
                reset_date,
                rate_name,
                use_closest_past_rate=use_closest_past_rate_for_fixing,
            )
            next_business_day = calendar.add_business_days(t, 1)
            acrrued_interest += rate.get_accrued_interest(
                notional + acrrued_interest, t, min(next_business_day, end_date)
            )
            t = calendar.add_business_days(t, 1)

        return acrrued_interest

    def accrue_rates_custom_reset_days(
        self,
        notional: float,
        rate_name: str,
        start_date: date,
        end_date: date,
        reset_dates: list[date],
        use_closest_past_rate_for_fixing: bool = False,
    ) -> float:
        reset_dates.sort()
        if reset_dates[0] > start_date:
            raise ValueError(
                f"First reset date must be after start date. Got Start date: {start_date} and min reset date: {reset_dates[0]}."
            )
        if start_date > end_date:
            raise ValueError(
                f"Start date must be earlier than end date. Start date: {start_date}, End date: {end_date}."
            )
        t = start_date
        reset_counter = 0
        acrrued_interest = 0
        while t < end_date:
            reset_date = reset_dates[reset_counter]
            rate = self.get_rate(
                reset_date,
                rate_name,
                use_closest_past_rate=use_closest_past_rate_for_fixing,
            )

            # If we have reached the end of the reset dates, we need to accrue till the end date. Case when custom dates are within accrual period
            if reset_counter == len(reset_dates) - 1:
                next_t = end_date
            else:
                next_t = reset_dates[reset_counter + 1]

            # If we have not reach the end of reset dates AND next_t is after end_date, we need to accrue till the end date. Case when custom dates are after accrual period
            if next_t > end_date:
                next_t = end_date  # As t becomes next_t, loop will end.
            acrrued_interest = rate.get_accrued_interest(
                notional + acrrued_interest, t, next_t
            )
            reset_counter += 1
            t = next_t
        return acrrued_interest

    def accrue_rates_single_reset_day(
        self,
        notional: float,
        rate_name: str,
        start_date: date,
        end_date: date,
        reset_date: date,
        use_closest_past_rate_for_fixing: bool = False,
    ) -> float:
        if start_date > end_date:
            raise ValueError(
                f"Start date must be earlier than end date. Start date: {start_date}, End date: {end_date}."
            )
        if reset_date > start_date:
            raise ValueError(
                f"Reset date must be earlier than start date. Reset date: {reset_date}, Start date: {start_date}."
            )
        rate = self.get_rate(
            reset_date,
            rate_name,
            use_closest_past_rate=use_closest_past_rate_for_fixing,
        )
        return rate.get_accrued_interest(notional, start_date, end_date)

    def get_volatility_surface(self, currency_pair: CurrencyPair) -> VolatilitySurface:
        if currency_pair not in self.volatility_surfaces:
            raise KeyError(f"No volatility surface found for currency pair {currency_pair}.")
        return self.volatility_surfaces[currency_pair]

    def add_fx_rate(self, t: date, fx_rate: FX_Rate):
        if fx_rate.currency_pair not in self.fx_history:
            self.fx_history[fx_rate.currency_pair] = FX_RateData(
                fx_rate.currency_pair, {t: fx_rate}
            )
        else:
            self.fx_history[fx_rate.currency_pair].add_date(t, fx_rate)

    def get_fx_rate(self, t: date, currency_pair: CurrencyPair) -> FX_Rate:
        if currency_pair in self.fx_history:
            return self.fx_history[currency_pair].get_fx_rate(t)
        else:
            raise KeyError(f"No {currency_pair} data loaded in market.")
