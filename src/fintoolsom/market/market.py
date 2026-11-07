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
from ..dates import Calendar, get_calendar

if TYPE_CHECKING:
    # Deferred: volatility_surface imports derivatives.calculator, which imports
    # Swap (derivatives.swaps -> market.currencies), so a runtime import here
    # would cycle back into this module before Market is defined.
    from .volatility_surface import VolatilitySurface


# A discount curve is identified by an Index (that index's own curve) or by a
# Currency (that currency's uncollateralised, cross-currency-implied curve).
CurveKey = Index | Currency


def funding_key(riskless_index: Index, currency: Currency) -> CurveKey:
    """Key of the uncollateralised discount curve of `currency`.

    Parameters
    ----------
    riskless_index : Index
        The market funding index (Chile: ICP).
    currency : Currency
        Currency of the flows.

    Returns
    -------
    Index or Currency
        `riskless_index` for its own currency; otherwise `currency`, whose curve
        is implied from cross-currency products.
    """
    if currency == riskless_index.currency:
        return riskless_index
    return currency


@dataclass(slots=True)
class Market:
    t: date
    fx_history: dict[CurrencyPair, FX_RateData] = field(default_factory=dict)
    indexes_history: dict[str, IndexHistory] = field(default_factory=dict)
    interest_rates: dict[str, dict[date, Rate]] = field(default_factory=dict)
    index_to_interest_rate_map: dict[str, str] = field(default_factory=dict)

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

    def get_curve(self, key: CurveKey) -> Curve:
        """Stored discount curve under `key`.

        Parameters
        ----------
        key : Index or Currency
            Curve key.

        Returns
        -------
        Curve

        Raises
        ------
        KeyError
            If no curve is stored under `key`.
        """
        try:
            return self.curves[key]
        except KeyError:
            kind = "index" if isinstance(key, Index) else "currency"
            raise KeyError(f"No discount curve for {kind} '{key}' in market.") from None

    def funding_key(self, currency: Currency) -> CurveKey:
        """Key of the uncollateralised curve of `currency`; see :func:`funding_key`.

        Raises
        ------
        ValueError
            If the market has no `riskless_index`.
        """
        if self.riskless_index is None:
            raise ValueError("Market has no riskless_index; cannot resolve funding curves.")
        return funding_key(self.riskless_index, currency)

    def get_funding_curve(self, currency: Currency) -> Curve:
        """Uncollateralised discount curve of `currency`.

        Parameters
        ----------
        currency : Currency
            Currency of the flows.

        Returns
        -------
        Curve
        """
        return self.get_curve(self.funding_key(currency))

    def get_discount_dfs(
        self, currency: Currency, dates: list[date], collateral: Index | None = None
    ) -> np.ndarray:
        """Discount factors of `currency` flows paid on `dates`.

        ``df = funding(ccy) * curve(collateral) / funding(collateral.ccy)``. When the
        collateral is the funding index of its own currency the ratio is 1: CLP
        flows under ICP collateral in an ICP market discount on plain ICP.

        Parameters
        ----------
        currency : Currency
            Currency of the flows.
        dates : list of date
            Payment dates.
        collateral : Index, optional
            CSA collateral index; None means uncollateralised.

        Returns
        -------
        numpy.ndarray
        """
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
        """Single-date :meth:`get_discount_dfs`."""
        return float(self.get_discount_dfs(currency, [t], collateral)[0])

    def get_projection(
        self, index: Index, *, interpolation_method=None
    ) -> ProjectionCurve | DiscountProjectionView:
        """Forward-projection curve of `index`, exposing ``get_accrual``.

        Returns the stored :class:`ProjectionCurve` of an index with an independent
        basis; otherwise a :class:`DiscountProjectionView` over the index's own
        discount curve, so an index's curve is never stored twice.

        Parameters
        ----------
        index : Index
            Index to project.
        interpolation_method : optional
            Reserved for a resampled view; currently ignored.

        Returns
        -------
        ProjectionCurve or DiscountProjectionView

        Raises
        ------
        KeyError
            If `index` has neither a projection nor a discount curve.
        """
        if index in self.projection_curves:
            return self.projection_curves[index]
        if index not in self.curves:
            raise KeyError(f"No projection or discount curve for index '{index.name}' in market.")
        return DiscountProjectionView(self.curves[index])

    def set_projection(self, index: Index, curve: ProjectionCurve) -> None:
        """Store the independent forward curve of a projection-only `index`.

        Parameters
        ----------
        index : Index
            Projection-only index.
        curve : ProjectionCurve
            Its forward curve.

        Raises
        ------
        ValueError
            If `index` already has a discount curve, which is its projection.
        """
        if index in self.curves:
            raise ValueError(
                f"Index '{index.name}' already has a discount curve; it projects off that "
                "curve and cannot also hold an independent projection curve."
            )
        self.projection_curves[index] = curve

    def add_index(self, history: IndexHistory):
        """Register `history` under its index name (case-insensitive)."""
        self.indexes_history[history.name.upper()] = history

    def get_index(self, name: str) -> IndexHistory:
        """History of the index `name` (case-insensitive).

        Raises
        ------
        KeyError
            If no history is registered under `name`.
        """
        name = name.upper()
        if name in self.indexes_history:
            return self.indexes_history[name]
        raise KeyError(f"Index {name} not found in market.")

    def get_rate(self, t: date, name: str, use_closest_past_rate: bool = False) -> Rate:
        """Rate `name` fixed on `t`, from ``interest_rates``.

        Parameters
        ----------
        t : date
            Fixing date.
        name : str
            Rate series name.
        use_closest_past_rate : bool, default False
            Fall back to the latest fixing before `t` when there is none on `t`.

        Returns
        -------
        Rate

        Raises
        ------
        ValueError
            If the series is unknown, or no fixing is found.
        """
        if name not in self.interest_rates:
            raise ValueError(f"Rate {name} not found.")
        series = self.interest_rates[name]
        if t in series:
            return series[t]
        if not use_closest_past_rate:
            raise ValueError(f"Rate {name} not found for date {t}.")
        past_dates = [d for d in series if d < t]
        if not past_dates:
            raise ValueError(f"Rate {name} not found for date {t} and there were no past dates.")
        return series[max(past_dates)]

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
            calendar = get_calendar()
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
                f"First reset date must be on or before the start date. Got Start date: {start_date} and min reset date: {reset_dates[0]}."
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
            acrrued_interest += rate.get_accrued_interest(
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

    def get_fx_known_until(self, currency_pair: CurrencyPair) -> date:
        """Last date whose rate of `currency_pair` is known as of ``self.t``.

        The latest date loaded for the pair when it is on or after ``self.t``,
        else ``self.t``. A traded pair has nothing loaded after today, so this is
        today; the UF (CLF/CLP) is published ahead of time, so loading its
        published levels makes them known (see
        :func:`fintoolsom.market.uf.last_published_date`). Rates after this date
        are projected on the curves; do not load assumed levels (e.g. from a
        forecast CPI) unless they are to be treated as fixed.

        Parameters
        ----------
        currency_pair : CurrencyPair
            Pair considered.

        Returns
        -------
        date
        """
        data = self.fx_history.get(currency_pair)
        if data is None or not data.historic_data:
            return self.t
        return max(self.t, max(data.historic_data))

    def add_fx_rate(self, t: date, fx_rate: FX_Rate):
        """Store `fx_rate` at `t`, together with its inverse.

        Parameters
        ----------
        t : date
            Fixing date.
        fx_rate : FX_Rate
            Rate to store.
        """
        for fxr in (fx_rate, fx_rate.invert()):
            data = self.fx_history.get(fxr.currency_pair)
            if data is None:
                self.fx_history[fxr.currency_pair] = FX_RateData(fxr.currency_pair, {t: fxr})
            else:
                data.add_date(t, fxr)

    def get_fx_rate(self, t: date, currency_pair: CurrencyPair) -> FX_Rate:
        """FX rate of `currency_pair` at `t` (1 for a currency against itself).

        Parameters
        ----------
        t : date
            Fixing date.
        currency_pair : CurrencyPair
            Pair to look up.

        Returns
        -------
        FX_Rate

        Raises
        ------
        KeyError
            If the pair or the date is not loaded.
        """
        if currency_pair.base_currency == currency_pair.quote_currency:
            return FX_Rate(currency_pair, 1.0)
        if currency_pair in self.fx_history:
            return self.fx_history[currency_pair].get_fx_rate(t)
        raise KeyError(f"No {currency_pair} data loaded in market.")
