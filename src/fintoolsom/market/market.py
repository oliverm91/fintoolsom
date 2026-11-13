from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from .currencies import Currency, CurrencyPair, FX_Rate, FX_RateData
from .index import Index
from .index_history import IndexHistory
from ..rates import Rate, Curve, IndexCurve, LegacyCurve
from ..dates import Calendar, get_calendar
from ..dates.date_counts import date_ordinals

if TYPE_CHECKING:
    # Deferred: volatility_surface imports derivatives.calculator, which imports
    # Swap (derivatives.swaps -> market.currencies), so a runtime import here
    # would cycle back into this module before Market is defined.
    from .volatility_surface import VolatilitySurface


@dataclass(slots=True)
class Market:
    t: date
    fx_history: dict[CurrencyPair, FX_RateData] = field(default_factory=dict)
    indexes_history: dict[str, IndexHistory] = field(default_factory=dict)
    interest_rates: dict[str, dict[date, Rate]] = field(default_factory=dict)
    index_to_interest_rate_map: dict[str, str] = field(default_factory=dict)

    interest_rate_to_index_map: dict[str, str] = field(init=False, default_factory=dict)

    # The market's funding index (Chile: ICP): the numeraire its discount curves are
    # built under. Uncollateralised flows in its currency discount on its curve.
    risk_free_index: Index | None = None

    # Every curve, under its id (``curve.get_id()``):
    #   IndexCurve    -> "<index name>" (ICP, SOFR, a term index): projects that index
    #                    and discounts with it.
    #   DiscountCurve -> "<currency>_<risk-free index>" (e.g. "USD_ICP"): a currency
    #                    other than the risk-free index's, implied from cross-currency
    #                    products.
    # Collateralised discounting across currencies is not stored: get_dfs derives it.
    # Construction also accepts an iterable of curves.
    curves: dict[str, Curve] = field(default_factory=dict)
    volatility_surfaces: dict[CurrencyPair, VolatilitySurface] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if self.risk_free_index is not None and self.risk_free_index.currency is None:
            raise ValueError(
                f"risk_free_index '{self.risk_free_index.name}' needs a currency: it funds that currency."
            )
        if isinstance(self.curves, dict):
            for key, curve in self.curves.items():
                if isinstance(curve, Curve) and key != curve.get_id():
                    raise ValueError(
                        f"Curve stored under '{key}' has id '{curve.get_id()}'."
                    )
            given = list(self.curves.values())
        else:
            given = list(self.curves)
        self.curves = {}
        for curve in given:
            self.set_curve(curve)
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

    # ── Curves ───────────────────────────────────────────────────────────────

    def set_curve(self, curve: Curve) -> None:
        """Store `curve` under its id, replacing any curve with that id.

        Two indexes with the same name share an id: keeping names unique is the
        caller's responsibility.

        Raises
        ------
        TypeError
            If `curve` is not a :class:`Curve` (e.g. a legacy ZeroCouponCurve).
        """
        if isinstance(curve, LegacyCurve):
            raise TypeError(
                f"{type(curve).__name__} is a legacy curve and cannot be stored in a Market; "
                "use IndexCurve / DiscountCurve."
            )
        if not isinstance(curve, Curve):
            raise TypeError(
                f"Market curves must be Curve instances, got {type(curve).__name__}."
            )
        self.curves[curve.get_id()] = curve

    def get_index_curve(self, index: Index) -> IndexCurve:
        """Curve of `index`: it projects the index and discounts with it.

        Raises
        ------
        KeyError
            If the market has no curve for `index`.
        """
        curve = self.curves.get(index.name)
        if not isinstance(curve, IndexCurve):
            raise KeyError(f"No index curve for '{index.name}' in market.")
        return curve

    def get_discount_curve(
        self, currency: Currency, *, risk_free_index: Index | None = None
    ) -> Curve:
        """Uncollateralised discount curve of `currency` under `risk_free_index`.

        The risk-free index's own curve for its currency, else the stored
        ``"<currency>_<index>"`` :class:`DiscountCurve`.

        Parameters
        ----------
        currency : Currency
            Currency of the flows.
        risk_free_index : Index, optional
            Numeraire; defaults to :attr:`risk_free_index`.

        Raises
        ------
        ValueError
            If no risk-free index is given or set on the market.
        KeyError
            If no curve was built for that currency under that numeraire.
        """
        rfi = risk_free_index if risk_free_index is not None else self.risk_free_index
        if rfi is None:
            raise ValueError(
                "Market has no risk_free_index; cannot resolve discount curves."
            )
        if currency == rfi.currency:
            return self.get_index_curve(rfi)
        curve_id = f"{currency.value}_{rfi.name}"
        try:
            return self.curves[curve_id]
        except KeyError:
            raise KeyError(f"No discount curve '{curve_id}' in market.") from None

    def get_dfs(
        self,
        currency: Currency,
        dates: list[date],
        *,
        collateral: Index | None = None,
        risk_free_index: Index | None = None,
    ) -> np.ndarray:
        """Discount factors of `currency` flows paid on `dates`.

        Uncollateralised: one curve, :meth:`get_discount_curve`. Collateral in the
        flows' currency: the collateral index's curve. Collateral in another
        currency: ``disc(ccy) * curve(collateral) / disc(collateral.ccy)``.

        Parameters
        ----------
        currency : Currency
            Currency of the flows.
        dates : list of date
            Payment dates.
        collateral : Index, optional
            CSA collateral index; None means uncollateralised.
        risk_free_index : Index, optional
            Numeraire of the discount curves; defaults to :attr:`risk_free_index`.

        Returns
        -------
        numpy.ndarray
        """
        return self._dfs_at_ordinals(
            currency,
            date_ordinals(dates),
            collateral=collateral,
            risk_free_index=risk_free_index,
        )

    def _dfs_at_ordinals(
        self,
        currency: Currency,
        ordinals: np.ndarray,
        *,
        collateral: Index | None = None,
        risk_free_index: Index | None = None,
    ) -> np.ndarray:
        """:meth:`get_dfs` for dates already held as integer ordinals."""
        if collateral is None:
            return self.get_discount_curve(
                currency, risk_free_index=risk_free_index
            )._dfs_at_ordinals(ordinals)
        if collateral.currency == currency:
            return self.get_index_curve(collateral)._dfs_at_ordinals(ordinals)
        return (
            self.get_discount_curve(
                currency, risk_free_index=risk_free_index
            )._dfs_at_ordinals(ordinals)
            * self.get_index_curve(collateral)._dfs_at_ordinals(ordinals)
            / self.get_discount_curve(
                collateral.currency, risk_free_index=risk_free_index
            )._dfs_at_ordinals(ordinals)
        )

    def get_df(
        self,
        currency: Currency,
        t: date,
        *,
        collateral: Index | None = None,
        risk_free_index: Index | None = None,
    ) -> float:
        """Single-date :meth:`get_dfs`."""
        return float(
            self.get_dfs(
                currency, [t], collateral=collateral, risk_free_index=risk_free_index
            )[0]
        )

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
            raise ValueError(
                f"Rate {name} not found for date {t} and there were no past dates."
            )
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
            raise KeyError(
                f"No volatility surface found for currency pair {currency_pair}."
            )
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
                self.fx_history[fxr.currency_pair] = FX_RateData(
                    fxr.currency_pair, {t: fxr}
                )
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

    def get_future_fx_rate(self, t: date, currency_pair: CurrencyPair) -> FX_Rate:
        """FX rate of `currency_pair` fixing at `t`: known, or projected on the curves.

        Let ``L = get_fx_known_until(currency_pair)``: :attr:`t` for a traded
        pair, the last published date for a pair priced by an index known ahead
        of time (the UF, up to the 9th of next month once CPI is out). A fixing
        on or before ``L`` is the loaded rate (:meth:`get_fx_rate`); a later one
        is projected from the rate at ``L`` with forward discount factors on the
        funding curves (:meth:`get_discount_curve`)::

            FX(t) = FX(L) * [df_base(t) / df_base(L)] / [df_quote(t) / df_quote(L)]

        which is ``spot * df_base(t) / df_quote(t)`` when ``L = self.t``. Under a
        common CSA the collateral adjustment of both currencies cancels in this
        ratio, so the funding curves are used regardless of collateral.

        Parameters
        ----------
        t : date
            Fixing date, after :attr:`t`; use :meth:`get_fx_rate` for earlier ones.
        currency_pair : CurrencyPair
            Pair to project.

        Returns
        -------
        FX_Rate

        Raises
        ------
        ValueError
            If `t` is not after :attr:`t`.
        KeyError
            If a known fixing, the rate at ``L`` or a funding curve is not loaded.
        """
        if t <= self.t:
            raise ValueError(
                f"get_future_fx_rate needs a date after the market date {self.t}; "
                f"got {t}. Use get_fx_rate for known rates."
            )
        if t <= self.get_fx_known_until(currency_pair):
            return self.get_fx_rate(t, currency_pair)
        rate = self._future_fx_rates_at_ordinals(
            currency_pair, np.array([t.toordinal()])
        )[0]
        return FX_Rate(currency_pair, float(rate))

    def _future_fx_rates_at_ordinals(
        self, currency_pair: CurrencyPair, ordinals: np.ndarray
    ) -> np.ndarray:
        """:meth:`get_future_fx_rate` values for dates held as integer ordinals.

        Unlike :meth:`get_future_fx_rate`, dates on or before :attr:`t` are allowed
        and read the history (a fixed but unpaid swap flow).
        """
        base, quote = currency_pair.base_currency, currency_pair.quote_currency
        if base == quote:
            return np.ones(len(ordinals))
        known_until = self.get_fx_known_until(currency_pair)
        last_known = known_until.toordinal()
        known = ordinals <= last_known
        fx = np.empty(len(ordinals))
        for i in np.where(known)[0]:
            fx[i] = self.get_fx_rate(
                date.fromordinal(int(ordinals[i])), currency_pair
            ).value
        if (~known).any():
            anchor = self.get_fx_rate(known_until, currency_pair).value
            nodes = np.concatenate(([last_known], ordinals[~known]))
            df_base = self.get_discount_curve(base)._dfs_at_ordinals(nodes)
            df_quote = self.get_discount_curve(quote)._dfs_at_ordinals(nodes)
            fx[~known] = (
                anchor * (df_base[1:] / df_base[0]) / (df_quote[1:] / df_quote[0])
            )
        return fx

    def convert_value_currency(
        self, value: float, from_currency: Currency, to_currency: Currency
    ) -> float:
        """`value` in `from_currency` converted to `to_currency` at spot (:attr:`t`).

        Raises
        ------
        KeyError
            If the pair has no rate loaded at :attr:`t`.
        """
        pair = CurrencyPair(from_currency, to_currency)
        return value * self.get_fx_rate(self.t, pair).value

    def convert_future_value_currency(
        self, value: float, from_currency: Currency, to_currency: Currency, t: date
    ) -> float:
        """`value` in `from_currency` at `t` converted to `to_currency` at the FX
        fixing of `t` (:meth:`get_future_fx_rate`).

        Raises
        ------
        ValueError
            If `t` is not after :attr:`t`.
        KeyError
            As :meth:`get_future_fx_rate`.
        """
        pair = CurrencyPair(from_currency, to_currency)
        return value * self.get_future_fx_rate(t, pair).value
