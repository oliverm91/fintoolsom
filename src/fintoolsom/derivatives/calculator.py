from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from ..rates import Curve, RateConvention, ExponentialInterestConvention
from ..dates import ActualDayCountConvention
from .forwards.forwards import Forward, NDF
from .options.options import Option
from .swaps.swaps import Swap

if TYPE_CHECKING:
    from ..market import Market
    from ..market.currencies import Currency

_default_option_rate_convention = RateConvention(
    interest_convention=ExponentialInterestConvention,
    day_count_convention=ActualDayCountConvention,
    time_fraction_base=365,
)


class Calculator:
    """Valuation of derivative instruments.

    Instruments only hold data. Every method is a staticmethod taking the
    instrument plus the market data it needs; nothing is cached or stored.
    """

    # --- Forwards ---

    @staticmethod
    def _convert(value: float, from_currency: Currency, to_currency: Currency, market: Market) -> float:
        """`value` in `from_currency` converted to `to_currency` at spot (``market.t``)."""
        from ..market.currencies import CurrencyPair as _CurrencyPair

        return value * market.get_fx_rate(market.t, _CurrencyPair(from_currency, to_currency)).value

    @staticmethod
    def get_forward_mtm(
        forward: Forward,
        market: Market,
        currency: Currency,
    ) -> float:
        """Deliverable FX forward mark-to-market.

        Domestic is the quote currency and foreign the base currency, both
        discounted on their uncollateralised funding curves; spot is read from
        `market` at ``market.t``. A forward paid on or before ``market.t`` is worth 0.

        Parameters
        ----------
        forward : Forward
            Deliverable forward.
        market : Market
            Market at the valuation date.
        currency : Currency
            Reporting currency; the quote-currency MTM is converted at spot.

        Returns
        -------
        float
            MTM in `currency`.
        """
        cp = forward.currency_pair
        if forward.payment_date <= market.t:
            return 0.0
        domestic_curve = market.get_funding_curve(cp.quote_currency)
        foreign_curve = market.get_funding_curve(cp.base_currency)
        spot = market.get_fx_rate(market.t, cp).value

        df_d = domestic_curve.get_df(forward.payment_date)
        df_f = foreign_curve.get_df(forward.payment_date)
        notional_leg_vp = spot * forward.notional * df_f
        strike_leg_vp = forward.strike * forward.notional * df_d

        mtm = forward.sign * (notional_leg_vp - strike_leg_vp)
        return Calculator._convert(mtm, cp.quote_currency, currency, market)

    @staticmethod
    def get_ndf_mtm(
        ndf: NDF,
        market: Market,
        currency: Currency,
    ) -> float:
        """Non-deliverable FX forward mark-to-market.

        The settlement FX at `fixing_date` comes from :meth:`get_fx_fixings`
        (realised or already published, else projected on the funding curves) and
        the settlement amount is discounted from `payment_date` on the domestic
        (quote-currency) funding curve. A UF forward is an NDF on ``CLF/CLP``. An
        NDF paid on or before ``market.t`` is worth 0.

        Parameters
        ----------
        ndf : NDF
            Non-deliverable forward.
        market : Market
            Market at the valuation date.
        currency : Currency
            Reporting currency; the quote-currency MTM is converted at spot.

        Returns
        -------
        float
            MTM in `currency`.
        """
        cp = ndf.currency_pair
        if ndf.payment_date <= market.t:
            return 0.0
        s_t = Calculator.get_fx_fixings(
            cp.base_currency, cp.quote_currency, [ndf.fixing_date], market
        )[0]
        fv = ndf.sign * ndf.notional * (s_t - ndf.strike)
        mtm = fv * market.get_funding_curve(cp.quote_currency).get_df(ndf.payment_date)
        return Calculator._convert(mtm, cp.quote_currency, currency, market)

    # --- Options (Garman-Kohlhagen / Black-Scholes for FX) ---

    @staticmethod
    def get_option_log_moneyness(
        option: Option,
        spot: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
    ) -> float:
        df_r = domestic_curve.get_df(option.maturity)
        df_q = foreign_curve.get_df(option.maturity)
        fwd_price = spot * df_q / df_r
        return np.log(option.strike / fwd_price)

    @staticmethod
    def _get_option_yf(option: Option, t: date) -> float:
        return ActualDayCountConvention.get_time_fraction(t, option.maturity, 365)

    @staticmethod
    def _get_option_rates(
        option: Option,
        t: date,
        domestic_curve: Curve,
        foreign_curve: Curve,
        rate_convention: RateConvention = None,
    ) -> tuple[float, float]:
        rc = rate_convention or _default_option_rate_convention
        interest_conv = rc.interest_convention
        yf = rc.day_count_convention.get_time_fraction(
            t, option.maturity, rc.time_fraction_base
        )
        df_r = domestic_curve.get_df(option.maturity)
        r = interest_conv.get_rate_from_df(df_r, yf)
        df_q = foreign_curve.get_df(option.maturity)
        q = interest_conv.get_rate_from_df(df_q, yf)
        return r, q

    @staticmethod
    def _get_option_d1(
        option: Option,
        t: date,
        spot: float,
        volatility: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        rate_convention: RateConvention = None,
        return_vol_sqrt_yf: bool = False,
    ) -> float | tuple[float, float]:
        yf = Calculator._get_option_yf(option, t)
        r, q = Calculator._get_option_rates(
            option, t, domestic_curve, foreign_curve, rate_convention
        )
        vol_sqrt_yf = volatility * math.sqrt(yf)
        d1 = (
            math.log(spot / option.strike) + (r - q + volatility * volatility / 2) * yf
        ) / vol_sqrt_yf
        if not return_vol_sqrt_yf:
            return d1
        return d1, vol_sqrt_yf

    @staticmethod
    def _get_option_d2(
        option: Option,
        t: date,
        spot: float,
        volatility: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        rate_convention: RateConvention = None,
        return_both_ds: bool = False,
    ) -> float | tuple[float, float]:
        d1, vol_sqrt_yf = Calculator._get_option_d1(
            option,
            t,
            spot,
            volatility,
            domestic_curve,
            foreign_curve,
            rate_convention,
            return_vol_sqrt_yf=True,
        )
        d2 = d1 - vol_sqrt_yf
        if not return_both_ds:
            return d2
        return d1, d2

    @staticmethod
    def get_option_mtm(
        option: Option,
        t: date,
        spot: float,
        volatility: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        rate_convention: RateConvention = None,
    ) -> float:
        d1, d2 = Calculator._get_option_d2(
            option,
            t,
            spot,
            volatility,
            domestic_curve,
            foreign_curve,
            rate_convention,
            return_both_ds=True,
        )
        r, q = Calculator._get_option_rates(
            option, t, domestic_curve, foreign_curve, rate_convention
        )
        yf = Calculator._get_option_yf(option, t)
        mtm = (
            option._sign
            * option.notional
            * (
                spot * math.exp(-q * yf) * norm.cdf(option._sign * d1)
                - option.strike * math.exp(-r * yf) * norm.cdf(option._sign * d2)
            )
        )
        return mtm

    @staticmethod
    def get_option_delta(
        option: Option,
        t: date,
        spot: float,
        volatility: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        rate_convention: RateConvention = None,
    ) -> float:
        _, q = Calculator._get_option_rates(
            option, t, domestic_curve, foreign_curve, rate_convention
        )
        yf = Calculator._get_option_yf(option, t)
        d1 = Calculator._get_option_d1(
            option, t, spot, volatility, domestic_curve, foreign_curve, rate_convention
        )
        delta = (
            option._sign
            * option.notional
            * math.exp(-q * yf)
            * norm.cdf(option._sign * d1)
        )
        return delta

    @staticmethod
    def get_option_gamma(
        option: Option,
        t: date,
        spot: float,
        volatility: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        rate_convention: RateConvention = None,
    ) -> float:
        _, q = Calculator._get_option_rates(
            option, t, domestic_curve, foreign_curve, rate_convention
        )
        yf = Calculator._get_option_yf(option, t)
        d1, vol_sqrt_yf = Calculator._get_option_d1(
            option,
            t,
            spot,
            volatility,
            domestic_curve,
            foreign_curve,
            rate_convention,
            return_vol_sqrt_yf=True,
        )
        gamma = (
            option.notional * math.exp(-q * yf) * norm.pdf(d1) / (spot * vol_sqrt_yf)
        )
        return gamma

    @staticmethod
    def get_option_vega(
        option: Option,
        t: date,
        spot: float,
        volatility: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        rate_convention: RateConvention = None,
    ) -> float:
        _, q = Calculator._get_option_rates(
            option, t, domestic_curve, foreign_curve, rate_convention
        )
        yf = Calculator._get_option_yf(option, t)
        d1 = Calculator._get_option_d1(
            option, t, spot, volatility, domestic_curve, foreign_curve, rate_convention
        )
        vega = option.notional * spot * math.exp(-q * yf) * math.sqrt(yf) * norm.pdf(d1)
        return vega

    @staticmethod
    def get_strike_from_delta(
        delta: float,
        spot: float,
        volatility: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        maturity: date,
        sign: int,
        rate_convention: RateConvention = None,
    ) -> float:
        rc = rate_convention or _default_option_rate_convention
        df_r = domestic_curve.get_df(maturity)
        t = domestic_curve.curve_date
        yf = rc.day_count_convention.get_time_fraction(
            t, maturity, rc.time_fraction_base
        )
        r = rc.interest_convention.get_rate_from_df(df_r, yf)
        df_q = foreign_curve.get_df(maturity)
        q = rc.interest_convention.get_rate_from_df(df_q, yf)

        k = spot * np.exp(
            -(
                sign * norm.ppf(sign * delta * (1 / df_q)) * volatility * np.sqrt(yf)
                - (r - q + volatility * volatility / 2) * yf
            )
        )
        return k

    @staticmethod
    def get_call_strike_from_delta(
        delta: float,
        spot: float,
        volatility: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        maturity: date,
        rate_convention: RateConvention = None,
    ) -> float:
        return Calculator.get_strike_from_delta(
            delta,
            spot,
            volatility,
            domestic_curve,
            foreign_curve,
            maturity,
            1,
            rate_convention,
        )

    @staticmethod
    def get_put_strike_from_delta(
        delta: float,
        spot: float,
        volatility: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        maturity: date,
        rate_convention: RateConvention = None,
    ) -> float:
        return Calculator.get_strike_from_delta(
            delta,
            spot,
            volatility,
            domestic_curve,
            foreign_curve,
            maturity,
            -1,
            rate_convention,
        )

    # --- Swaps ---

    @staticmethod
    def _leg_flows(leg, market: Market) -> tuple[np.ndarray, np.ndarray]:
        """Flows, in the leg currency, of the coupons not yet paid.

        A coupon paid on or before ``market.t`` is dropped.

        Realised fixings come from the index history (``market.get_index``).

        TermRateLeg, split by fixing date:

        * fixed (``fixing_date <= t``): the fixed rate over the whole coupon.
        * to project (``fixing_date > t``): forward accrual from the projection curve.

        OvernightLeg and XCCYFloatingLeg, split by start date:

        * running (``start <= t < payment``): realised ``[start, t]`` (index business
          days, or ICP level ratio) compounded with the projection ``[t, end]``, or
          fully realised when ``end <= t``.
        * future (``start > t``): forward accrual from the projection curve.

        Parameters
        ----------
        leg : SwapLeg
            Fixed, term-rate, overnight or cross-currency floating leg.
        market : Market
            Market at the valuation date.

        Returns
        -------
        future_payment : numpy.ndarray of bool
            Per-coupon mask of the coupons paid after ``market.t``.
        flows : numpy.ndarray
            Flows of those coupons (interest + spread + amortization).

        Raises
        ------
        KeyError
            If a realised fixing is needed and is missing from the index history.
        """
        from .swaps.legs import FixedLeg as _FixedLeg, TermRateLeg as _TermRateLeg

        future_payment = np.array([t > market.t for t in leg.payment_dates], dtype=bool)
        if not future_payment.any():
            return future_payment, np.zeros(0)

        if isinstance(leg, _FixedLeg):
            return future_payment, leg.flows[future_payment]

        flows = np.zeros(int(future_payment.sum()))
        # Projection goes through get_accrual (interest per unit notional): the index
        # discount curve viewed as forwards, or an independent ProjectionCurve.
        proj_curve = market.get_projection(leg.index)

        if isinstance(leg, _TermRateLeg):
            # Once fixing_date <= t the rate is locked, wherever start_date falls.
            fixed_mask = future_payment & np.array([d <= market.t for d in leg.fixing_dates], dtype=bool)
            to_proj_mask = future_payment & ~fixed_mask

            if fixed_mask.any():
                history = Calculator._index_history(leg, market)
                fixed_flows = [
                    history.get_rate(leg.fixing_dates[i]).get_accrued_interest(
                        leg.residuals[i], leg.start_dates[i], leg.end_dates[i]
                    )
                    + leg.spreads_values[i]
                    + leg.amortizations[i]
                    for i in np.where(fixed_mask)[0]
                ]
                flows[fixed_mask[future_payment]] = np.array(fixed_flows)

            if to_proj_mask.any():
                starts = [s for s, f in zip(leg.start_dates, to_proj_mask) if f]
                ends = [e for e, f in zip(leg.end_dates, to_proj_mask) if f]
                accruals = proj_curve.get_accrual(starts, ends)
                flows[to_proj_mask[future_payment]] = leg.residuals[to_proj_mask] * accruals + leg.spreads_values[to_proj_mask] + leg.amortizations[to_proj_mask]

        else:
            future_start = np.array([s > market.t for s in leg.start_dates], dtype=bool)
            pure_future = future_payment & future_start
            current_mask = future_payment & ~future_start

            if pure_future.any():
                starts = [s for s, f in zip(leg.start_dates, pure_future) if f]
                ends = [e for e, f in zip(leg.end_dates, pure_future) if f]
                accruals = proj_curve.get_accrual(starts, ends)
                # amortizations: the notional repaid with the final coupon (final
                # exchange). Same-currency legs cancel it; cross-currency legs exchange it.
                flows[pure_future[future_payment]] = (
                    leg.residuals[pure_future] * accruals
                    + leg.spreads_values[pure_future]
                    + leg.amortizations[pure_future]
                )

            if current_mask.any():
                idx = int(np.where(current_mask)[0][0])
                c = leg.coupons[idx]
                # Realised part from the index history: compounded on the index's own
                # business days (rate index) or the ratio of published levels (ICP).
                realised_end = min(market.t, c.end_date)
                accrued = Calculator._index_history(leg, market).get_accrued_interest(
                    c.residual, c.start_date, realised_end
                )
                interest = accrued
                if realised_end < c.end_date:
                    interest += (c.residual + accrued) * proj_curve.get_accrual(market.t, c.end_date)
                flows[current_mask[future_payment]] = interest + leg.spreads_values[idx] + leg.amortizations[idx]

        return future_payment, flows

    @staticmethod
    def _index_history(leg, market: Market):
        """History of the leg index, holding its realised fixings.

        Raises
        ------
        KeyError
            If `market` has no history for the index.
        """
        return market.get_index(leg.index.name)

    @staticmethod
    def _leg_pv(leg, collateral, market: Market) -> float:
        """PV of a leg settled in its own currency.

        Parameters
        ----------
        leg : SwapLeg
            Leg to value.
        collateral : InterestIndex or None
            CSA collateral index; None means uncollateralised.
        market : Market
            Market at the valuation date.

        Returns
        -------
        float
            PV in the leg currency.
        """
        future_payment, flows = Calculator._leg_flows(leg, market)
        if not future_payment.any():
            return 0.0
        pay_dates = [d for d, f in zip(leg.payment_dates, future_payment) if f]
        dfs = market.get_discount_dfs(leg.currency, pay_dates, collateral)
        return float(np.dot(flows, dfs))

    @staticmethod
    def get_fx_fixings(
        base_currency: Currency, quote_currency: Currency, fixing_dates: list[date], market: Market
    ) -> np.ndarray:
        """FX rates (quote per unit of base) fixed on `fixing_dates`.

        Let ``L = market.get_fx_known_until(pair)``: ``market.t`` for a traded
        pair, the last published date for a pair priced by an index known ahead
        of time (the UF, up to the 9th of next month once CPI is out). A fixing on
        or before ``L`` is read as is (FX history, or UF history including its
        already-published future levels); a later one is projected from the rate
        at ``L`` with forward discount factors on the funding curves::

            FX(d) = FX(L) * [df_base(d) / df_base(L)] / [df_quote(d) / df_quote(L)]

        which is ``spot * df_base(d) / df_quote(d)`` when ``L = market.t``. Under
        a common CSA the collateral adjustment of both currencies cancels in this
        ratio, so funding curves are used regardless.

        Parameters
        ----------
        base_currency, quote_currency : Currency
            Currency pair.
        fixing_dates : list of date
            Fixing dates.
        market : Market
            Market at the valuation date.

        Returns
        -------
        numpy.ndarray
            One FX rate per fixing date.

        Raises
        ------
        KeyError
            If a known fixing, or the rate at ``L``, is not loaded.
        """
        from ..market.currencies import CurrencyPair as _CurrencyPair

        cp = _CurrencyPair(base_currency, quote_currency)
        known_until = market.get_fx_known_until(cp)
        known = np.array([d <= known_until for d in fixing_dates], dtype=bool)
        fx = np.empty(len(fixing_dates))
        for i in np.where(known)[0]:
            fx[i] = market.get_fx_rate(fixing_dates[i], cp).value
        if (~known).any():
            anchor = market.get_fx_rate(known_until, cp).value
            projected = [d for d, k in zip(fixing_dates, known) if not k]
            dates = [known_until] + projected
            df_base = market.get_funding_curve(base_currency).get_dfs(dates)
            df_quote = market.get_funding_curve(quote_currency).get_dfs(dates)
            fx[~known] = anchor * (df_base[1:] / df_base[0]) / (df_quote[1:] / df_quote[0])
        return fx

    @staticmethod
    def _leg_pv_non_deliverable(
        leg, collateral, market: Market, settlement_currency: Currency
    ) -> float:
        """PV of a leg settled non-deliverably in `settlement_currency`.

        Each unpaid flow is converted at its FX fixing (realised if fixed on or
        before ``market.t``, including a flow fixed but not yet paid; projected
        otherwise) and discounted from its payment date in `settlement_currency`.

        Parameters
        ----------
        leg : SwapLeg
            Leg to value.
        collateral : InterestIndex or None
            CSA collateral index; None means uncollateralised.
        market : Market
            Market at the valuation date.
        settlement_currency : Currency
            Currency the flows are paid in.

        Returns
        -------
        float
            PV in `settlement_currency`.
        """
        if leg.currency == settlement_currency:
            return Calculator._leg_pv(leg, collateral, market)
        future_payment, flows = Calculator._leg_flows(leg, market)
        if not future_payment.any():
            return 0.0
        pay_dates = [d for d, f in zip(leg.payment_dates, future_payment) if f]
        fixing_dates = [d for d, f in zip(leg.fx_fixing_dates, future_payment) if f]
        fx = Calculator.get_fx_fixings(leg.currency, settlement_currency, fixing_dates, market)
        dfs = market.get_discount_dfs(settlement_currency, pay_dates, collateral)
        return float(np.sum(flows * fx * dfs))

    @staticmethod
    def get_swap_mtm(swap: Swap, market: Market, currency: Currency) -> float:
        """Swap mark-to-market in the reporting `currency`.

        Each leg settles in :meth:`Swap.leg_settlement_currency`. A leg settled in
        its own currency is valued there; otherwise (non-deliverable swap, or a
        UF leg, always paid in CLP) every flow is converted at its FX fixing and
        discounted in the settlement currency. Each leg PV is then converted to
        `currency` at spot. Only the final notional exchange (the last coupon's
        amortization) is valued; there is no initial exchange.

        Parameters
        ----------
        swap : Swap
            Swap to value.
        market : Market
            Market at the valuation date.
        currency : Currency
            Reporting currency.

        Returns
        -------
        float
            PV of the received leg minus PV of the paid leg, in `currency`.
        """
        def _pv(leg) -> float:
            settlement = swap.leg_settlement_currency(leg)
            pv = Calculator._leg_pv_non_deliverable(leg, swap.collateral_index, market, settlement)
            return Calculator._convert(pv, settlement, currency, market)

        return _pv(swap.receive_leg) - _pv(swap.pay_leg)

    # --- Dispatcher ---

    @staticmethod
    def get_mtm_option(instrument: Option, market: Market) -> float:
        """FX option mark-to-market (Garman-Kohlhagen) from market data.

        Curves are the funding curves of each currency of the pair; volatility is
        read from the pair's volatility surface at the option's log-moneyness.

        Parameters
        ----------
        instrument : Option
            FX option.
        market : Market
            Market at the valuation date.

        Returns
        -------
        float
            MTM in the quote currency.
        """
        domestic_curve = market.get_funding_curve(instrument.currency_pair.quote_currency)
        foreign_curve = market.get_funding_curve(instrument.currency_pair.base_currency)
        spot = market.get_fx_rate(market.t, instrument.currency_pair).value

        log_moneyness = Calculator.get_option_log_moneyness(
            instrument, spot, domestic_curve, foreign_curve
        )
        days = (instrument.maturity - market.t).days
        vol_surface = market.get_volatility_surface(instrument.currency_pair)
        volatility = vol_surface.get_volatility(log_moneyness, days)

        return Calculator.get_option_mtm(
            instrument, market.t, spot, volatility, domestic_curve, foreign_curve
        )

    @staticmethod
    def valuate(instrument: Forward | NDF | Option | Swap, market: Market, currency: Currency) -> float:
        """Mark-to-market of any supported instrument.

        NDF is checked before Forward, since NDF subclasses Forward.

        Parameters
        ----------
        instrument : Forward, NDF, Option or Swap
            Instrument to value.
        market : Market
            Market at the valuation date.
        currency : Currency
            Reporting currency.

        Returns
        -------
        float

        Raises
        ------
        NotImplementedError
            For an unsupported instrument type.
        """
        if isinstance(instrument, NDF):
            return Calculator.get_ndf_mtm(instrument, market, currency)
        if isinstance(instrument, Forward):
            return Calculator.get_forward_mtm(instrument, market, currency)
        if isinstance(instrument, Swap):
            return Calculator.get_swap_mtm(instrument, market, currency)
        if isinstance(instrument, Option):
            return Calculator.get_mtm_option(instrument, market)

        raise NotImplementedError(
            f"No valuation routing implemented for instrument type {type(instrument).__name__}."
        )
