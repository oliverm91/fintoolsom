from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from ..rates import ZeroCouponCurve, RateConvention, ExponentialInterestConvention
from ..dates import ActualDayCountConvention
from .forwards.forwards import Forward, NDF
from .options.options import Option

if TYPE_CHECKING:
    from ..market import Market, Locality

_default_option_rate_convention = RateConvention(
    interest_convention=ExponentialInterestConvention,
    day_count_convention=ActualDayCountConvention,
    time_fraction_base=365,
)


class Calculator:
    """Valuation logic for derivatives instruments. Instruments only hold data;
    every method here is a staticmethod that takes the instrument plus whatever
    market data it needs and returns a value. Nothing is cached or stored."""

    # --- Forwards ---

    @staticmethod
    def get_forward_mtm(
        forward: Forward,
        spot: float,
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
    ) -> float:
        df_d = domestic_curve.get_df(forward.payment_date)
        df_f = foreign_curve.get_df(forward.payment_date)
        notional_leg_vp = spot * forward.notional * df_f
        strike_leg_vp = forward.strike * forward.notional * df_d

        mtm = forward.sign * (notional_leg_vp - strike_leg_vp)
        return mtm

    @staticmethod
    def get_ndf_mtm(
        ndf: NDF,
        spot: float,
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
    ) -> float:
        df_d = domestic_curve.get_df(ndf.fixing_date)
        df_f = foreign_curve.get_df(ndf.fixing_date)

        s_t = spot * df_f / df_d

        fv = ndf.sign * ndf.notional * (s_t - ndf.strike)
        mtm = fv * domestic_curve.get_df(ndf.payment_date)
        return mtm

    @staticmethod
    def get_uf_forward_mtm(
        ndf: NDF,
        uf_history: dict[date, float],
        clp_curve: ZeroCouponCurve,
        uf_curve: ZeroCouponCurve,
    ) -> float:
        """Values an NDF settling against the Chilean UF (ndf.is_uf_indexed=True).
        The UF value is known in advance until the 9th of the following month. If
        the fixing_date falls beyond the last known UF print, the UF is projected
        from that last known value using the forward factor implied by the nominal
        CLP curve (clp_curve) and the UF-linked curve (uf_curve, i.e. CLF)."""
        known_uf = uf_history.get(ndf.fixing_date)
        if known_uf is not None:
            s_t = known_uf
        else:
            known_dates = [t for t in uf_history if t <= ndf.fixing_date]
            if len(known_dates) == 0:
                raise ValueError(
                    f"No UF value known on or before fixing_date {ndf.fixing_date} to project from."
                )
            last_known_date = max(known_dates)
            last_known_uf = uf_history[last_known_date]
            df_d_fwd = clp_curve.get_df_fwd(last_known_date, ndf.fixing_date)
            df_f_fwd = uf_curve.get_df_fwd(last_known_date, ndf.fixing_date)
            s_t = last_known_uf * df_f_fwd / df_d_fwd

        fv = ndf.sign * ndf.notional * (s_t - ndf.strike)
        mtm = fv * clp_curve.get_df(ndf.payment_date)
        return mtm

    # --- Options (Garman-Kohlhagen / Black-Scholes for FX) ---

    @staticmethod
    def get_option_log_moneyness(
        option: Option,
        spot: float,
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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
        domestic_curve: ZeroCouponCurve,
        foreign_curve: ZeroCouponCurve,
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

    # --- Dispatcher ---

    @staticmethod
    def valuate(
        instrument: Forward | NDF | Option,
        market: Market,
        locality: Locality = None,
        volatility: float = None,
        rate_convention: RateConvention = None,
    ) -> float:
        """Extracts whatever spot/curves/history the instrument needs from `market`
        and routes it to the matching valuation method. NDF must be checked before
        Forward since NDF is a subclass of Forward."""
        # Deferred import: market imports derivatives.calculator (via
        # volatility_surface), so importing market at module level here would cycle.
        from ..market.currencies import Currency

        if isinstance(instrument, NDF) and instrument.is_uf_indexed:
            clp_curve = market.get_zero_coupon_curve(Currency.CLP, locality=locality)
            uf_curve = market.get_zero_coupon_curve(Currency.CLP, index_name="UF")
            return Calculator.get_uf_forward_mtm(
                instrument, market.uf_history, clp_curve, uf_curve
            )

        if isinstance(instrument, (NDF, Forward, Option)):
            spot = market.get_fx_rate(market.t, instrument.currency_pair).value
            domestic_curve = market.get_zero_coupon_curve(
                instrument.currency_pair.quote_currency, locality=locality
            )
            foreign_curve = market.get_zero_coupon_curve(
                instrument.currency_pair.base_currency, locality=locality
            )

            if isinstance(instrument, NDF):
                return Calculator.get_ndf_mtm(
                    instrument, spot, domestic_curve, foreign_curve
                )
            if isinstance(instrument, Forward):
                return Calculator.get_forward_mtm(
                    instrument, spot, domestic_curve, foreign_curve
                )
            if volatility is None:
                raise ValueError(
                    "volatility must be provided to valuate an Option (Market does "
                    "not yet store volatility surfaces)."
                )
            return Calculator.get_option_mtm(
                instrument,
                market.t,
                spot,
                volatility,
                domestic_curve,
                foreign_curve,
                rate_convention,
            )

        raise NotImplementedError(
            f"No valuation routing implemented for instrument type {type(instrument).__name__}."
        )
