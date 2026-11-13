"""FX options: Garman-Kohlhagen (Black-Scholes for FX) pricing and greeks."""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtr, ndtri

from ...dates import ActualDayCountConvention
from ...rates import Curve, ExponentialInterestConvention, RateConvention

if TYPE_CHECKING:
    from ...market import Market
    from ..options.options import Option

_default_option_rate_convention = RateConvention(
    interest_convention=ExponentialInterestConvention,
    day_count_convention=ActualDayCountConvention,
    time_fraction_base=365,
)

_SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_pdf(x: float) -> float:
    """Standard normal density."""
    return math.exp(-x * x / 2) / _SQRT_2PI


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


def _get_option_yf(option: Option, t: date) -> float:
    return ActualDayCountConvention.get_time_fraction(t, option.maturity, 365)


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
    yf = _get_option_yf(option, t)
    r, q = _get_option_rates(option, t, domestic_curve, foreign_curve, rate_convention)
    vol_sqrt_yf = volatility * math.sqrt(yf)
    d1 = (
        math.log(spot / option.strike) + (r - q + volatility * volatility / 2) * yf
    ) / vol_sqrt_yf
    if not return_vol_sqrt_yf:
        return d1
    return d1, vol_sqrt_yf


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
    d1, vol_sqrt_yf = _get_option_d1(
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


def get_option_mtm(
    option: Option,
    t: date,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    rate_convention: RateConvention = None,
) -> float:
    d1, d2 = _get_option_d2(
        option,
        t,
        spot,
        volatility,
        domestic_curve,
        foreign_curve,
        rate_convention,
        return_both_ds=True,
    )
    r, q = _get_option_rates(option, t, domestic_curve, foreign_curve, rate_convention)
    yf = _get_option_yf(option, t)
    mtm = (
        option._sign
        * option.notional
        * (
            spot * math.exp(-q * yf) * ndtr(option._sign * d1)
            - option.strike * math.exp(-r * yf) * ndtr(option._sign * d2)
        )
    )
    return mtm


def get_option_delta(
    option: Option,
    t: date,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    rate_convention: RateConvention = None,
) -> float:
    _, q = _get_option_rates(option, t, domestic_curve, foreign_curve, rate_convention)
    yf = _get_option_yf(option, t)
    d1 = _get_option_d1(
        option, t, spot, volatility, domestic_curve, foreign_curve, rate_convention
    )
    delta = option._sign * option.notional * math.exp(-q * yf) * ndtr(option._sign * d1)
    return delta


def get_option_gamma(
    option: Option,
    t: date,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    rate_convention: RateConvention = None,
) -> float:
    _, q = _get_option_rates(option, t, domestic_curve, foreign_curve, rate_convention)
    yf = _get_option_yf(option, t)
    d1, vol_sqrt_yf = _get_option_d1(
        option,
        t,
        spot,
        volatility,
        domestic_curve,
        foreign_curve,
        rate_convention,
        return_vol_sqrt_yf=True,
    )
    gamma = option.notional * math.exp(-q * yf) * _norm_pdf(d1) / (spot * vol_sqrt_yf)
    return gamma


def get_option_vega(
    option: Option,
    t: date,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    rate_convention: RateConvention = None,
) -> float:
    _, q = _get_option_rates(option, t, domestic_curve, foreign_curve, rate_convention)
    yf = _get_option_yf(option, t)
    d1 = _get_option_d1(
        option, t, spot, volatility, domestic_curve, foreign_curve, rate_convention
    )
    vega = option.notional * spot * math.exp(-q * yf) * math.sqrt(yf) * _norm_pdf(d1)
    return vega


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
    yf = rc.day_count_convention.get_time_fraction(t, maturity, rc.time_fraction_base)
    r = rc.interest_convention.get_rate_from_df(df_r, yf)
    df_q = foreign_curve.get_df(maturity)
    q = rc.interest_convention.get_rate_from_df(df_q, yf)

    k = spot * np.exp(
        -(
            sign * ndtri(sign * delta * (1 / df_q)) * volatility * np.sqrt(yf)
            - (r - q + volatility * volatility / 2) * yf
        )
    )
    return k


def get_call_strike_from_delta(
    delta: float,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    maturity: date,
    rate_convention: RateConvention = None,
) -> float:
    return get_strike_from_delta(
        delta,
        spot,
        volatility,
        domestic_curve,
        foreign_curve,
        maturity,
        1,
        rate_convention,
    )


def get_put_strike_from_delta(
    delta: float,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    maturity: date,
    rate_convention: RateConvention = None,
) -> float:
    return get_strike_from_delta(
        delta,
        spot,
        volatility,
        domestic_curve,
        foreign_curve,
        maturity,
        -1,
        rate_convention,
    )


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
    domestic_curve = market.get_discount_curve(instrument.currency_pair.quote_currency)
    foreign_curve = market.get_discount_curve(instrument.currency_pair.base_currency)
    spot = market.get_fx_rate(market.t, instrument.currency_pair).value

    log_moneyness = get_option_log_moneyness(
        instrument, spot, domestic_curve, foreign_curve
    )
    days = (instrument.maturity - market.t).days
    vol_surface = market.get_volatility_surface(instrument.currency_pair)
    volatility = vol_surface.get_volatility(log_moneyness, days)

    return get_option_mtm(
        instrument, market.t, spot, volatility, domestic_curve, foreign_curve
    )
