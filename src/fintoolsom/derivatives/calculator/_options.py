"""FX options: Garman-Kohlhagen (Black-Scholes for FX) pricing and greeks."""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtr, ndtri

from ...dates import ActualDayCountConvention
from ...rates import Curve

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency
    from ..options.options import Option

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
    """Act/365 years from `t` to the option's maturity."""
    return ActualDayCountConvention.get_time_fraction(t, option.maturity, 365)


def _option_dfs(
    maturity: date, t: date, domestic_curve: Curve, foreign_curve: Curve
) -> tuple[float, float]:
    """Domestic and foreign discount factors from `t` to `maturity`."""
    df_r = domestic_curve.get_df(maturity) / domestic_curve.get_df(t)
    df_q = foreign_curve.get_df(maturity) / foreign_curve.get_df(t)
    return df_r, df_q


def _get_option_rates(
    option: Option,
    t: date,
    domestic_curve: Curve,
    foreign_curve: Curve,
) -> tuple[float, float]:
    """Domestic and foreign rates ``r, q`` to maturity, continuously compounded
    Act/365: the convention of the Garman-Kohlhagen formulas, so that
    ``exp(-r * yf)`` and ``exp(-q * yf)`` are the curves' discount factors."""
    yf = _get_option_yf(option, t)
    df_r, df_q = _option_dfs(option.maturity, t, domestic_curve, foreign_curve)
    return -math.log(df_r) / yf, -math.log(df_q) / yf


def _get_option_d1(
    option: Option,
    t: date,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    return_vol_sqrt_yf: bool = False,
) -> float | tuple[float, float]:
    yf = _get_option_yf(option, t)
    r, q = _get_option_rates(option, t, domestic_curve, foreign_curve)
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
    return_both_ds: bool = False,
) -> float | tuple[float, float]:
    d1, vol_sqrt_yf = _get_option_d1(
        option,
        t,
        spot,
        volatility,
        domestic_curve,
        foreign_curve,
        return_vol_sqrt_yf=True,
    )
    d2 = d1 - vol_sqrt_yf
    if not return_both_ds:
        return d2
    return d1, d2


def _expired(option: Option, t: date) -> bool:
    """No time value left: `t` is on or after the maturity."""
    return option.maturity <= t


def _in_the_money(option: Option, spot: float) -> bool:
    return option._sign * (spot - option.strike) > 0


def get_option_mtm(
    option: Option,
    t: date,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
) -> float:
    """Garman-Kohlhagen MTM in the quote currency.

    On the maturity date the option is worth its intrinsic value,
    ``notional * max(sign * (spot - strike), 0)``; after it, nothing."""
    if _expired(option, t):
        if option.maturity < t:
            return 0.0
        return option.notional * max(option._sign * (spot - option.strike), 0.0)
    d1, d2 = _get_option_d2(
        option,
        t,
        spot,
        volatility,
        domestic_curve,
        foreign_curve,
        return_both_ds=True,
    )
    df_r, df_q = _option_dfs(option.maturity, t, domestic_curve, foreign_curve)
    mtm = (
        option._sign
        * option.notional
        * (
            spot * df_q * ndtr(option._sign * d1)
            - option.strike * df_r * ndtr(option._sign * d2)
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
) -> float:
    """Spot delta. On the maturity date ``sign * notional`` in the money, else 0;
    0 after it."""
    if _expired(option, t):
        if option.maturity == t and _in_the_money(option, spot):
            return option._sign * option.notional
        return 0.0
    _, df_q = _option_dfs(option.maturity, t, domestic_curve, foreign_curve)
    d1 = _get_option_d1(option, t, spot, volatility, domestic_curve, foreign_curve)
    delta = option._sign * option.notional * df_q * ndtr(option._sign * d1)
    return delta


def get_option_gamma(
    option: Option,
    t: date,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
) -> float:
    """Gamma; 0 on or after the maturity date."""
    if _expired(option, t):
        return 0.0
    _, df_q = _option_dfs(option.maturity, t, domestic_curve, foreign_curve)
    d1, vol_sqrt_yf = _get_option_d1(
        option,
        t,
        spot,
        volatility,
        domestic_curve,
        foreign_curve,
        return_vol_sqrt_yf=True,
    )
    gamma = option.notional * df_q * _norm_pdf(d1) / (spot * vol_sqrt_yf)
    return gamma


def get_option_vega(
    option: Option,
    t: date,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
) -> float:
    """Vega (per unit of volatility); 0 on or after the maturity date."""
    if _expired(option, t):
        return 0.0
    _, df_q = _option_dfs(option.maturity, t, domestic_curve, foreign_curve)
    yf = _get_option_yf(option, t)
    d1 = _get_option_d1(option, t, spot, volatility, domestic_curve, foreign_curve)
    vega = option.notional * spot * df_q * math.sqrt(yf) * _norm_pdf(d1)
    return vega


def get_strike_from_delta(
    delta: float,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    maturity: date,
    sign: int,
) -> float:
    """Strike of the option with spot delta `delta` (per unit notional) at
    `maturity`, as of the domestic curve's date; rates continuous Act/365."""
    t = domestic_curve.curve_date
    yf = ActualDayCountConvention.get_time_fraction(t, maturity, 365)
    df_r, df_q = _option_dfs(maturity, t, domestic_curve, foreign_curve)
    r, q = -math.log(df_r) / yf, -math.log(df_q) / yf

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
) -> float:
    return get_strike_from_delta(
        delta, spot, volatility, domestic_curve, foreign_curve, maturity, 1
    )


def get_put_strike_from_delta(
    delta: float,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    maturity: date,
) -> float:
    return get_strike_from_delta(
        delta, spot, volatility, domestic_curve, foreign_curve, maturity, -1
    )


def get_mtm_option(
    instrument: Option, market: Market, currency: Currency | None = None
) -> float:
    """FX option mark-to-market (Garman-Kohlhagen) from market data.

    Curves are the funding curves of each currency of the pair; volatility is
    read from the pair's volatility surface at the option's log-moneyness. On or
    after the maturity date no volatility is read (:func:`get_option_mtm`).

    Parameters
    ----------
    instrument : Option
        FX option.
    market : Market
        Market at the valuation date.
    currency : Currency, optional
        Reporting currency; the quote currency when omitted. Another currency is
        converted at spot (``market.convert_value_currency``).

    Returns
    -------
    float
        MTM in `currency`.
    """
    pair = instrument.currency_pair
    domestic_curve = market.get_discount_curve(pair.quote_currency)
    foreign_curve = market.get_discount_curve(pair.base_currency)
    spot = market.get_fx_rate(market.t, pair).value

    volatility = float("nan")  # unused once expired
    if not _expired(instrument, market.t):
        log_moneyness = get_option_log_moneyness(
            instrument, spot, domestic_curve, foreign_curve
        )
        days = (instrument.maturity - market.t).days
        vol_surface = market.get_volatility_surface(pair)
        volatility = vol_surface.get_volatility(log_moneyness, days)

    mtm = get_option_mtm(
        instrument, market.t, spot, volatility, domestic_curve, foreign_curve
    )
    if currency is None or currency == pair.quote_currency:
        return mtm
    return market.convert_value_currency(mtm, pair.quote_currency, currency)
