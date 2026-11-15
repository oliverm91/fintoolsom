"""FX forwards, NDFs and options on the C++ backend (``fintoolsom._native``).

Each function mirrors one in :mod:`._forwards` or :mod:`._options` and gives the same
result to the bit. Python keeps what reads the market, in the same order so a missing
entry raises the same error: spot and FX fixings, curves, the volatility surface and
the rate conventions (``r`` and ``q`` from the discount factors). C++ does the
arithmetic: discount factors (straight from a curve's C++ core when it has one), the
projected settlement FX, Garman-Kohlhagen prices and greeks, and the strike from a
delta, with ``ndtr`` / ``ndtri`` computed as :mod:`._normal` does.

A forward or option uses this module when it was created under the ``"cpp"`` backend
of ``"derivatives"`` (:mod:`fintoolsom.settings`); :func:`strike_from_delta` when that
is the backend at the call.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from ... import settings
from ...market.currencies import FX_Rate, CurrencyPair
from . import _options as _py
from ._native_swap_pv import _discount_curves

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency
    from ...rates import Curve, RateConvention
    from ..forwards.forwards import NDF, Forward
    from ..options.options import Option


def _abc(curves: tuple) -> tuple:
    """:func:`._native_swap_pv._discount_curves` padded to ``(a, b, c)``."""
    return curves if len(curves) == 3 else (curves[0], None, None)


def _to_currency(market: Market, from_currency: Currency, currency: Currency) -> float:
    """The spot rate of :meth:`Market.convert_value_currency`."""
    return market.get_fx_rate(market.t, CurrencyPair(from_currency, currency)).value


def forward_mtm(forward: Forward, market: Market, currency: Currency) -> float:
    """:func:`._forwards.get_forward_mtm`; on a C++ market, in one C++ call."""
    if market._native:
        return settings.native().market_forward_mtm(forward, market, currency)
    cp = forward.currency_pair
    if forward.payment_date <= market.t:
        return 0.0
    collateral = forward.collateral_index
    spot = market.get_fx_rate(market.t, cp).value
    quote = _abc(_discount_curves(market, cp.quote_currency, collateral))
    base = _abc(_discount_curves(market, cp.base_currency, collateral))
    return settings.native().forward_mtm(
        forward.sign,
        forward.notional,
        forward.strike,
        spot,
        forward.payment_date.toordinal(),
        *quote,
        *base,
        _to_currency(market, cp.quote_currency, currency),
    )


def _settlement_fx(ndf: NDF, market: Market) -> float:
    """The settlement FX of :func:`._forwards.get_ndf_mtm`: realised or published (from
    Python), else projected on the funding curves in C++ as
    :meth:`Market._future_fx_rates_at_ordinals` does."""
    cp = ndf.currency_pair
    if ndf.fixing_date <= market.t:
        return market.get_fx_rate(ndf.fixing_date, cp).value
    known_until = market.get_fx_known_until(cp)
    if ndf.fixing_date <= known_until or cp.base_currency == cp.quote_currency:
        return market.get_future_fx_rate(ndf.fixing_date, cp).value
    anchor = market.get_fx_rate(known_until, cp).value
    rate = settings.native().projected_fx(
        anchor,
        known_until.toordinal(),
        ndf.fixing_date.toordinal(),
        market.get_discount_curve(cp.base_currency),
        market.get_discount_curve(cp.quote_currency),
    )
    return FX_Rate(cp, rate).value


def ndf_mtm(ndf: NDF, market: Market, currency: Currency) -> float:
    """:func:`._forwards.get_ndf_mtm`; on a C++ market, in one C++ call."""
    if market._native:
        return settings.native().market_ndf_mtm(ndf, market, currency)
    cp = ndf.currency_pair
    if ndf.payment_date <= market.t:
        return 0.0
    s_t = _settlement_fx(ndf, market)
    quote = _abc(_discount_curves(market, cp.quote_currency, ndf.collateral_index))
    return settings.native().ndf_mtm(
        ndf.sign,
        ndf.notional,
        ndf.strike,
        s_t,
        ndf.payment_date.toordinal(),
        *quote,
        _to_currency(market, cp.quote_currency, currency),
    )


def option_log_moneyness(
    option: Option, spot: float, domestic_curve: Curve, foreign_curve: Curve
) -> float:
    """:func:`._options.get_option_log_moneyness`."""
    df_r = domestic_curve.get_df(option.maturity)
    df_q = foreign_curve.get_df(option.maturity)
    return settings.native().option_log_moneyness(option.strike, spot, df_r, df_q)


def _model_inputs(
    option: Option,
    t: date,
    domestic_curve: Curve,
    foreign_curve: Curve,
    rate_convention: RateConvention | None,
) -> tuple[float, float, float]:
    """``(yf, r, q)`` as the Python greeks read them, once."""
    yf = _py._get_option_yf(option, t)
    r, q = _py._get_option_rates(
        option, t, domestic_curve, foreign_curve, rate_convention
    )
    return yf, r, q


def option_mtm(option, t, spot, volatility, domestic_curve, foreign_curve, rc):
    """:func:`._options.get_option_mtm`."""
    yf, r, q = _model_inputs(option, t, domestic_curve, foreign_curve, rc)
    return settings.native().option_mtm(
        option._sign, option.notional, option.strike, spot, volatility, yf, r, q
    )


def option_delta(option, t, spot, volatility, domestic_curve, foreign_curve, rc):
    """:func:`._options.get_option_delta`."""
    yf, r, q = _model_inputs(option, t, domestic_curve, foreign_curve, rc)
    return settings.native().option_delta(
        option._sign, option.notional, option.strike, spot, volatility, yf, r, q
    )


def option_gamma(option, t, spot, volatility, domestic_curve, foreign_curve, rc):
    """:func:`._options.get_option_gamma`."""
    yf, r, q = _model_inputs(option, t, domestic_curve, foreign_curve, rc)
    return settings.native().option_gamma(
        option.notional, option.strike, spot, volatility, yf, r, q
    )


def option_vega(option, t, spot, volatility, domestic_curve, foreign_curve, rc):
    """:func:`._options.get_option_vega`."""
    yf, r, q = _model_inputs(option, t, domestic_curve, foreign_curve, rc)
    return settings.native().option_vega(
        option.notional, option.strike, spot, volatility, yf, r, q
    )


def strike_from_delta(
    delta: float,
    spot: float,
    volatility: float,
    domestic_curve: Curve,
    foreign_curve: Curve,
    maturity: date,
    sign: int,
    rate_convention: RateConvention | None,
) -> float:
    """:func:`._options.get_strike_from_delta`."""
    rc = rate_convention or _py._default_option_rate_convention
    df_r = domestic_curve.get_df(maturity)
    t = domestic_curve.curve_date
    yf = rc.day_count_convention.get_time_fraction(t, maturity, rc.time_fraction_base)
    r = rc.interest_convention.get_rate_from_df(df_r, yf)
    df_q = foreign_curve.get_df(maturity)
    q = rc.interest_convention.get_rate_from_df(df_q, yf)
    return settings.native().strike_from_delta(
        delta, spot, volatility, yf, r, q, df_q, sign
    )
