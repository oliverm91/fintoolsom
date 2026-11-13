"""Valuation of derivative instruments.

The functions live in topic modules (``_fx``, ``_forwards``, ``_options``,
``_swap_pv``, ``_swap_risk``); :class:`Calculator` exposes them as staticmethods
so ``from fintoolsom.derivatives.calculator import Calculator`` keeps working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..forwards.forwards import NDF, Forward
from ..options.options import Option
from ..swaps.swaps import Swap
from . import _forwards, _fx, _options, _swap_pv, _swap_risk

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency

__all__ = ["Calculator"]


class Calculator:
    """Valuation of derivative instruments.

    Instruments only hold data. Every method is a staticmethod taking the
    instrument plus the market data it needs; nothing is cached or stored.
    """

    # --- FX ---
    _convert = staticmethod(_fx._convert)
    get_fx_fixings = staticmethod(_fx.get_fx_fixings)
    _fx_fixings_at_ordinals = staticmethod(_fx._fx_fixings_at_ordinals)

    # --- Forwards ---
    get_forward_mtm = staticmethod(_forwards.get_forward_mtm)
    get_ndf_mtm = staticmethod(_forwards.get_ndf_mtm)

    # --- Options (Garman-Kohlhagen / Black-Scholes for FX) ---
    get_option_log_moneyness = staticmethod(_options.get_option_log_moneyness)
    _get_option_yf = staticmethod(_options._get_option_yf)
    _get_option_rates = staticmethod(_options._get_option_rates)
    _get_option_d1 = staticmethod(_options._get_option_d1)
    _get_option_d2 = staticmethod(_options._get_option_d2)
    get_option_mtm = staticmethod(_options.get_option_mtm)
    get_option_delta = staticmethod(_options.get_option_delta)
    get_option_gamma = staticmethod(_options.get_option_gamma)
    get_option_vega = staticmethod(_options.get_option_vega)
    get_strike_from_delta = staticmethod(_options.get_strike_from_delta)
    get_call_strike_from_delta = staticmethod(_options.get_call_strike_from_delta)
    get_put_strike_from_delta = staticmethod(_options.get_put_strike_from_delta)
    get_mtm_option = staticmethod(_options.get_mtm_option)

    # --- Swaps ---
    _leg_flows = staticmethod(_swap_pv._leg_flows)
    _index_history = staticmethod(_swap_pv._index_history)
    _leg_pv = staticmethod(_swap_pv._leg_pv)
    _leg_pv_non_deliverable = staticmethod(_swap_pv._leg_pv_non_deliverable)
    get_swap_mtm = staticmethod(_swap_pv.get_swap_mtm)
    get_leg_pv = staticmethod(_swap_pv.get_leg_pv)
    _rate_sensitive_flows = staticmethod(_swap_risk._rate_sensitive_flows)
    get_leg_duration = staticmethod(_swap_risk.get_leg_duration)
    get_leg_dv01 = staticmethod(_swap_risk.get_leg_dv01)

    # --- Dispatcher ---

    @staticmethod
    def valuate(
        instrument: Forward | NDF | Option | Swap, market: Market, currency: Currency
    ) -> float:
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
            return _forwards.get_ndf_mtm(instrument, market, currency)
        if isinstance(instrument, Forward):
            return _forwards.get_forward_mtm(instrument, market, currency)
        if isinstance(instrument, Swap):
            return _swap_pv.get_swap_mtm(instrument, market, currency)
        if isinstance(instrument, Option):
            return _options.get_mtm_option(instrument, market)

        raise NotImplementedError(
            f"No valuation routing implemented for instrument type {type(instrument).__name__}."
        )
