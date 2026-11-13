"""CLF curve from UF forwards (CFNP) and UF/ICP swaps (CHSWC) on top of the CLP/USD set.

As of 2026-11-13, with the published UF (SII) up to 9-Dec-2026. CFNP settle on the
next 1..17 CPI releases, uncollateralised; CHSWC are CLF fixed vs ICP, semiannual,
SOFR-collateralised and settled in CLP, from 2Y. Quote levels are made up (flat
3.3% breakeven for the CFNP).

The CFNP read ICP past the end of the CLP-collateralised ICP swaps: the build must
extend ICP (SOFR-collateralised CLP swaps) before fitting them, or that extension
moves them afterwards.
"""

from datetime import date

import pytest

from fintoolsom.curve_builder import build_curves
from fintoolsom.curve_builder._keys import _instrument_of
from fintoolsom.dates import (
    ActualDayCountConvention,
    ModifiedFollowingConvention,
    TF_Actual360,
    last_known_uf_date,
)
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import (
    CrossCurrencyFixedFloatQuote,
    Currency,
    CurrencyPair,
    FixedLegSpec,
    FloatingLegSpec,
    PaymentFrequency,
    QuotedSide,
    UFForward,
    uf,
)
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.rates import LinearInterestConvention, Rate, RateConvention
from tests.factories.quote_factory import ICP, SOFR, load_quotes, quote_market

T = date(2026, 11, 13)
NOV9, DEC9 = date(2026, 11, 9), date(2026, 12, 9)
UF_T, UF_DEC9 = 41_322.93, 41_501.94
CLFCLP = CurrencyPair(Currency.CLF, Currency.CLP)
CL = ModifiedFollowingConvention("CL")
LIN360 = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)
CHSWC_RATES = {
    2: 0.0205,
    3: 0.0210,
    4: 0.0215,
    5: 0.0220,
    6: 0.0223,
    7: 0.0226,
    8: 0.0229,
    9: 0.0232,
    10: 0.0235,
    12: 0.0240,
    15: 0.0245,
    20: 0.0250,
}


def _uf_levels() -> dict[date, float]:
    """Published UF from 9-Nov to 9-Dec, from the 13-Nov and 9-Dec levels."""
    cpi = (UF_DEC9 / UF_T) ** ((DEC9 - NOV9).days / (DEC9 - T).days) - 1
    levels = uf.extend_with_cpi(
        {NOV9: round(UF_DEC9 / (1 + cpi), 2), DEC9: UF_DEC9}, cpi, base_date=NOV9
    )
    levels[T] = UF_T
    return levels


def _cfnp() -> list[UFForward]:
    out = []
    for n in range(1, 18):
        settle = UFForward(0.0, True, n, quote_date=T).settlement_date
        level = round(UF_DEC9 * 1.033 ** ((settle - DEC9).days / 365), 2)
        out.append(UFForward(level, True, n, quote_date=T))
    return out


def _chswc() -> list[CrossCurrencyFixedFloatQuote]:
    def spec(ccy):
        return (ccy, PaymentFrequency.SEMIANNUAL, CL, TF_Actual360())

    return [
        CrossCurrencyFixedFloatQuote(
            QuotedSide.RECEIVE,
            Term(years, TermUnit.Y, CL),
            CL,
            quote_date=T,
            collateral_index=SOFR,
            fixed_leg=FixedLegSpec(*spec(Currency.CLF), Rate(LIN360, rate)),
            floating_leg=FloatingLegSpec(*spec(Currency.CLP), ICP, Rate(LIN360, 0.0)),
            is_deliverable=False,
            payment_currency=Currency.CLP,
        )
        for years, rate in CHSWC_RATES.items()
    ]


def quote_mtm(quote, market) -> float:
    """MTM in CLP of the instrument the builder fits for `quote`."""
    return Calculator.valuate(_instrument_of(quote, market), market, Currency.CLP)


@pytest.fixture(scope="module")
def built():
    market = quote_market(T)
    for d, fx in uf.uf_rate_data(_uf_levels()).historic_data.items():
        market.add_fx_rate(d, fx)
    assert market.get_fx_known_until(CLFCLP) == last_known_uf_date(T) == DEC9
    cfnp, chswc = _cfnp(), _chswc()
    with pytest.warns(Warning):  # RedundantQuoteWarning from the base CLP/USD set
        build_curves(load_quotes(T) + cfnp + chswc, market)
    return market, cfnp, chswc


def test_builds_a_clf_curve(built):
    market, _, _ = built
    assert {"CLF_ICP", "ICP", "SOFR", "USD_ICP"} <= set(market.curves)


def test_uf_forwards_reprice_and_match_the_projected_uf(built):
    market, cfnp, _ = built
    for q in cfnp:
        assert quote_mtm(q, market) == pytest.approx(0.0, abs=1e-6)
        projected = market.get_future_fx_rate(q.settlement_date, CLFCLP).value
        assert projected == pytest.approx(q.value, rel=1e-12)


def test_uf_icp_swaps_reprice(built):
    market, _, chswc = built
    for q in chswc:
        assert quote_mtm(q, market) == pytest.approx(0.0, abs=1e-6)
