import math
from datetime import date, timedelta
import pytest
from fintoolsom.derivatives.calculator import Calculator
from fintoolsom.derivatives.forwards.forwards import NDF
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.dates import Calendar
from fintoolsom.dates.adjustments import ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency, Market, OvernightRateIndex, uf
from fintoolsom.market.currencies import CurrencyPair, FX_Rate
from fintoolsom.rates import (
    DiscountCurve,
    IndexCurve,
    Rate,
    RateConvention,
    LinearInterestConvention,
)
from fintoolsom.dates import ActualDayCountConvention
from tests.factories.curve_factory import discount_curve_from_dfs


CLF, CLP, USD = Currency.CLF, Currency.CLP, Currency.USD


CLFCLP = CurrencyPair(CLF, CLP)


RISKLESS = OvernightRateIndex("OIS", currency=CLP)


# Before the 8th (CPI release): the UF is published up to the 9th of this month.
T = date(2024, 8, 4)


KNOWN_UNTIL = date(2024, 8, 9)  # T + 5


R_CLP, R_CLF = 0.06, 0.02


UF_T = 37_500.0


def uf_level(d: date) -> float:
    return UF_T * 1.0001 ** (d - T).days


def _flat_curve(
    rate: float, currency: Currency | None = None
) -> IndexCurve | DiscountCurve:
    """Flat continuous zero rate: the risk-free index curve, or `currency`'s discount curve."""
    if currency is None:
        return IndexCurve(
            T,
            RISKLESS,
            [T, T + timedelta(days=30), T + timedelta(days=1825)],
            [rate, rate],
        )
    pillars = [T + timedelta(days=d) for d in (1, 7, 30, 91, 365, 730, 1825)]
    return discount_curve_from_dfs(
        T,
        currency,
        RISKLESS,
        pillars,
        [math.exp(-rate * (d - T).days / 365) for d in pillars],
    )


def _market() -> Market:
    # Load only what is published as of T: up to KNOWN_UNTIL.
    assert uf.last_published_date(T) == KNOWN_UNTIL
    levels = {
        T + timedelta(days=i): uf_level(T + timedelta(days=i)) for i in range(-60, 6)
    }
    return Market(
        t=T,
        risk_free_index=RISKLESS,
        curves=[_flat_curve(R_CLP), _flat_curve(R_CLF, CLF)],
        fx_history={CLFCLP: uf.uf_rate_data(levels)},
    )


CLP_CURVE, CLF_CURVE = _flat_curve(R_CLP), _flat_curve(R_CLF, CLF)


def _projected(d: date) -> float:
    """UF(L) * [df_CLF(d)/df_CLF(L)] / [df_CLP(d)/df_CLP(L)]."""
    L = KNOWN_UNTIL
    return (
        uf_level(L)
        * (CLF_CURVE.get_df(d) / CLF_CURVE.get_df(L))
        / (CLP_CURVE.get_df(d) / CLP_CURVE.get_df(L))
    )


def test_uf_fixings_inside_the_known_window_are_read_and_later_ones_projected():
    """Payments at T+3 and T+7 with the UF known until T+5."""
    market = _market()
    d3, d7 = T + timedelta(days=3), T + timedelta(days=7)
    fx = Calculator.get_fx_fixings(CLF, CLP, [d3, d7], market)
    assert fx[0] == uf_level(d3)  # published: no curve involved
    assert fx[1] == pytest.approx(_projected(d7), rel=1e-12)
    # Anchored on the last published level, not on spot.
    spot_projection = UF_T * CLF_CURVE.get_df(d7) / CLP_CURVE.get_df(d7)
    assert fx[1] != pytest.approx(spot_projection, rel=1e-9)


def test_traded_pair_projection_still_starts_from_spot():
    market = _market()
    market.set_curve(_flat_curve(0.05, USD))
    market.add_fx_rate(T, FX_Rate(CurrencyPair(USD, CLP), 950.0))
    d = T + timedelta(days=30)
    fx = Calculator.get_fx_fixings(USD, CLP, [d], market)[0]
    usd = market.get_discount_curve(USD)
    assert fx == pytest.approx(950.0 * usd.get_df(d) / CLP_CURVE.get_df(d), rel=1e-12)


@pytest.mark.parametrize(
    "fixing",
    [T + timedelta(days=3), T + timedelta(days=40)],
    ids=["known", "projected"],
)
def test_uf_ndf_settles_on_the_uf_of_its_fixing(fixing):
    market = _market()
    payment = fixing + timedelta(days=2)
    ndf = NDF(
        1_000, 38_000.0, payment, is_buy=True, currency_pair=CLFCLP, fixing_date=fixing
    )
    uf = uf_level(fixing) if fixing <= KNOWN_UNTIL else _projected(fixing)
    expected = 1_000 * (uf - 38_000.0) * CLP_CURVE.get_df(payment)
    assert Calculator.valuate(ndf, market, CLP) == pytest.approx(expected, rel=1e-12)


ADJ = ModifiedFollowingConvention(Calendar(country=CLF.locality.value))


LIN = RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)


def _fixed(ccy: Currency, notional: float, rate: float):
    return builders.fixed_leg(
        notional=notional,
        start_date=date(2024, 8, 6),
        term=Term(1, TermUnit.Y, ADJ),
        frequency="QUARTERLY",
        adj_convention=ADJ,
        rate=Rate(LIN, rate),
        currency=ccy,
    )


def test_uf_leg_settled_in_clp_is_paid_at_each_payment_uf():
    market = _market()
    uf_leg, clp_leg = _fixed(CLF, 1_000.0, 0.02), _fixed(CLP, 37_500_000.0, 0.06)
    swap = Swap(receive_leg=uf_leg, pay_leg=clp_leg, payment_currency=CLP)
    assert swap.leg_settlement_currency(uf_leg) is CLP
    assert uf_leg.fx_fixing_dates == uf_leg.payment_dates

    _, flows = Calculator._leg_flows(uf_leg, market)
    fx = [_projected(d) for d in uf_leg.payment_dates]  # all after the 9th
    dfs = [CLP_CURVE.get_df(d) for d in uf_leg.payment_dates]
    expected_uf = sum(f * x * df for f, x, df in zip(flows, fx, dfs))
    expected = expected_uf - Calculator._leg_pv(clp_leg, None, market)
    assert swap.valuate(market, CLP) == pytest.approx(expected, rel=1e-10)
