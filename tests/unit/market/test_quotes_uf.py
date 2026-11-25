import math
from datetime import date, timedelta
import pytest
from fintoolsom.derivatives.forwards.forwards import NDF
from fintoolsom.market import Currency, Market, OvernightRateIndex, uf
from fintoolsom.market.currencies import CurrencyPair
from fintoolsom.market.quotes import ForwardPointsQuote, ForwardPriceQuote
from fintoolsom.rates import PillarCurve, InterpolationMethod
from tests.factories.curve_factory import curve_from_dfs


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


def _flat_curve(rate: float, currency: Currency | None = None) -> PillarCurve:
    """Flat continuous zero rate: the risk-free index curve, or `currency`'s discount curve."""
    if currency is None:
        return PillarCurve(
            T,
            [T, T + timedelta(days=30), T + timedelta(days=1825)],
            [rate, rate],
            InterpolationMethod.PiecewiseConstant,
        )
    pillars = [T + timedelta(days=d) for d in (1, 7, 30, 91, 365, 730, 1825)]
    return curve_from_dfs(
        T,
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
        curves={RISKLESS: _flat_curve(R_CLP), CLF: _flat_curve(R_CLF, CLF)},
        fx_history={CLFCLP: uf.uf_rate_data(levels)},
    )


def test_uf_forward_quotes_build_ndfs_when_a_fixing_date_is_given():
    pmt = date(2025, 1, 10)
    price = ForwardPriceQuote(
        CLFCLP, 38_150.25, True, quote_date=T, payment_date=pmt, fixing_date=pmt
    ).get_instrument()
    assert (
        isinstance(price, NDF)
        and price.fixing_date == pmt
        and price.strike == 38_150.25
    )

    points = ForwardPointsQuote(
        currency_pair=CLFCLP,
        value=15_025.0,
        is_buy=False,
        payment_date=pmt,
        fixing_date=pmt,
        quote_date=T,
        points_divisor=100,
    ).get_instrument(_market())
    assert isinstance(points, NDF) and points.currency_pair == CLFCLP
    assert points.strike == pytest.approx(UF_T + 150.25, rel=1e-12)


@pytest.mark.parametrize("pair", [CLFCLP, CLFCLP.invert()], ids=["clf-clp", "clp-clf"])
def test_non_ndf_forward_quote_with_clf_is_rejected(pair):
    # The check is _ForwardQuote's, shared by price and points quotes.
    with pytest.raises(ValueError, match="cannot be delivered.*NDF"):
        ForwardPriceQuote(
            pair, 38_150.25, True, quote_date=T, payment_date=date(2025, 1, 10)
        )


def test_uf_irs_and_basis_quotes_build_swaps_settling_in_clp():
    from fintoolsom.dates import ModifiedFollowingConvention, Term, TermUnit
    from fintoolsom.market import (
        FixedLegSpec,
        FloatingLegSpec,
        IRBasisQuote,
        IRSQuote,
        OvernightInterestPriceIndex,
        PaymentFrequency,
        QuotedSide,
    )
    from fintoolsom.rates import Rate, RateConvention

    cl = ModifiedFollowingConvention("CL")
    uf_idx_a = OvernightInterestPriceIndex("ICP_UF_A", currency=CLF)
    uf_idx_b = OvernightInterestPriceIndex("ICP_UF_B", currency=CLF)

    for side in (QuotedSide.RECEIVE, QuotedSide.PAY):
        irs = IRSQuote(
            quoted_side=side,
            term=Term(2, TermUnit.Y, cl),
            adj_convention=cl,
            quote_date=T,
            collateral_index=RISKLESS,
            fixed_leg=FixedLegSpec(
                CLF, PaymentFrequency.SEMIANNUAL, cl, Rate(RateConvention(), 0.02)
            ),
            floating_leg=FloatingLegSpec(
                CLF,
                PaymentFrequency.SEMIANNUAL,
                cl,
                uf_idx_a,
                Rate(RateConvention(), 0.0),
            ),
        ).get_instrument()
        assert irs.payment_currency == CLP

    basis = IRBasisQuote(
        quoted_side=QuotedSide.RECEIVE,
        term=Term(2, TermUnit.Y, cl),
        adj_convention=cl,
        quote_date=T,
        collateral_index=RISKLESS,
        receive_leg=FloatingLegSpec(
            CLF, PaymentFrequency.SEMIANNUAL, cl, uf_idx_a, Rate(RateConvention(), 0.0)
        ),
        pay_leg=FloatingLegSpec(
            CLF,
            PaymentFrequency.SEMIANNUAL,
            cl,
            uf_idx_b,
            Rate(RateConvention(), 0.001),
        ),
    ).get_instrument()
    assert basis.payment_currency == CLP
