import math
from datetime import date, timedelta
import pytest
from fintoolsom.derivatives.forwards.forwards import NDF
from fintoolsom.market import Currency, Market, OvernightRateIndex, uf
from fintoolsom.market.currencies import CurrencyPair
from fintoolsom.market.quotes import ForwardPointsQuote, ForwardPriceQuote
from fintoolsom.rates import DiscountCurve, IndexCurve
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
    pmt = date(2025, 1, 10)
    with pytest.raises(ValueError, match="cannot be delivered.*NDF"):
        ForwardPriceQuote(pair, 38_150.25, True, quote_date=T, payment_date=pmt)
    with pytest.raises(ValueError, match="cannot be delivered.*NDF"):
        ForwardPointsQuote(
            currency_pair=pair, value=1.0, is_buy=True, payment_date=pmt, quote_date=T
        )
