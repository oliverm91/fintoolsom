from datetime import date

import pytest

from fintoolsom.dates import last_known_uf_date
from fintoolsom.derivatives.forwards.forwards import NDF
from fintoolsom.market import Currency, CurrencyPair, UFForward

CLFCLP = CurrencyPair(Currency.CLF, Currency.CLP)


@pytest.mark.parametrize(
    "quote_date, n, settlement",
    [
        # Before the 6-Nov-2026 release the UF is known up to 9-Nov: n=1 is 9-Dec.
        (date(2026, 11, 5), 1, date(2026, 12, 9)),  # business day, not adjusted
        # On the release day the UF is known up to 9-Dec: n=1 is 9-Jan-2027 (Sat).
        (date(2026, 11, 6), 1, date(2027, 1, 8)),
        (date(2026, 11, 13), 1, date(2027, 1, 8)),
        (date(2026, 11, 13), 12, date(2027, 12, 9)),
        (date(2026, 11, 13), 24, date(2028, 12, 7)),
    ],
)
def test_settles_on_the_nth_cpi_release(quote_date, n, settlement):
    quote = UFForward(38_000.0, True, n, quote_date=quote_date)
    assert quote.settlement_date == settlement
    assert settlement > last_known_uf_date(quote_date)  # its UF is not yet known


def test_builds_a_clfclp_ndf_with_no_payment_lag():
    quote = UFForward(38_150.25, False, 3, quote_date=date(2026, 11, 13))
    ndf = quote.get_instrument()
    assert isinstance(ndf, NDF)
    assert ndf.currency_pair == CLFCLP == quote.currency_pair
    assert ndf.fixing_date == ndf.payment_date == quote.settlement_date
    assert ndf.strike == 38_150.25 and not ndf.is_buy and ndf.notional == 100.0


@pytest.mark.parametrize("n", [0, 25, -1, 1.0, True, "1"])
def test_rejects_out_of_range_settlements(n):
    with pytest.raises(ValueError, match="next_n_cpi_settlements"):
        UFForward(38_000.0, True, n, quote_date=date(2026, 11, 13))


@pytest.mark.parametrize(
    "kwarg", ["term", "payment_date", "fixing_date", "currency_pair"]
)
def test_takes_no_tenor_dates_or_pair(kwarg):
    with pytest.raises(TypeError):
        UFForward(38_000.0, True, 1, quote_date=date(2026, 11, 13), **{kwarg: None})
