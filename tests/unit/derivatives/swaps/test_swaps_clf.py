from datetime import date
import pytest
from fintoolsom.derivatives.swaps import builders
from fintoolsom.derivatives.swaps.swaps import Swap
from fintoolsom.dates import Calendar
from fintoolsom.dates.adjustments import ModifiedFollowingConvention
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.market import Currency
from fintoolsom.rates import Rate, RateConvention, LinearInterestConvention
from fintoolsom.dates import ActualDayCountConvention


CLF, CLP, USD = Currency.CLF, Currency.CLP, Currency.USD


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


def test_a_swap_cannot_settle_in_clf():
    leg = _fixed(CLF, 1_000.0, 0.02)
    with pytest.raises(ValueError, match="cannot be delivered"):
        Swap(receive_leg=leg, pay_leg=_fixed(CLF, 1_000.0, 0.01), payment_currency=CLF)


@pytest.mark.parametrize("other", [CLP, USD, CLF], ids=["clp", "usd", "clf"])
@pytest.mark.parametrize("uf_side", ["receive", "pay"])
def test_a_deliverable_swap_with_a_clf_leg_is_rejected(other, uf_side):
    uf, other_leg = _fixed(CLF, 1_000.0, 0.02), _fixed(other, 1_000.0, 0.05)
    legs = (uf, other_leg) if uf_side == "receive" else (other_leg, uf)
    with pytest.raises(ValueError, match="CLF cannot be delivered.*non-deliverable"):
        Swap(receive_leg=legs[0], pay_leg=legs[1], is_deliverable=True)


@pytest.mark.parametrize("other", [CLP, USD])
def test_a_clf_swap_settled_in_clp_is_accepted(other):
    uf = _fixed(CLF, 1_000.0, 0.02)
    swap = Swap(
        receive_leg=uf, pay_leg=_fixed(other, 1_000.0, 0.05), payment_currency=CLP
    )
    assert swap.leg_settlement_currency(uf) is CLP
