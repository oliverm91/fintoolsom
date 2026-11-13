from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...market.currencies import CurrencyPair
    from ...market.index import InterestIndex


def _non_deliverable_currencies(currency_pair: CurrencyPair) -> list:
    return [
        ccy
        for ccy in (currency_pair.base_currency, currency_pair.quote_currency)
        if not ccy.is_deliverable
    ]


@dataclass
class Forward:
    """Deliverable FX forward: both currencies are exchanged at `payment_date`.

    Parameters
    ----------
    notional : float
        Amount of the base currency bought or sold.
    strike : float
        Agreed rate (quote currency per unit of base currency).
    payment_date : date
        Settlement date.
    is_buy : bool
        True when buying the base currency.
    currency_pair : CurrencyPair
        Traded pair.
    collateral_index : InterestIndex, optional
        CSA collateral index (keyword-only); None means uncollateralised. It
        changes how the flows are discounted, not the projected FX.

    Attributes
    ----------
    sign : int
        +1 for a buy, -1 for a sell.

    Raises
    ------
    ValueError
        If `currency_pair` is missing or holds a non-deliverable currency (the UF,
        ``CLF``): such a forward can only be an :class:`NDF`.
    """

    notional: float
    strike: float
    payment_date: date
    is_buy: bool
    currency_pair: CurrencyPair = None
    _: KW_ONLY
    collateral_index: InterestIndex | None = None

    sign: int = field(init=False)

    def __post_init__(self):
        self.sign = 1 if self.is_buy else -1
        if self.currency_pair is None:
            raise ValueError(f"currency_pair is required for a {type(self).__name__}.")
        self._check_settlement()

    def _check_settlement(self) -> None:
        undeliverable = _non_deliverable_currencies(self.currency_pair)
        if undeliverable:
            raise ValueError(
                f"{undeliverable[0]} cannot be delivered; a forward on "
                f"{self.currency_pair} must be an NDF."
            )


@dataclass
class NDF(Forward):
    """Non-deliverable FX forward, cash-settled at the rate of `fixing_date`.

    Pays ``notional * (FX(fixing_date) - strike)`` in the quote currency at
    `payment_date`. A UF forward is an NDF on the ``CLF/CLP`` pair, whose
    fixing is the UF level of `fixing_date`.

    Parameters
    ----------
    fixing_date : date
        Date whose FX rate settles the contract; not after `payment_date`.

    Raises
    ------
    ValueError
        If `fixing_date` is missing or after `payment_date`, or `currency_pair` is
        missing.
    """

    fixing_date: date = None

    def __post_init__(self):
        if self.fixing_date is None:
            raise ValueError("fixing_date is required for an NDF.")
        if self.fixing_date > self.payment_date:
            raise ValueError(
                f"fixing_date {self.fixing_date} is after payment_date {self.payment_date}."
            )
        super().__post_init__()

    def _check_settlement(self) -> None:
        # Cash settlement: any pair, including one quoted in a unit of account.
        pass
