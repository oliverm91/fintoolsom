"""Builds a fintoolsom ``Swap`` from a swap booked in the ``derivatives`` schema (``swaps`` and ``swap_legs``).

Supported: single-currency swaps with one fixed leg and one floating leg on ICP or SOFR. DB conventions assumed:

* ``fixed_rate`` is a percent (4.92 = 4.92%), lin/act/360.
* ``spread_basis`` is in bp (all the booked swaps have 0).
* ``collateral = 1`` means SOFR collateral, ``0`` means uncollateralised.
* ``payment_frequency``: ``zero`` (one coupon), ``monthly``, ``quarterly``, ``semiannually``, ``annually``.
"""
from dataclasses import dataclass
from datetime import date

import pandas as pd

from fintoolsom.derivatives.swaps import Swap, builders
from fintoolsom.market import Currency, Term, TermUnit

from build_quotes import CL_ADJ, ICP, SOFR, rate  # also enables the zero-coupon frequency used by ``zero`` legs
from local_db import read_sql

FREQUENCIES = {"zero": "ZERO", "monthly": "MONTHLY", "quarterly": "QUARTERLY", "semiannually": "SEMIANNUAL", "annually": "ANNUAL"}
INDEXES = {"icp": ICP, "sofr": SOFR}


@dataclass
class BookedSwap:
    swap: Swap
    receive_leg_id: str
    pay_leg_id: str
    start_date: date


COLLATERAL_CHOICES = (1, 0)  # same flag as derivatives.swaps.collateral: 1 = SOFR collateralised, 0 = uncollateralised


def get_swap(swap_id: int, collateral: int | None = None) -> BookedSwap:
    """Swap ``swap_id`` as booked. ``collateral`` overrides its CSA flag (1 = SOFR, 0 = none); None keeps the booked one."""
    if collateral is not None and collateral not in COLLATERAL_CHOICES:
        raise ValueError(f"collateral must be one of {COLLATERAL_CHOICES} or None, got '{collateral}'.")
    header = read_sql("SELECT * FROM swaps WHERE id=:id", id=swap_id)
    legs = read_sql("SELECT * FROM swap_legs WHERE fk_swap=:id", id=swap_id)
    if header.empty or len(legs) != 2:
        raise ValueError(f"Swap {swap_id} not found in the local snapshot or without exactly two legs.")
    swap_row = header.iloc[0]
    start, maturity = pd.to_datetime(swap_row.start_date).date(), pd.to_datetime(swap_row.maturity_date).date()
    term = Term((maturity.year - start.year) * 12 + maturity.month - start.month, TermUnit.M, CL_ADJ)
    collateral_flag = int(swap_row.collateral) if collateral is None else collateral
    collateral_index = SOFR if collateral_flag == 1 else None
    payment_currency = getattr(Currency, swap_row.comp_currency.upper())

    built = {}
    for leg in legs.itertuples():
        if leg.payment_frequency not in FREQUENCIES:
            raise ValueError(f"Swap {swap_id}: unsupported payment frequency '{leg.payment_frequency}'.")
        common = dict(notional=float(leg.notional), start_date=start, term=term, frequency=FREQUENCIES[leg.payment_frequency],
                      adj_convention=CL_ADJ, maturity_date=maturity)
        if pd.notna(leg.fixed_rate):
            built[leg.side] = (leg.id, builders.fixed_leg(rate=rate(float(leg.fixed_rate) / 100),
                                                          currency=getattr(Currency, leg.currency.upper()), **common))
        else:
            built[leg.side] = (leg.id, builders.floating_leg(index=INDEXES[leg.float_index.lower()],
                                                             spread=rate(float(leg.spread_basis) / 1e4), **common))
    (receive_id, receive_leg), (pay_id, pay_leg) = built["receive"], built["pay"]
    swap = Swap(receive_leg=receive_leg, pay_leg=pay_leg, collateral_index=collateral_index, payment_currency=payment_currency)
    return BookedSwap(swap, receive_id, pay_id, start)
