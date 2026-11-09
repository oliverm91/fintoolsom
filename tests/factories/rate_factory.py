"""Rates and rate conventions used across the tests."""

from fintoolsom.dates import ActualDayCountConvention, Days30EDayCountConvention
from fintoolsom.rates import (
    CompoundedInterestConvention,
    ExponentialInterestConvention,
    LinearInterestConvention,
    Rate,
    RateConvention,
)


def lin_act360() -> RateConvention:
    """Linear act/360 (money-market, ICP/SOFR accruals)."""
    return RateConvention(LinearInterestConvention, ActualDayCountConvention, 360)


def lin_30e360() -> RateConvention:
    """Linear 30E/360 (Chilean bond coupon accrual)."""
    return RateConvention(LinearInterestConvention, Days30EDayCountConvention, 360)


def comp_act365() -> RateConvention:
    """Annually compounded act/365 (Chilean bond IRR / TERA)."""
    return RateConvention(CompoundedInterestConvention, ActualDayCountConvention, 365)


def exp_act365() -> RateConvention:
    """Continuously compounded act/365."""
    return RateConvention(ExponentialInterestConvention, ActualDayCountConvention, 365)


def rate(value: float, convention: RateConvention | None = None) -> Rate:
    """`value` in `convention` (annually compounded act/365 by default)."""
    return Rate(convention if convention is not None else comp_act365(), value)
