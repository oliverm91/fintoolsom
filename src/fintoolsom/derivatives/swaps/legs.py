from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import cast

import numpy as np

from fintoolsom.rates.Rates import Rate, RateConvention

from .coupons import FixedCoupon, OvernightCoupon, TermRateCoupon, XCCYCoupon
from ...dates import AdjustmentDateConventionBase, ScheduleGenerator
from ...dates.date_counts import date_ordinals
from ...dates.term import Term
from ...market.currencies import Currency
from ...market.index import Index, InterestIndex


# Maps PaymentFrequency enum value to ScheduleGenerator frequency string.
_FREQ_TO_TENOR: dict[str, str] = {
    "MONTHLY": "1m",
    "QUARTERLY": "3m",
    "SEMIANNUAL": "6m",
    "ANNUAL": "1y",
}


def _zero_spread() -> Rate:
    """A fresh zero spread (never share one mutable Rate between legs)."""
    return Rate(RateConvention(), 0.0)


@dataclass
class SwapLeg:
    """Base swap leg: a list of coupons plus cached per-coupon arrays.

    Parameters
    ----------
    coupons : list of SwapCoupon
        The leg's coupons, in schedule order.

    Attributes
    ----------
    residuals, amortizations : numpy.ndarray
        Per-coupon outstanding notional and notional repayment.
    start_dates, end_dates, payment_dates : list of date
        Per-coupon accrual start, accrual end and payment date.
    fx_fixing_dates : list of date
        Per-coupon FX fixing date, used when the leg settles non-deliverably in
        another currency.
    """

    coupons: list

    residuals: np.ndarray = field(init=False)
    amortizations: np.ndarray = field(init=False)
    start_dates: list[date] = field(init=False)
    end_dates: list[date] = field(init=False)
    payment_dates: list[date] = field(init=False)
    fx_fixing_dates: list[date] = field(init=False)

    # The dates above as integer ordinals, so a valuation needs no date conversion.
    _payment_ordinals: np.ndarray = field(init=False, repr=False)
    _start_ordinals: np.ndarray = field(init=False, repr=False)
    _end_ordinals: np.ndarray = field(init=False, repr=False)
    _fx_fixing_ordinals: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.residuals = np.array([c.residual for c in self.coupons])
        self.amortizations = np.array([c.amortization for c in self.coupons])
        self.start_dates = [c.start_date for c in self.coupons]
        self.end_dates = [c.end_date for c in self.coupons]
        self.payment_dates = [c.payment_date for c in self.coupons]
        self.fx_fixing_dates = [c.fx_fixing_date for c in self.coupons]
        self._payment_ordinals = date_ordinals(self.payment_dates)
        self._start_ordinals = date_ordinals(self.start_dates)
        self._end_ordinals = date_ordinals(self.end_dates)
        self._fx_fixing_ordinals = date_ordinals(self.fx_fixing_dates)

    @property
    def n_coupons(self) -> int:
        """Number of coupons."""
        return len(self.coupons)

    _label = "Leg"

    def _terms(self) -> str:
        """Rate description used by :meth:`__str__`; empty for the base leg."""
        return ""

    def __str__(self) -> str:
        ccy = getattr(self, "currency", None)
        head = f"{self._label:<15}{ccy if ccy is not None else '?'}"
        if not self.coupons:
            return f"{head}  (no coupons)"
        notional = f"{self.residuals[0]:,.0f}"
        if np.count_nonzero(self.amortizations) > 1:
            notional += " amortizing"
        return f"{head}  {notional}  {self._terms()}".rstrip()

    @staticmethod
    def _build_schedule(
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        stub_first: bool,
        long_stub: bool,
        maturity_date: date | None = None,
    ) -> list[date]:
        """Adjusted schedule dates ``[start, d1, ..., maturity]``.

        Raises
        ------
        ValueError
            If `payment_frequency` is not one of MONTHLY, QUARTERLY,
            SEMIANNUAL or ANNUAL.
        """
        freq = _FREQ_TO_TENOR.get(payment_frequency)
        if freq is None:
            raise ValueError(
                f"Unsupported payment frequency '{payment_frequency}'. "
                f"Supported values: {list(_FREQ_TO_TENOR)}"
            )
        return ScheduleGenerator.generate_schedule(
            start_date=start_date,
            maturity_tenor=str(term),
            frequency_tenor=freq,
            adj_conv=adj_convention,
            maturity_adj_conv=term.adj_convention,
            end_date=maturity_date,
            stub_first=stub_first,
            long_stub=long_stub,
        )

    @staticmethod
    def _fx_fixing_dates(
        payment_dates: list[date],
        adj_convention: AdjustmentDateConventionBase,
        fx_fixing_lag: int,
    ) -> list[date]:
        """FX fixing dates `fx_fixing_lag` business days before each payment date.

        Raises
        ------
        ValueError
            If `fx_fixing_lag` is negative.
        """
        if fx_fixing_lag < 0:
            raise ValueError(f"fx_fixing_lag must be >= 0. Got {fx_fixing_lag}.")
        if fx_fixing_lag == 0:
            return list(payment_dates)
        calendar = adj_convention.calendar
        return [calendar.add_business_days(d, -fx_fixing_lag) for d in payment_dates]


@dataclass
class FixedLeg(SwapLeg):
    """Fixed-rate leg. Flows are computed at construction.

    Parameters
    ----------
    currency : Currency
        Currency of the flows.

    Attributes
    ----------
    rates : numpy.ndarray
        Per-coupon fixed :class:`Rate`.
    flows : numpy.ndarray
        Per-coupon interest plus amortization.
    """

    _label = "Fixed"

    currency: Currency
    rates: np.ndarray = field(init=False)
    flows: np.ndarray = field(init=False)

    def _terms(self) -> str:
        values = np.array([r.rate for r in self.rates])
        prefix = "from " if np.ptp(values) > 0 else ""
        return f"{prefix}{values[0]:.4%} {self.rates[0].rate_convention}"

    def __post_init__(self):
        super().__post_init__()
        self.rates = np.array([c.rate for c in self.coupons])
        self.flows = np.array([c.flow for c in self.coupons])

    @classmethod
    def from_term(
        cls,
        notional: float,
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        rate: Rate,
        currency: Currency,
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
        fx_fixing_lag: int = 0,
    ) -> FixedLeg:
        """Bullet fixed leg generated from a start date and a term.

        Parameters
        ----------
        notional : float
            Constant notional, repaid with the final coupon.
        start_date : date
            First accrual start.
        term : Term
            Leg tenor; its adjustment convention adjusts the maturity.
        payment_frequency : str
            MONTHLY, QUARTERLY, SEMIANNUAL or ANNUAL.
        adj_convention : AdjustmentDateConventionBase
            Business-day adjustment of the schedule dates.
        rate : Rate
            Fixed rate of every coupon.
        currency : Currency
            Currency of the flows.
        stub_first : bool, default True
            Place an irregular period at the start (True) or at the end.
        long_stub : bool, default False
            Merge the irregular period into its neighbour.
        maturity_date : date, optional
            Explicit maturity overriding `term`.
        fx_fixing_lag : int, default 0
            Business days between each FX fixing and its payment date.

        Returns
        -------
        FixedLeg
        """
        schedule = cls._build_schedule(
            start_date,
            term,
            payment_frequency,
            adj_convention,
            stub_first,
            long_stub,
            maturity_date,
        )
        maturity = max(schedule)
        fx_dates = cls._fx_fixing_dates(schedule[1:], adj_convention, fx_fixing_lag)
        coupons = [
            FixedCoupon(
                residual=notional,
                amortization=0.0 if schedule[i + 1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                fx_fixing_date=fx_dates[i],
                rate=rate,
            )
            for i in range(len(schedule) - 1)
        ]
        return cls(coupons=coupons, currency=currency)


@dataclass
class FloatingLeg(SwapLeg):
    """Floating-rate leg. Flows are projected at valuation from the index curve.

    Parameters
    ----------
    index : InterestIndex
        Floating index; its currency is the leg currency.
    spread : Rate
        Spread accrued on top of the index interest.

    Attributes
    ----------
    spreads_values : numpy.ndarray
        Per-coupon spread interest, added (not compounded) to the flow.
    """

    _label = "Floating"

    index: InterestIndex
    spread: Rate
    spreads_values: np.ndarray = field(init=False)

    def _terms(self) -> str:
        return (
            f"{getattr(self.index, 'name', self.index)} {self.spread.rate * 1e4:+.1f}bp"
        )

    def __post_init__(self):
        super().__post_init__()
        self.spreads_values = self.residuals * (
            self.spread.get_wealth_factor(self.start_dates, self.end_dates) - 1
        )

    @property
    def currency(self) -> Currency:
        """Currency of the flows: the index currency."""
        # All concrete InterestIndex types also extend Index, which holds currency.
        return cast(Index, self.index).currency  # type: ignore[return-value]


@dataclass
class TermRateLeg(FloatingLeg):
    """Floating leg on a term-rate index: one fixing per coupon.

    Parameters
    ----------
    fixing_dates : list of date
        Per-coupon rate fixing dates (usually before the accrual start).

    Warns
    -----
    UserWarning
        If a payment date is a holiday of the index calendar.
    """

    _label = "Term"

    fixing_dates: list[date] = field(default_factory=list)

    _fixing_ordinals: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self._fixing_ordinals = date_ordinals(self.fixing_dates)
        calendar = getattr(self.index, "calendar", None)
        if calendar is None:
            return
        holidays = [d for d in self.payment_dates if calendar.is_holiday(d)]
        if holidays:
            warnings.warn(
                f"Term-rate leg on {getattr(self.index, 'name', self.index)} pays on "
                f"holidays of the index calendar: {[d.isoformat() for d in holidays]}.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def from_term(
        cls,
        notional: float,
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        index: InterestIndex,
        spread: Rate | None = None,
        fixing_lag: int | None = None,
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
        fx_fixing_lag: int = 0,
    ) -> TermRateLeg:
        """Bullet term-rate leg generated from a start date and a term.

        Parameters
        ----------
        notional, start_date, term, payment_frequency, adj_convention
            As in :meth:`FixedLeg.from_term`.
        index : InterestIndex
            Term-rate index.
        spread : Rate, optional
            Spread over the index; zero when omitted.
        fixing_lag : int, optional
            Business days between each rate fixing and its accrual start;
            defaults to the index ``spot_lag``.
        stub_first, long_stub, maturity_date, fx_fixing_lag
            As in :meth:`FixedLeg.from_term`.

        Returns
        -------
        TermRateLeg
        """
        schedule = cls._build_schedule(
            start_date,
            term,
            payment_frequency,
            adj_convention,
            stub_first,
            long_stub,
            maturity_date,
        )
        spread = _zero_spread() if spread is None else spread
        if fixing_lag is None:
            fixing_lag = getattr(index, "spot_lag", 0)
        calendar = adj_convention.calendar
        n = len(schedule) - 1
        fixing_dates = [
            calendar.add_business_days(schedule[i], -fixing_lag) for i in range(n)
        ]
        maturity = max(schedule)
        fx_dates = cls._fx_fixing_dates(schedule[1:], adj_convention, fx_fixing_lag)
        coupons = [
            TermRateCoupon(
                residual=notional,
                amortization=0.0 if schedule[i + 1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                fx_fixing_date=fx_dates[i],
                spread=spread,
                fixing_date=fixing_dates[i],
            )
            for i in range(n)
        ]
        return cls(
            coupons=coupons, index=index, spread=spread, fixing_dates=fixing_dates
        )


@dataclass
class OvernightLeg(FloatingLeg):
    """Floating leg on an overnight index, compounded daily (OIS-style)."""

    _label = "Overnight"

    @classmethod
    def from_term(
        cls,
        notional: float,
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        index: InterestIndex,
        spread: Rate | None = None,
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
        fx_fixing_lag: int = 0,
    ) -> OvernightLeg:
        """Bullet overnight leg generated from a start date and a term.

        Parameters
        ----------
        notional, start_date, term, payment_frequency, adj_convention
            As in :meth:`FixedLeg.from_term`.
        index : InterestIndex
            Overnight rate or interest price index.
        spread : Rate, optional
            Spread over the index; zero when omitted.
        stub_first, long_stub, maturity_date, fx_fixing_lag
            As in :meth:`FixedLeg.from_term`.

        Returns
        -------
        OvernightLeg
        """
        return cls._from_schedule(
            OvernightCoupon,
            notional,
            start_date,
            term,
            payment_frequency,
            adj_convention,
            index,
            spread,
            stub_first,
            long_stub,
            maturity_date,
            fx_fixing_lag,
        )

    @classmethod
    def _from_schedule(
        cls,
        coupon_cls,
        notional,
        start_date,
        term,
        payment_frequency,
        adj_convention,
        index,
        spread,
        stub_first,
        long_stub,
        maturity_date,
        fx_fixing_lag,
    ):
        """Bullet daily-compounded leg of `coupon_cls` coupons (shared by subclasses)."""
        schedule = cls._build_schedule(
            start_date,
            term,
            payment_frequency,
            adj_convention,
            stub_first,
            long_stub,
            maturity_date,
        )
        spread = _zero_spread() if spread is None else spread
        maturity = max(schedule)
        fx_dates = cls._fx_fixing_dates(schedule[1:], adj_convention, fx_fixing_lag)
        coupons = [
            coupon_cls(
                residual=notional,
                amortization=0.0 if schedule[i + 1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=schedule[i + 1],
                fx_fixing_date=fx_dates[i],
                spread=spread,
            )
            for i in range(len(schedule) - 1)
        ]
        return cls(coupons=coupons, index=index, spread=spread)


@dataclass
class XCCYFloatingLeg(OvernightLeg):
    """Floating leg of a cross-currency swap.

    Compounded daily like :class:`OvernightLeg`, built from :class:`XCCYCoupon`.
    The final coupon carries the notional (final exchange); no initial notional
    exchange is modelled.
    """

    _label = "XCCY Overnight"

    @classmethod
    def from_term(
        cls,
        notional: float,
        start_date: date,
        term: Term,
        payment_frequency: str,
        adj_convention: AdjustmentDateConventionBase,
        index: InterestIndex,
        spread: Rate | None = None,
        stub_first: bool = True,
        long_stub: bool = False,
        maturity_date: date | None = None,
        fx_fixing_lag: int = 0,
    ) -> XCCYFloatingLeg:
        """Bullet cross-currency floating leg.

        Parameters are as in :meth:`OvernightLeg.from_term`.

        Returns
        -------
        XCCYFloatingLeg
        """
        return cls._from_schedule(
            XCCYCoupon,
            notional,
            start_date,
            term,
            payment_frequency,
            adj_convention,
            index,
            spread,
            stub_first,
            long_stub,
            maturity_date,
            fx_fixing_lag,
        )
