from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar, cast

import numpy as np

from fintoolsom import settings
from fintoolsom.rates.Rates import Rate, RateConvention

from .coupons import (
    FixedCoupon,
    FloatingCoupon,
    OvernightCoupon,
    SwapCoupon,
    TermRateCoupon,
    XCCYCoupon,
)
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


def _read_only(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


@dataclass(kw_only=True, eq=False)
class SwapLeg:
    """Base swap leg: per-coupon arrays of dates and notionals.

    A leg stores arrays, not coupon objects: :meth:`get_coupons` and
    :meth:`get_coupon` build those on request, and :meth:`from_coupons` builds a leg
    from them.

    Parameters
    ----------
    start_dates, end_dates : sequence of date
        Per-coupon accrual start and end.
    residuals : array_like of float
        Per-coupon outstanding notional the coupon accrues on.
    payment_dates : sequence of date, optional
        Per-coupon payment date; defaults to `end_dates` (no payment lag).
    fx_fixing_dates : sequence of date, optional
        Per-coupon FX fixing date, used when the leg settles non-deliverably in
        another currency; defaults to `payment_dates`.

    Raises
    ------
    ValueError
        If the arrays differ in length; a start date is not before its end date, an
        end date is after its payment date or an FX fixing date after its payment
        date; or the start or payment dates decrease along the leg.

    Attributes
    ----------
    amortizations : numpy.ndarray
        Per-coupon notional repayment, from the residuals: the notional is fully
        repaid by the last coupon, ``amortizations[i] = residuals[i] -
        residuals[i + 1]`` and ``amortizations[-1] = residuals[-1]``. A negative one
        is a drawdown (an accreting notional). There is no initial exchange.
    fixed_interest : numpy.ndarray
        Per-coupon interest known at construction: at the fixed rate for a fixed
        leg, at the spread for a floating one.
    fixed_flows : numpy.ndarray
        ``fixed_interest + amortizations``: a fixed leg's flows, and the part of a
        floating leg's flows added to its index interest.

    Notes
    -----
    Every array is read-only: a leg is immutable, its derived arrays and its C++
    core are computed once. Use :func:`dataclasses.replace` for a changed copy.
    """

    start_dates: Sequence[date]
    end_dates: Sequence[date]
    residuals: Any
    payment_dates: Sequence[date] | None = None
    fx_fixing_dates: Sequence[date] | None = None

    amortizations: np.ndarray = field(init=False)
    fixed_interest: np.ndarray = field(init=False)
    fixed_flows: np.ndarray = field(init=False)

    # The dates above as integer ordinals, so a valuation needs no date conversion.
    _payment_ordinals: np.ndarray = field(init=False, repr=False)
    _start_ordinals: np.ndarray = field(init=False, repr=False)
    _end_ordinals: np.ndarray = field(init=False, repr=False)
    _fx_fixing_ordinals: np.ndarray = field(init=False, repr=False)
    # Pays on its end dates, each coupon starting where the previous one ends.
    _par_schedule: bool = field(init=False, repr=False)

    # C++ backend (fintoolsom.settings, "derivatives"), read at construction. The
    # leg's fintoolsom._native.SwapLegCore is built on its first valuation.
    _native: bool = field(init=False, repr=False)
    _core: Any = field(init=False, default=None, repr=False)

    _label: ClassVar[str] = "Leg"

    def __post_init__(self):
        self._native = settings.is_cpp("derivatives")
        self._core = None
        self.start_dates = list(self.start_dates)
        self.end_dates = list(self.end_dates)
        n = len(self.start_dates)
        self.payment_dates = (
            self.end_dates if self.payment_dates is None else list(self.payment_dates)
        )
        self.fx_fixing_dates = (
            self.payment_dates
            if self.fx_fixing_dates is None
            else list(self.fx_fixing_dates)
        )
        r = np.array(self.residuals, dtype=float)
        if r.shape != (n,) or not (
            len(self.end_dates)
            == len(self.payment_dates)
            == len(self.fx_fixing_dates)
            == n
        ):
            raise ValueError(
                "A leg's start, end, payment and FX fixing dates and residuals must "
                f"have one value per coupon; got {n} start dates, {len(self.end_dates)}"
                f" end dates, {len(self.payment_dates)} payment dates, "
                f"{len(self.fx_fixing_dates)} FX fixing dates and residuals of shape "
                f"{r.shape}."
            )
        self.residuals = _read_only(r)
        own_pay = self.payment_dates is not self.end_dates
        own_fx = self.fx_fixing_dates is not self.payment_dates
        arrays = (
            settings.native().leg_arrays(
                self.start_dates,
                self.end_dates,
                self.payment_dates if own_pay else None,
                self.fx_fixing_dates if own_fx else None,
                r,
            )
            if self._native
            else None
        )
        if arrays is None:
            a = self._python_arrays(own_pay, own_fx)
        else:
            start, end, pay, fx, a, self._par_schedule = arrays
            self._start_ordinals = _read_only(start)
            self._end_ordinals = _read_only(end)
            self._payment_ordinals = end if pay is None else _read_only(pay)
            self._fx_fixing_ordinals = (
                self._payment_ordinals if fx is None else _read_only(fx)
            )
        self.amortizations = _read_only(a)
        rate = self.fixed_rate
        interest = (
            np.zeros(n)
            if rate is None or n == 0
            else np.array(
                rate.get_accrued_interest(r, self.start_dates, self.end_dates),
                dtype=float,
            ).reshape(n)
        )
        self.fixed_interest = _read_only(interest)
        self.fixed_flows = _read_only(interest + a)

    def _python_arrays(self, own_pay: bool, own_fx: bool) -> np.ndarray:
        """The ordinals (checked) and the amortizations, without the C++ backend; the
        backend's ``leg_arrays`` gives the same."""
        self._start_ordinals = _read_only(date_ordinals(self.start_dates))
        end = self._end_ordinals = _read_only(date_ordinals(self.end_dates))
        pay = self._payment_ordinals = (
            _read_only(date_ordinals(self.payment_dates)) if own_pay else end
        )
        self._fx_fixing_ordinals = (
            _read_only(date_ordinals(self.fx_fixing_dates)) if own_fx else pay
        )
        self._check_dates()
        self._par_schedule = bool(
            len(end)
            and np.array_equal(pay, end)
            and np.array_equal(self._start_ordinals[1:], end[:-1])
        )
        r = self.residuals
        a = np.empty_like(r)
        a[:-1] = r[:-1] - r[1:]
        a[-1:] = r[-1:]
        return a

    def _check_dates(self) -> None:
        start, end = self._start_ordinals, self._end_ordinals
        pay, fx = self._payment_ordinals, self._fx_fixing_ordinals
        problems = [
            ("a start date is not before its end date", start >= end),
            ("an end date is after its payment date", end > pay),
            ("an FX fixing date is after its payment date", fx > pay),
        ]
        for what, bad in problems:
            if bad.any():
                i = int(np.argmax(bad))
                raise ValueError(f"Coupon {i}: {what}.")
        if (np.diff(start) < 0).any() or (np.diff(pay) < 0).any():
            raise ValueError(
                "Coupons must be in schedule order: start and payment dates "
                "non-decreasing."
            )

    def __getstate__(self) -> dict:
        # The C++ core is not picklable; a copy rebuilds it on its first valuation.
        return {**self.__dict__, "_core": None}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        for value in state.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    @property
    def fixed_rate(self) -> Rate | None:
        """Rate of :attr:`fixed_interest`: the fixed rate or the spread; None for the
        base leg (no interest)."""
        return None

    @property
    def n_coupons(self) -> int:
        """Number of coupons."""
        return len(self._start_ordinals)

    def get_coupon(self, i: int) -> SwapCoupon:
        """Coupon `i` (negative counts from the end), built on request: changing it
        does not change the leg.

        Raises
        ------
        IndexError
            If the leg has no coupon `i`.
        """
        i = range(self.n_coupons)[i]
        return self._coupon(
            i,
            residual=float(self.residuals[i]),
            amortization=float(self.amortizations[i]),
            start_date=self.start_dates[i],
            end_date=self.end_dates[i],
            payment_date=self.payment_dates[i],  # type: ignore[index]
            fx_fixing_date=self.fx_fixing_dates[i],  # type: ignore[index]
        )

    def get_coupons(self) -> list[SwapCoupon]:
        """Every coupon, in schedule order, built on request (see :meth:`get_coupon`)."""
        return [self.get_coupon(i) for i in range(self.n_coupons)]

    def _coupon(self, i: int, **common) -> SwapCoupon:
        return SwapCoupon(**common)

    @classmethod
    def from_coupons(cls, coupons: Sequence[SwapCoupon], **kwargs):
        """Leg of `coupons`, for code written against coupon lists.

        Parameters
        ----------
        coupons : sequence of SwapCoupon
            The coupons, in schedule order. Their amortizations must repay the
            notional as the leg does (see :attr:`amortizations`).
        **kwargs
            The leg's other fields (``currency``, ``index``, ``spread``, ...). A
            fixed leg's ``rate`` and a term-rate leg's ``fixing_dates`` default to
            the coupons'.

        Raises
        ------
        ValueError
            If the amortizations do not follow the residuals, or the coupons of a
            fixed leg do not share one rate.
        """
        coupons = list(coupons)
        leg = cls(
            start_dates=[c.start_date for c in coupons],
            end_dates=[c.end_date for c in coupons],
            payment_dates=[c.payment_date for c in coupons],
            fx_fixing_dates=[c.fx_fixing_date for c in coupons],  # type: ignore[misc]
            residuals=[c.residual for c in coupons],
            **cls._coupon_fields(coupons, kwargs),
        )
        given = np.array([c.amortization for c in coupons], dtype=float)
        scale = float(np.abs(leg.residuals).max(initial=0.0))
        if not np.allclose(given, leg.amortizations, rtol=0.0, atol=1e-12 * scale):
            raise ValueError(
                "The coupons' amortizations must repay the notional: residual minus "
                "the next residual, the last residual on the last coupon. Got "
                f"amortizations {given.tolist()} for residuals "
                f"{leg.residuals.tolist()}."
            )
        return leg

    @classmethod
    def _coupon_fields(cls, coupons: list, kwargs: dict) -> dict:
        """The leg's fields `from_coupons` takes from `kwargs` or the coupons."""
        return kwargs

    def _terms(self) -> str:
        """Rate description used by :meth:`__str__`; empty for the base leg."""
        return ""

    def __str__(self) -> str:
        ccy = getattr(self, "currency", None)
        head = f"{self._label:<15}{ccy if ccy is not None else '?'}"
        if not self.n_coupons:
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
    def _schedule_dates(
        schedule: list[date],
        adj_convention: AdjustmentDateConventionBase,
        payment_lag: int,
        fx_fixing_lag: int,
    ) -> dict:
        """A bullet leg's date fields from its schedule: payment dates `payment_lag`
        business days after each end date, FX fixings `fx_fixing_lag` business days
        before each payment date (None, the defaults, for no lag).

        Raises
        ------
        ValueError
            If a lag is negative.
        """
        if payment_lag < 0:
            raise ValueError(f"payment_lag must be >= 0. Got {payment_lag}.")
        if fx_fixing_lag < 0:
            raise ValueError(f"fx_fixing_lag must be >= 0. Got {fx_fixing_lag}.")
        end = schedule[1:]
        calendar = adj_convention.calendar
        pay = (
            [calendar.add_business_days(d, payment_lag) for d in end]
            if payment_lag
            else None
        )
        fx = (
            [calendar.add_business_days(d, -fx_fixing_lag) for d in pay or end]
            if fx_fixing_lag
            else None
        )
        return {
            "start_dates": schedule[:-1],
            "end_dates": end,
            "payment_dates": pay,
            "fx_fixing_dates": fx,
        }


@dataclass(kw_only=True, eq=False)
class FixedLeg(SwapLeg):
    """Fixed-rate leg. Flows are computed at construction.

    Parameters
    ----------
    currency : Currency
        Currency of the flows.
    rate : Rate
        Fixed rate of every coupon.

    Attributes
    ----------
    flows : numpy.ndarray
        Per-coupon interest plus amortization (:attr:`fixed_flows`).
    """

    _label: ClassVar[str] = "Fixed"

    currency: Currency
    rate: Rate

    @property
    def fixed_rate(self) -> Rate:
        return self.rate

    @property
    def flows(self) -> np.ndarray:
        """Per-coupon interest plus amortization (:attr:`fixed_flows`)."""
        return self.fixed_flows

    def _terms(self) -> str:
        return f"{self.rate.rate:.4%} {self.rate.rate_convention}"

    def _coupon(self, i: int, **common) -> FixedCoupon:
        return FixedCoupon(**common, rate=self.rate)

    @classmethod
    def _coupon_fields(cls, coupons: list, kwargs: dict) -> dict:
        if "rate" not in kwargs:
            rates = [c.rate for c in coupons]
            if not rates:
                raise ValueError("A fixed leg with no coupon needs a rate.")
            if any(r is not rates[0] and r != rates[0] for r in rates[1:]):
                raise ValueError(
                    "A fixed leg has one rate: its coupons' rates differ "
                    f"({sorted({r.rate for r in rates})})."
                )
            kwargs = {**kwargs, "rate": rates[0]}
        return kwargs

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
        payment_lag: int = 0,
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
        payment_lag : int, default 0
            Business days between each period end date and its payment date.

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
        return cls(
            **cls._schedule_dates(schedule, adj_convention, payment_lag, fx_fixing_lag),
            residuals=np.full(len(schedule) - 1, float(notional)),
            rate=rate,
            currency=currency,
        )


@dataclass(kw_only=True, eq=False)
class FloatingLeg(SwapLeg):
    """Floating-rate leg. Flows are projected at valuation from the index curve.

    Parameters
    ----------
    index : InterestIndex
        Floating index; its currency is the leg currency.
    spread : Rate, optional
        Spread of every coupon, accrued on top of the index interest and added (not
        compounded) to the flow; zero when omitted.

    Attributes
    ----------
    spreads_values : numpy.ndarray
        Per-coupon spread interest (:attr:`fixed_interest`).

    Notes
    -----
    A leg paying on its accrual end dates, with each coupon starting where the
    previous one ends, is *par eligible*: discounted on the curve that projects its
    index, its index interest plus amortizations sum to ``residuals[0] *
    df(first start)``. The calculator then values it from that one df plus the
    spreads.
    """

    _label: ClassVar[str] = "Floating"

    index: InterestIndex
    spread: Rate | None = None
    _par_eligible: bool = field(init=False, repr=False)
    _has_spread: bool = field(init=False, repr=False)

    def __post_init__(self):
        if self.spread is None:
            self.spread = _zero_spread()
        super().__post_init__()
        self._par_eligible = self._repays_at_par()
        self._has_spread = bool(self.fixed_interest.any())

    @property
    def fixed_rate(self) -> Rate:
        return self.spread  # type: ignore[return-value]

    @property
    def spreads_values(self) -> np.ndarray:
        """Per-coupon spread interest (:attr:`fixed_interest`)."""
        return self.fixed_interest

    def _terms(self) -> str:
        return (
            f"{getattr(self.index, 'name', self.index)} "
            f"{self.spread.rate * 1e4:+.1f}bp"  # type: ignore[union-attr]
        )

    def _repays_at_par(self) -> bool:
        """No payment lag and a contiguous schedule (the notional is always fully
        repaid).

        Under these, sum over coupons of ``residual * (df(start) - df(end))``
        plus ``amortization * df(end)`` telescopes to ``residuals[0] * df(start_0)``.
        """
        return self._par_schedule

    @property
    def currency(self) -> Currency:
        """Currency of the flows: the index currency."""
        # All concrete InterestIndex types also extend Index, which holds currency.
        return cast(Index, self.index).currency  # type: ignore[return-value]

    @classmethod
    def _bullet(
        cls,
        notional,
        start_date,
        term,
        payment_frequency,
        adj_convention,
        stub_first,
        long_stub,
        maturity_date,
        fx_fixing_lag,
        payment_lag,
        **fields,
    ):
        """Bullet leg from a schedule (shared by the subclasses' ``from_term``)."""
        schedule = cls._build_schedule(
            start_date,
            term,
            payment_frequency,
            adj_convention,
            stub_first,
            long_stub,
            maturity_date,
        )
        return cls(
            **cls._schedule_dates(schedule, adj_convention, payment_lag, fx_fixing_lag),
            residuals=np.full(len(schedule) - 1, float(notional)),
            **fields,
        )


@dataclass(kw_only=True, eq=False)
class TermRateLeg(FloatingLeg):
    """Floating leg on a term-rate index: one fixing per coupon.

    Parameters
    ----------
    fixing_dates : sequence of date, optional
        Per-coupon rate fixing dates, on or before each accrual start; defaults to
        the start dates.

    Raises
    ------
    ValueError
        If `fixing_dates` has not one date per coupon or a fixing is after its
        coupon's start.

    Warns
    -----
    UserWarning
        If a payment date is a holiday of the index calendar.
    """

    _label: ClassVar[str] = "Term"

    fixing_dates: Sequence[date] | None = None

    _fixing_ordinals: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.fixing_dates is None:
            self.fixing_dates = self.start_dates
            self._fixing_ordinals = self._start_ordinals
        else:
            self.fixing_dates = list(self.fixing_dates)
            if len(self.fixing_dates) != self.n_coupons:
                raise ValueError(
                    f"A term-rate leg takes one fixing date per coupon; got "
                    f"{len(self.fixing_dates)} for {self.n_coupons} coupons."
                )
            self._fixing_ordinals = _read_only(date_ordinals(self.fixing_dates))
            late = self._fixing_ordinals > self._start_ordinals
            if late.any():
                raise ValueError(
                    f"Coupon {int(np.argmax(late))}: the fixing date is after the "
                    "start date."
                )
        calendar = getattr(self.index, "calendar", None)
        if calendar is None:
            return
        holidays = [d for d in self.payment_dates if calendar.is_holiday(d)]  # type: ignore[union-attr]
        if holidays:
            warnings.warn(
                f"Term-rate leg on {getattr(self.index, 'name', self.index)} pays on "
                f"holidays of the index calendar: {[d.isoformat() for d in holidays]}.",
                UserWarning,
                stacklevel=3,
            )

    def _coupon(self, i: int, **common) -> TermRateCoupon:
        return TermRateCoupon(
            **common,
            spread=self.spread,  # type: ignore[arg-type]
            fixing_date=self.fixing_dates[i],  # type: ignore[index]
        )

    @classmethod
    def _coupon_fields(cls, coupons: list, kwargs: dict) -> dict:
        if "fixing_dates" not in kwargs:
            kwargs = {**kwargs, "fixing_dates": [c.fixing_date for c in coupons]}
        return kwargs

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
        payment_lag: int = 0,
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
        stub_first, long_stub, maturity_date, fx_fixing_lag, payment_lag
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
        lag: int = getattr(index, "spot_lag", 0) if fixing_lag is None else fixing_lag
        calendar = adj_convention.calendar
        fixing_dates = [calendar.add_business_days(d, -lag) for d in schedule[:-1]]
        return cls(
            **cls._schedule_dates(schedule, adj_convention, payment_lag, fx_fixing_lag),
            residuals=np.full(len(schedule) - 1, float(notional)),
            index=index,
            spread=spread,
            fixing_dates=fixing_dates,
        )


@dataclass(kw_only=True, eq=False)
class OvernightLeg(FloatingLeg):
    """Floating leg on an overnight index, compounded daily (OIS-style)."""

    _label: ClassVar[str] = "Overnight"

    def _coupon(self, i: int, **common) -> FloatingCoupon:
        return OvernightCoupon(**common, spread=self.spread)  # type: ignore[arg-type]

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
        payment_lag: int = 0,
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
        stub_first, long_stub, maturity_date, fx_fixing_lag, payment_lag
            As in :meth:`FixedLeg.from_term`.

        Returns
        -------
        OvernightLeg
        """
        return cls._bullet(
            notional,
            start_date,
            term,
            payment_frequency,
            adj_convention,
            stub_first,
            long_stub,
            maturity_date,
            fx_fixing_lag,
            payment_lag,
            index=index,
            spread=spread,
        )


@dataclass(kw_only=True, eq=False)
class XCCYFloatingLeg(OvernightLeg):
    """Floating leg of a cross-currency swap.

    Compounded daily like :class:`OvernightLeg`; its coupons are
    :class:`XCCYCoupon`. The final coupon carries the notional (final exchange); no
    initial notional exchange is modelled.
    """

    _label: ClassVar[str] = "XCCY Overnight"

    def _coupon(self, i: int, **common) -> FloatingCoupon:
        return XCCYCoupon(**common, spread=self.spread)  # type: ignore[arg-type]

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
        payment_lag: int = 0,
    ) -> XCCYFloatingLeg:
        """Bullet cross-currency floating leg.

        Parameters are as in :meth:`OvernightLeg.from_term`.

        Returns
        -------
        XCCYFloatingLeg
        """
        return cls._bullet(
            notional,
            start_date,
            term,
            payment_frequency,
            adj_convention,
            stub_first,
            long_stub,
            maturity_date,
            fx_fixing_lag,
            payment_lag,
            index=index,
            spread=spread,
        )
