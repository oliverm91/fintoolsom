from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import cast

import numpy as np

from fintoolsom.rates.Rates import LinearInterestConvention, Rate, RateConvention

from .coupons import FixedCoupon, OvernightCoupon, TermRateCoupon, XCCYCoupon
from ...dates import AdjustmentDateConventionBase, ScheduleGenerator
from ...dates.date_counts import date_ordinals
from ...dates.term import Term
from ...market.currencies import Currency
from ...market.index import Index, InterestIndex, TermRateIndex


# Maps PaymentFrequency enum value to ScheduleGenerator frequency string.
_FREQ_TO_TENOR: dict[str, str] = {
    "MONTHLY": "1m",
    "QUARTERLY": "3m",
    "SEMIANNUAL": "6m",
    "ANNUAL": "1y",
}


def _zero_spread(index: InterestIndex) -> Rate:
    """The spread of a floating leg built without one: zero, linear in the index's
    day count and base.

    A spread is added to the index flow, not compounded with it, so it is linear;
    its basis is the index's own (``index.rate_convention``: linear Act/360 for SOFR
    or ICP, Act/365 for SONIA if defined so), linear Act/360 for an index without
    one. The value is zero either way: the convention matters to whatever moves the
    spread later, such as a 1bp bump. Each call returns a new :class:`Rate`, so legs
    never share one mutable spread.
    """
    convention = getattr(index, "rate_convention", None)
    if convention is None:
        return Rate(
            RateConvention(LinearInterestConvention, time_fraction_base=360), 0.0
        )
    basis = RateConvention(
        LinearInterestConvention,
        convention.day_count_convention,
        convention.time_fraction_base,
    )
    return Rate(basis, 0.0)


def _interest_from_window(
    window_accruals: np.ndarray, ratios: np.ndarray, power: np.ndarray
) -> np.ndarray:
    """Interest per unit notional of coupons whose rate is the index forward over a
    projection window, from that window's accrual ``a = wf(window) - 1``.

    The forward ``F`` over the window is accrued over the coupon's own period in the
    index convention. With ``ratio = tau(period) / tau(window)`` in that convention's
    day count, linear interest is ``F * tau(period) = a * ratio``; compounded and
    exponential interest are ``(1 + a) ** ratio - 1`` (`power`). A coupon projected
    over its own period has ratio 1 and is linear, so its interest is ``a`` exactly.

    The calculator and :class:`~fintoolsom.derivatives.calculator.ValuationBatch`
    both go through this function, elementwise, so they agree to the bit.
    """
    linear = window_accruals * ratios
    if not power.any():
        return linear
    return np.where(power, np.expm1(ratios * np.log1p(window_accruals)), linear)


def _interest_from_window_slope(
    window_accruals: np.ndarray, ratios: np.ndarray, power: np.ndarray
) -> np.ndarray:
    """d :func:`_interest_from_window` / d window accrual."""
    if not power.any():
        return ratios
    return np.where(
        power, ratios * np.exp((ratios - 1) * np.log1p(window_accruals)), ratios
    )


@dataclass
class SwapLeg:
    """Base swap leg: a list of coupons plus cached per-coupon arrays.

    Parameters
    ----------
    coupons : list of SwapCoupon
        The leg's coupons, in schedule order.

    Raises
    ------
    ValueError
        If a coupon's start date is not before its end date, its end date is after
        its payment date or its FX fixing date after its payment date; or the
        coupons' start or payment dates decrease along the list.

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
        start, end = self._start_ordinals, self._end_ordinals
        pay, fx = self._payment_ordinals, self._fx_fixing_ordinals
        for what, bad in (
            ("a start date is not before its end date", start >= end),
            ("an end date is after its payment date", end > pay),
            ("an FX fixing date is after its payment date", fx > pay),
        ):
            if bad.any():
                raise ValueError(f"Coupon {int(np.argmax(bad))}: {what}.")
        if (np.diff(self._start_ordinals) < 0).any() or (
            np.diff(self._payment_ordinals) < 0
        ).any():
            raise ValueError(
                "Coupons must be in schedule order: start and payment dates "
                "non-decreasing."
            )

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
    def _payment_dates(
        end_dates: list[date],
        adj_convention: AdjustmentDateConventionBase,
        payment_lag: int,
    ) -> list[date]:
        """Payment dates `payment_lag` business days after each period end date.

        Raises
        ------
        ValueError
            If `payment_lag` is negative.
        """
        if payment_lag < 0:
            raise ValueError(f"payment_lag must be >= 0. Got {payment_lag}.")
        if payment_lag == 0:
            return list(end_dates)
        calendar = adj_convention.calendar
        return [calendar.add_business_days(d, payment_lag) for d in end_dates]

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
        maturity = max(schedule)
        pay_dates = cls._payment_dates(schedule[1:], adj_convention, payment_lag)
        fx_dates = cls._fx_fixing_dates(pay_dates, adj_convention, fx_fixing_lag)
        coupons = [
            FixedCoupon(
                residual=notional,
                amortization=0.0 if schedule[i + 1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=pay_dates[i],
                fx_fixing_date=fx_dates[i],
                rate=rate,
                termination_date=schedule[-1],
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
    fixed_flows : numpy.ndarray
        Per-coupon part of the flow known at construction: spread interest
        plus amortization. A coupon's flow is its index interest plus this.

    Notes
    -----
    A leg paying on its accrual end dates, with each coupon starting where the
    previous one ends and the notional repaid through the amortizations, is
    *par eligible*: discounted on the curve that projects its index, its index
    interest plus amortizations sum to ``residuals[0] * df(first start)``. The
    calculator then values it from that one df plus the spreads.

    Each coupon projects its index over a *projection window*: its own accrual
    period, except for a term-rate coupon (:class:`TermRateLeg`), whose rate is the
    index forward over the index's own tenor from its fixing.
    """

    _label = "Floating"

    index: InterestIndex
    spread: Rate
    spreads_values: np.ndarray = field(init=False)
    fixed_flows: np.ndarray = field(init=False)
    _par_eligible: bool = field(init=False, repr=False)
    _has_spread: bool = field(init=False, repr=False)
    # Projection windows (ordinals) and the conversion of their accruals into each
    # coupon's interest (_interest_from_window); the accrual period itself, ratio 1
    # and linear, unless a TermRateLeg says otherwise.
    _window_start_ordinals: np.ndarray = field(init=False, repr=False)
    _window_end_ordinals: np.ndarray = field(init=False, repr=False)
    _window_ratios: np.ndarray = field(init=False, repr=False)
    _window_power: np.ndarray = field(init=False, repr=False)
    _projects_on_accrual: bool = field(init=False, repr=False)

    def _terms(self) -> str:
        return (
            f"{getattr(self.index, 'name', self.index)} {self.spread.rate * 1e4:+.1f}bp"
        )

    def __post_init__(self):
        super().__post_init__()
        termination = self.end_dates[-1] if self.end_dates else None
        self.spreads_values = self.residuals * (
            self.spread.get_wealth_factor(self.start_dates, self.end_dates, termination)
            - 1
        )
        self.fixed_flows = self.spreads_values + self.amortizations
        self._par_eligible = self._repays_at_par()
        self._has_spread = bool(self.spreads_values.any())
        self._window_start_ordinals = self._start_ordinals
        self._window_end_ordinals = self._end_ordinals
        self._window_ratios = np.ones(len(self.coupons))
        self._window_power = np.zeros(len(self.coupons), dtype=bool)
        self._projects_on_accrual = True

    def _projected_interest(
        self, window_accruals: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Interest per unit notional of the `mask` coupons from the accruals of
        their projection windows (:func:`_interest_from_window`)."""
        if self._projects_on_accrual:
            return window_accruals
        return _interest_from_window(
            window_accruals, self._window_ratios[mask], self._window_power[mask]
        )

    def _repays_at_par(self) -> bool:
        """No payment lag, a contiguous schedule and the notional fully repaid.

        Under these, sum over coupons of ``residual * (df(start) - df(end))``
        plus ``amortization * df(end)`` telescopes to ``residuals[0] * df(start_0)``.
        """
        if not self.coupons:
            return False
        r, a = self.residuals, self.amortizations
        return bool(
            np.array_equal(self._payment_ordinals, self._end_ordinals)
            and np.array_equal(self._start_ordinals[1:], self._end_ordinals[:-1])
            and np.allclose(r[1:], r[:-1] - a[:-1], rtol=1e-12, atol=0.0)
            and np.isclose(r[-1], a[-1], rtol=1e-12, atol=0.0)
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

    Notes
    -----
    A coupon pays the index fixing of its fixing date, accrued over its own period
    ``[start, end]`` in the index's convention. That fixing covers the index's
    tenor from its value date: ``s = fixing + spot_lag`` business days of the index
    calendar to ``e = index.get_maturity(s)``. Before it fixes, the coupon projects
    the index curve's forward over ``[s, e]`` the same way, so a stub or a period
    adjusted off the index tenor gets the index's rate, not the curve's rate over
    its own length. On a regular period ``[s, e]`` is ``[start, end]`` and nothing
    changes. An index without a tenor (not a ``TermRateIndex``) projects over the
    accrual period.

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
        self._set_projection_windows()
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

    def _set_projection_windows(self) -> None:
        """Each coupon's fixing window ``[fixing + spot_lag, + index tenor]`` and
        the conversion of its accrual into the coupon's interest."""
        index = self.index
        if not isinstance(index, TermRateIndex) or not self.coupons:
            return
        starts = [
            index.calendar.add_business_days(d, index.spot_lag)
            for d in self.fixing_dates
        ]
        ends = [index.get_maturity(s) for s in starts]
        window_starts, window_ends = date_ordinals(starts), date_ordinals(ends)
        on_accrual = (window_starts == self._start_ordinals) & (
            window_ends == self._end_ordinals
        )
        if on_accrual.all():
            return
        convention = index.rate_convention
        day_count, base = convention.day_count_convention, convention.time_fraction_base
        ratios = np.asarray(
            day_count.get_time_fraction(self.start_dates, self.end_dates, base),
            dtype=float,
        ) / np.asarray(day_count.get_time_fraction(starts, ends, base), dtype=float)
        self._window_start_ordinals = window_starts
        self._window_end_ordinals = window_ends
        # A coupon on its window keeps ratio 1 and linear: its interest is exact.
        self._window_ratios = np.where(on_accrual, 1.0, ratios)
        self._window_power = ~on_accrual & (
            convention.interest_convention is not LinearInterestConvention
        )
        self._projects_on_accrual = False
        # Index interest no longer telescopes to residuals[0] * df(first start).
        self._par_eligible = False

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
            Spread over the index. When omitted, zero and linear in the index's
            day count and base (see ``InterestIndex``).
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
        spread = _zero_spread(index) if spread is None else spread
        if fixing_lag is None:
            fixing_lag = getattr(index, "spot_lag", 0)
        calendar = adj_convention.calendar
        n = len(schedule) - 1
        fixing_dates = [
            calendar.add_business_days(schedule[i], -fixing_lag) for i in range(n)
        ]
        maturity = max(schedule)
        pay_dates = cls._payment_dates(schedule[1:], adj_convention, payment_lag)
        fx_dates = cls._fx_fixing_dates(pay_dates, adj_convention, fx_fixing_lag)
        coupons = [
            TermRateCoupon(
                residual=notional,
                amortization=0.0 if schedule[i + 1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=pay_dates[i],
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
            Spread over the index. When omitted, zero and linear in the index's
            day count and base (see ``InterestIndex``).
        stub_first, long_stub, maturity_date, fx_fixing_lag, payment_lag
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
            payment_lag,
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
        payment_lag=0,
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
        spread = _zero_spread(index) if spread is None else spread
        maturity = max(schedule)
        pay_dates = cls._payment_dates(schedule[1:], adj_convention, payment_lag)
        fx_dates = cls._fx_fixing_dates(pay_dates, adj_convention, fx_fixing_lag)
        coupons = [
            coupon_cls(
                residual=notional,
                amortization=0.0 if schedule[i + 1] != maturity else notional,
                start_date=schedule[i],
                end_date=schedule[i + 1],
                payment_date=pay_dates[i],
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
        payment_lag: int = 0,
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
            payment_lag,
        )
