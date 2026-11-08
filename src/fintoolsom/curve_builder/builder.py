"""Curve bootstrap from market quotes: :func:`build_curves`.

The build is split in private modules:

* :mod:`._keys`   — data arrangement: the curves each quote reads, its pillar dates;
* :mod:`._plan`   — planning: pillar ownership, solve layout, readiness, cycles;
* :mod:`._seeds`  — initial guesses and short-end anchors;
* :mod:`._solver` — the unknown layout, the atomic root, the date-by-date bootstrap.

:class:`_CurveBuilder` drives them: it solves one ready group at a time, merges
groups that wait on each other, finally checks that every quote that pinned a
pillar still reprices and warns about the quotes it ditched (no pillar of their own).
"""
from __future__ import annotations

import dataclasses
import warnings

from ..derivatives.calculator import Calculator
from ..derivatives.forwards import Forward
from ..derivatives.swaps import Swap
from ..market.currencies import Currency, CurrencyName
from ..market.index import Index
from ..market.conventions import FixedLegSpec, FloatingLegSpec
from ..market.market import Market
from ..rates import InterpolationMethod, Rate
from ._errors import InsufficientQuotesError, RedundantQuoteWarning, RepricingError
from ._keys import _instrument_of
from ._plan import _Planner
from ._seeds import _Seeder
from ._solver import _GroupSolve, _solve_unknowns

# Curve-solve residuals are valued in one reporting currency; a par/MTM==0 residual
# is zero in any currency, so the choice is arbitrary — CLP by convention.
_REPORT_CURRENCY = Currency(CurrencyName.CLP)
# After the whole build every quote that pinned a pillar must reprice to this (report
# currency): a larger residual means a later group moved a curve it was fitted on.
_REPRICE_TOL = 1e-4


class _CurveBuilder:
    """One curve build: every index and discount curve, by one scheduler.

    Quotes are grouped by the curves they read; each quote owns at most one pillar.
    A group is solved once no other pending group still owns a pillar it reads, so
    every curve is complete where it is read; groups waiting on each other are
    merged and solved jointly. Curves go to ``market.curves`` under their ids.
    Mutates `market` in place.

    Parameters
    ----------
    quotes : list of InstrumentQuote
        Quotes to reprice.
    market : Market
        Market to build into; needs a `risk_free_index`.
    discount_interp_method : InterpolationMethod, optional
        Interpolation of the discount curves it creates (log-linear by default).
    index_interp_method : InterpolationMethod, optional
        Interpolation of the index curves it creates (piecewise-constant forwards
        by default).

    Raises
    ------
    ValueError
        If the market has no `risk_free_index`.
    """

    # Bootstrap each group date by date when its interpolation allows it; False always
    # runs one joint solve per group (the reference the date-by-date sweep must match).
    date_by_date: bool = True

    def __init__(
        self,
        quotes: list,
        market: Market,
        discount_interp_method: InterpolationMethod = InterpolationMethod.LogLinear,
        index_interp_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant,
    ):
        if market.risk_free_index is None:
            raise ValueError("Market has no risk_free_index; curve building needs one.")
        self.market = market
        self.t = market.t
        self.discount_interp_method = discount_interp_method
        self.index_interp_method = index_interp_method
        self.seeder = _Seeder(market, market.risk_free_index)
        self.plan = _Planner(quotes, market, self.seeder)
        self._pinning: list = []  # (instrument, report currency) of every quote that owned a pillar

    def build(self) -> None:
        """Solve every group in dependency order, then check the repricing.

        Raises
        ------
        InsufficientQuotesError
            If the remaining groups can neither be solved nor merged: they are
            under-determined or read a curve no quote builds.
        RepricingError
            If a quote that pinned a pillar no longer reprices at the end.

        Warns
        -----
        RedundantQuoteWarning
            Listing the quotes that owned no pillar, with their MTM.
        """
        plan = self.plan
        pending = [g for g in plan.groups if g]
        while pending:
            owned = {g: plan.owned_pillars(g) for g in pending}
            waits = {g: plan.waits(g, pending, owned) for g in pending}
            ready = (g for g in sorted(pending, key=plan.priority) if not waits[g])
            solved = next(
                (g for g in ready if self._solve_group(g, plan.instruments_of(g), candidates=plan.candidates_of(g))
                 in ("solved", "nothing")),
                None,
            )
            if solved is not None:
                pending.remove(solved)  # ownership changed: re-plan before the next solve
            elif not plan.merge_cycles(pending, waits):
                raise InsufficientQuotesError(
                    "Curve build stalled: the remaining groups are under-determined given "
                    f"the available quotes: {[set(g) for g in pending]}."
                )
        self._check_repricing()
        self._warn_ditched()

    def _solve_group(
        self, group, instruments, report_currency: Currency = _REPORT_CURRENCY, candidates: list | None = None
    ) -> str:
        """Solve the free pillars of `group` so the instruments owning them reprice.

        Parameters
        ----------
        group : frozenset of curve keys
            Curves the group reads.
        instruments : list
            Instruments of the group.
        report_currency : Currency, optional
            Currency the residuals are valued in.
        candidates : list of tuple, optional
            Candidate curves of each instrument; default: every curve of `group`.

        Returns
        -------
        str
            'solved'; 'nothing' when no instrument owns a new pillar; 'defer' when a
            curve it reads does not exist or the system is not square.
        """
        plan = self.plan
        if candidates is None:  # a single-curve solve: every instrument pins that curve
            candidates = [tuple(group)] * len(instruments)
        owners = plan.assign_pillars(instruments, candidates)
        fixed_pillars, free_dates, free_seeds = plan.partition_pillars(group, instruments, owners)
        if not free_dates:
            return "nothing"
        if not all(plan.exists(c) or c in free_dates for c in group):
            return "defer"  # it reads a curve that is not built and gets no pillar from it.

        residual_instruments, pillar_dates = plan.residual_instruments(instruments, free_dates, owners)
        solve = _GroupSolve.from_free_pillars(
            self.market, self.t, self.discount_interp_method, fixed_pillars, free_dates, free_seeds,
            self.index_interp_method,
        )
        if solve.n_unknowns != len(residual_instruments):
            return "defer"  # duplicate pillars: over-determined.

        y = _solve_unknowns(
            solve, residual_instruments, pillar_dates, report_currency,
            f"Curve solve for group {set(free_dates)}", date_by_date=self.date_by_date,
        )
        solve.apply(y)  # root's last f-eval is not guaranteed to be at the solution.
        self._pinning += [(i, report_currency) for i in residual_instruments]
        return "solved"

    def _check_repricing(self) -> None:
        """Raise if a quote that pinned a pillar was moved off par by a later group."""
        off = [
            (abs(v), i) for i, ccy in self._pinning
            if abs(v := Calculator.valuate(i, self.market, ccy)) > _REPRICE_TOL
        ]
        if off:
            worst, instr = max(off, key=lambda p: p[0])
            raise RepricingError(
                f"{len(off)} quote(s) no longer reprice after the build (worst |MTM| = {worst:.3g} "
                f"on a {type(instr).__name__}): a curve they were fitted on was changed afterwards."
            )

    def ditched_quotes(self) -> list[tuple]:
        """``(quote, mtm, mtm_pct, mtm_bp)`` of every quote that owned no pillar.

        See :class:`RedundantQuoteWarning` for the fields.
        """
        pinning = {id(i) for i, _ in self._pinning}
        out = []
        for quote, instr in self.plan.instrument.items():
            if id(instr) in pinning:
                continue
            mtm = Calculator.valuate(instr, self.market, _REPORT_CURRENCY)
            if isinstance(instr, Forward):  # no quoted rate: MTM against its notional
                notional = _forward_notional(instr, self.market, _REPORT_CURRENCY)
                out.append((quote, mtm, mtm / notional if notional else float("nan"), None))
            else:  # a swap: MTM in bp of its quoted rate
                dv01 = _quote_dv01(quote, mtm, self.market)
                out.append((quote, mtm, None, mtm / abs(dv01) if dv01 else None))
        return out

    def _warn_ditched(self) -> None:
        """Warn, once per build, about the quotes the curves ignore."""
        ditched = self.ditched_quotes()
        if not ditched:
            return
        lines = [
            f"  {_describe(q, self.plan.instrument[q], self.market)}: MTM {mtm:,.4f} {_REPORT_CURRENCY}"
            + (f" = {pct:.4%} of notional" if pct is not None else "")
            + (f" = {bp:+.4f} bp" if bp is not None else "")
            for q, mtm, pct, bp in ditched
        ]
        warnings.warn(
            RedundantQuoteWarning(
                f"{len(ditched)} quote(s) own no pillar and were not fitted (the curves they would "
                "pin are already built past their maturity by other quotes):\n" + "\n".join(lines),
                ditched,
            ),
            stacklevel=4,  # the caller of build_curves
        )

    def _bootstrap_projection_curve(self, index: Index, swaps: list[Swap]) -> None:
        """Solve the curve of `index` alone from `swaps`.

        The discount curves the swaps read must already be in the market.

        Raises
        ------
        InsufficientQuotesError
            If the swaps do not pin one distinct new segment each.
        """
        if self._solve_group(frozenset({index}), swaps, index.currency) != "solved":
            raise InsufficientQuotesError(
                f"Projection curve for '{index.name}' is under-determined by its swaps."
            )


def _forward_notional(forward: Forward, market: Market, currency: Currency) -> float:
    """Base-currency notional of `forward`, in `currency`."""
    amount, ccy = forward.notional, forward.currency_pair.base_currency
    return abs(amount if ccy == currency else Calculator._convert(amount, ccy, currency, market))


def _quote_dv01(quote, mtm: float, market: Market) -> float | None:
    """MTM change, in the report currency, for +1bp on the quote's quoted rate.

    The fixed rate of a fixed-float swap quote, the receive-leg spread of a
    float-float one; None for a quote that is not a swap (an FX forward).
    """
    bp = 1e-4
    if isinstance(fixed := getattr(quote, "fixed_leg", None), FixedLegSpec):
        bumped = dataclasses.replace(
            quote, fixed_leg=dataclasses.replace(fixed, rate=Rate(fixed.rate.rate_convention, fixed.rate.rate + bp))
        )
    elif isinstance(leg := getattr(quote, "receive_leg", None), FloatingLegSpec) and leg.spread is not None:
        bumped = dataclasses.replace(
            quote, receive_leg=dataclasses.replace(leg, spread=Rate(leg.spread.rate_convention, leg.spread.rate + bp))
        )
    else:
        return None
    return Calculator.valuate(_instrument_of(bumped, market), market, _REPORT_CURRENCY) - mtm


def _describe(quote, instrument, market: Market) -> str:
    """Short label of a quote: type, tenor (or payment date) and currencies.

    One currency (``CLP``) for a single-currency swap; two as a pair (``USD-CLP``)
    for a cross-currency swap or an FX forward: its base-quote order for a forward,
    the risk-free currency last for a swap.
    """
    when = getattr(quote, "term", None) or getattr(quote, "payment_date", "")
    if isinstance(instrument, Forward):
        currencies = [instrument.currency_pair.base_currency, instrument.currency_pair.quote_currency]
    else:
        local = getattr(market.risk_free_index, "currency", None)
        legs = {instrument.receive_leg.currency, instrument.pay_leg.currency}
        currencies = sorted(legs, key=lambda c: (c == local, str(c)))
    return f"{type(quote).__name__} {when} {'-'.join(map(str, currencies))}"


def build_curves(
    quotes: list,
    market: Market,
    *,
    discount_interp_method: InterpolationMethod = InterpolationMethod.LogLinear,
    index_interp_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant,
) -> None:
    """Bootstrap the curves that reprice `quotes` into `market`, in place.

    Every index a quote reads gets an :class:`IndexCurve`, under the index name;
    a currency other than the risk-free index's gets a :class:`DiscountCurve`,
    under ``<currency>_<risk-free index>``. Both go to ``market.curves``.

    Parameters
    ----------
    quotes : list of InstrumentQuote
        Quotes sharing one quote date, equal to ``market.t``.
    market : Market
        Market as of the quote date, with the FX and fixings valuation needs.
    discount_interp_method : InterpolationMethod, optional
        Interpolation of the discount curves this build creates: ``LogLinear``
        (default, solved date by date) or ``HermiteCubicSpline``. A curve already
        in the market keeps its own method when the build extends it.
    index_interp_method : InterpolationMethod, optional
        Interpolation of the index curves this build creates (``PiecewiseConstant``
        forwards by default); an extended one keeps its own.

    Raises
    ------
    ValueError
        If no quote has a quote date, the dates differ, or they differ from
        ``market.t``.
    InsufficientQuotesError
        If the quotes under-determine a curve.
    RepricingError
        If a quote that pinned a pillar does not reprice once every curve is built.

    Warns
    -----
    RedundantQuoteWarning
        When quotes were ditched (they own no pillar): each with its MTM and, for a
        swap, that MTM in basis points of its DV01; for an FX forward, as a
        percentage of its notional.
    """
    # Derive the valuation date from the quotes and cross-check market.t, since
    # Calculator.valuate reads FX (and other) market data off market.t internally.
    quote_dates = {q.quote_date for q in quotes if getattr(q, "quote_date", None) is not None}
    if not quote_dates:
        raise ValueError("No quotes with a quote_date were provided; cannot derive a valuation date.")
    if len(quote_dates) > 1:
        raise ValueError(f"All quotes must share the same quote_date; got {sorted(quote_dates)}.")
    t = next(iter(quote_dates))
    if t != market.t:
        raise ValueError(
            f"Quotes' quote_date ({t}) does not match market.t ({market.t}); market must be "
            "valued as of the same date as the quotes (Calculator.valuate reads FX and other "
            "data off market.t)."
        )

    _CurveBuilder(
        quotes, market, discount_interp_method=discount_interp_method, index_interp_method=index_interp_method
    ).build()
