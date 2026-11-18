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
import math
import warnings
from collections import defaultdict
from datetime import date

from ..derivatives.calculator import Calculator
from ..derivatives.forwards import Forward
from ..derivatives.swaps import Swap
from ..market.currencies import Currency, CurrencyName
from ..market.index import Index
from ..market.conventions import FixedLegSpec, FloatingLegSpec
from ..market.market import Market
from ..rates import InterpolationMethod, Rate
from ..rates.PillarCurve import MIN_KNOTS
from ._errors import InsufficientQuotesError, RedundantQuoteWarning, RepricingError
from ._keys import CurveKey, _built, _curve_id, _instrument_of
from ._plan import _Planner
from ._seeds import _Seeder
from ._solver import _LOCAL_METHODS, _SWEEP_TOL, _GroupSolve, _solve_unknowns

# Defaults of build_curves. Curve-solve residuals are valued in one reporting
# currency; a par/MTM==0 residual is zero in any currency, so the choice is arbitrary:
# CLP by convention.
_REPORT_CURRENCY = Currency(CurrencyName.CLP)
# After the whole build every quote that pinned a pillar must reprice to this (report
# currency): a larger residual means a later group moved a curve it was fitted on.
_REPRICE_TOL = 1e-2
# ... and the |MTM| of the fitted quotes of each solve group must sum to at most this.
_GROUP_TOL = 1.0


class _CurveBuilder:
    """One curve build: every index and discount curve, by one scheduler.

    Quotes are grouped by the curves they read; each quote owns at most one pillar.
    A group is solved once no other pending group still owns a pillar it reads, so
    every curve is complete where it is read; groups waiting on each other are
    merged and solved jointly. Curves go to ``market.curves`` under their keys.
    Mutates `market` in place.

    Parameters
    ----------
    quotes : list of InstrumentQuote
        Quotes to reprice.
    market : Market
        Market to build into; needs a `risk_free_index`.
    interp_method : InterpolationMethod, optional
        Interpolation of the discount and index curves it creates (``PiecewiseConstant``
        by default).
    report_currency, reprice_tol, sweep_tol, group_tol : optional
        As in :func:`build_curves`.

    Raises
    ------
    ValueError
        If the market has no `risk_free_index`, or a tolerance is not positive.
    """

    # Bootstrap each group date by date when its interpolation allows it; False always
    # runs one joint solve per group (the reference the date-by-date sweep must match).
    date_by_date: bool = True

    def __init__(
        self,
        quotes: list,
        market: Market,
        interp_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant,
        *,
        report_currency: Currency = _REPORT_CURRENCY,
        reprice_tol: float = _REPRICE_TOL,
        sweep_tol: float = _SWEEP_TOL,
        group_tol: float = _GROUP_TOL,
    ):
        if market.risk_free_index is None:
            raise ValueError("Market has no risk_free_index; curve building needs one.")
        for name, tol in (
            ("reprice_tol", reprice_tol),
            ("sweep_tol", sweep_tol),
            ("group_tol", group_tol),
        ):
            if not (isinstance(tol, (int, float)) and math.isfinite(tol) and tol > 0):
                raise ValueError(
                    f"{name} must be a positive finite number; got {tol!r}."
                )
        self.market = market
        self.t = market.t
        self.interp_method = interp_method
        self.report_currency = report_currency
        self.reprice_tol = float(reprice_tol)
        self.sweep_tol = float(sweep_tol)
        self.group_tol = float(group_tol)
        self.seeder = _Seeder(market, market.risk_free_index)
        self.plan = _Planner(quotes, market, self.seeder)
        self._pinning: list = []  # (instrument, report currency) of every quote that owned a pillar
        self._pinned_dates: list[date] = []  # the pillar each of `_pinning` owns
        self._pinned_groups: list = []  # the group each of `_pinning` was solved in
        # (instruments, report currency, the solver's batch of them) per solved group,
        # in `_pinning` order: the repricing check reuses the batches.
        self._pinned_batches: list = []
        self._fixed: dict = {}  # curve -> its fixed part before this build's first pillar
        self._free_dates: dict[CurveKey, list[date]] = defaultdict(list)

    def build(self) -> None:
        """Solve every group in dependency order, then check the repricing.

        Raises
        ------
        InsufficientQuotesError
            If the remaining groups can neither be solved nor merged: they are
            under-determined or read a curve no quote builds.
        RepricingError
            If a quote that pinned a pillar no longer reprices at the end, or the
            fitted quotes of a group add up to more than group_tol.

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
            ready_set = {g for g in pending if not waits[g]}
            # A group reading a curve past its end goes after the ready group that
            # extends it there, so the extension does not move its quotes afterwards.
            ready = (
                g
                for g in sorted(
                    ready_set,
                    key=lambda g: (
                        bool(plan.tail_waits(g, pending, owned) & ready_set),
                        plan.priority(g),
                    ),
                )
            )
            solved = next(
                (
                    g
                    for g in ready
                    if self._solve_group(
                        g, plan.instruments_of(g), candidates=plan.candidates_of(g)
                    )
                    in ("solved", "nothing")
                ),
                None,
            )
            if solved is not None:
                pending.remove(
                    solved
                )  # ownership changed: re-plan before the next solve
            elif not plan.merge_cycles(pending, waits):
                raise InsufficientQuotesError(
                    "Curve build stalled: the remaining groups are under-determined given "
                    f"the available quotes: {[set(g) for g in pending]}."
                )
        self._solve_splines_jointly()
        self._precalculate_built_curves()
        self._check_repricing()
        self._warn_ditched()

    def _precalculate_built_curves(self) -> None:
        """Give the curves of this build the daily grid the solve skipped."""
        for key in self._free_dates:
            curve = _built(self.market, key)
            if not curve.precalculate:
                curve.precalculate_dfs()

    def _solve_group(
        self,
        group,
        instruments,
        report_currency: Currency | None = None,
        candidates: list | None = None,
    ) -> str:
        """Solve the free pillars of `group` so the instruments owning them reprice.

        Parameters
        ----------
        group : frozenset of curve keys
            Curves the group reads.
        instruments : list
            Instruments of the group.
        report_currency : Currency, optional
            Currency the residuals are valued in; the build's by default.
        candidates : list of tuple, optional
            Candidate curves of each instrument; default: every curve of `group`.

        Returns
        -------
        str
            'solved'; 'nothing' when no instrument owns a new pillar; 'defer' when a
            curve it reads does not exist or the system is not square.
        """
        plan = self.plan
        if report_currency is None:
            report_currency = self.report_currency
        if candidates is None:  # a single-curve solve: every instrument pins that curve
            candidates = [tuple(group)] * len(instruments)
        owners = plan.assign_pillars(instruments, candidates)
        fixed_pillars, free_dates, free_seeds = plan.partition_pillars(
            group, instruments, owners
        )
        if not free_dates:
            return "nothing"
        if not all(plan.exists(c) or c in free_dates for c in group):
            return "defer"  # it reads a curve that is not built and gets no pillar from it.

        residual_instruments, pillar_dates = plan.residual_instruments(
            instruments, free_dates, owners
        )
        solve = _GroupSolve.from_free_pillars(
            self.market,
            self.t,
            self.interp_method,
            fixed_pillars,
            free_dates,
            free_seeds,
        )
        if solve.n_unknowns != len(residual_instruments):
            return "defer"  # duplicate pillars: over-determined.
        for curve, dates in free_dates.items():
            if len(fixed_pillars[curve].knots) + len(dates) < MIN_KNOTS:
                raise InsufficientQuotesError(
                    f"Curve {_curve_id(curve, self.market.risk_free_index)} would have "
                    "a single segment: a curve needs at least 2 pillars, so it needs "
                    "more quotes in this build."
                )

        batches: dict = {}
        y = _solve_unknowns(
            solve,
            residual_instruments,
            pillar_dates,
            report_currency,
            f"Curve solve for group {set(free_dates)}",
            date_by_date=self.date_by_date,
            sweep_tol=self.sweep_tol,
            batches=batches,
        )
        solve.apply(y)  # root's last f-eval is not guaranteed to be at the solution.
        self._pinning += [(i, report_currency) for i in residual_instruments]
        self._pinned_dates += pillar_dates
        self._pinned_groups += [group] * len(residual_instruments)
        self._pinned_batches.append(
            (
                residual_instruments,
                report_currency,
                batches.get(id(residual_instruments)),
            )
        )
        for curve, dates in free_dates.items():
            self._fixed.setdefault(curve, fixed_pillars[curve])
            self._free_dates[curve] += dates
        return "solved"

    def _solve_splines_jointly(self) -> None:
        """Re-solve every pillar of this build at once if a curve's method is not local.

        Groups are solved in sequence on the premise that a later pillar leaves what
        was already fitted alone. A pillar of a non-local method moves the curve before
        it too, so the quotes fitted earlier drift; the sequential result only
        warm-starts this solve.
        """
        built = {c: _built(self.market, c) for c in self._free_dates}
        if all(
            curve.interpolation_method in _LOCAL_METHODS for curve in built.values()
        ):
            return
        # New pillars are always appended, so each curve's are its last values.
        seeds = {
            c: built[c].forward_rates[-len(dates) :]
            for c, dates in self._free_dates.items()
        }
        solve = _GroupSolve.from_free_pillars(
            self.market,
            self.t,
            self.interp_method,
            self._fixed,
            dict(self._free_dates),
            seeds,
        )
        y = _solve_unknowns(
            solve,
            [instrument for instrument, _ in self._pinning],
            self._pinned_dates,
            self.report_currency,
            "Joint solve of the cubic-spline curves",
            date_by_date=False,
        )
        solve.apply(y)

    def _pinned_mtms(self) -> list[float]:
        """MTM of each quote of `_pinning` on the curves as they are now.

        From the solver's batches, which give to the bit what Calculator.valuate
        gives; a group without one is valued instrument by instrument.
        """
        mtms: list[float] = []
        for instruments, ccy, batch in self._pinned_batches:
            if batch is None:
                mtms += [Calculator.valuate(i, self.market, ccy) for i in instruments]
            else:
                mtms += batch.values().tolist()
        return mtms

    def _check_repricing(self) -> None:
        """Raise if a quote that pinned a pillar was moved off par by a later group, or
        the fitted quotes of a group add up to more than `group_tol`."""
        mtms = [abs(v) for v in self._pinned_mtms()]
        off = [(v, i) for v, (i, _) in zip(mtms, self._pinning) if v > self.reprice_tol]
        if off:
            worst, instr = max(off, key=lambda p: p[0])
            raise RepricingError(
                f"{len(off)} quote(s) no longer reprice after the build (worst |MTM| = {worst:.3g} "
                f"on a {type(instr).__name__}): a curve they were fitted on was changed afterwards."
            )
        totals: dict = defaultdict(float)
        for v, group in zip(mtms, self._pinned_groups):
            totals[group] += v
        over = {g: total for g, total in totals.items() if total > self.group_tol}
        if over:
            group, total = max(over.items(), key=lambda gt: gt[1])
            rfi = self.market.risk_free_index
            curves = ", ".join(sorted(_curve_id(c, rfi) for c in group))
            raise RepricingError(
                f"{len(over)} group(s) of quotes do not reprice after the build: the "
                f"fitted quotes on {{{curves}}} sum to |MTM| = {total:.3g} "
                f"{self.report_currency}, above group_tol = {self.group_tol:g}."
            )

    def ditched_quotes(self) -> list[tuple]:
        """``(quote, mtm, mtm_pct, mtm_bp)`` of every quote that owned no pillar.

        See :class:`RedundantQuoteWarning` for the fields.
        """
        pinning = {id(i) for i, _ in self._pinning}
        ccy = self.report_currency
        out = []
        for quote, instr in self.plan.instrument.items():
            if id(instr) in pinning:
                continue
            mtm = Calculator.valuate(instr, self.market, ccy)
            if isinstance(instr, Forward):  # no quoted rate: MTM against its notional
                notional = _forward_notional(instr, self.market, ccy)
                out.append(
                    (quote, mtm, mtm / notional if notional else float("nan"), None)
                )
            else:  # a swap: MTM in bp of its quoted rate
                dv01 = _quote_dv01(quote, mtm, self.market, ccy)
                out.append((quote, mtm, None, mtm / abs(dv01) if dv01 else None))
        return out

    def _warn_ditched(self) -> None:
        """Warn, once per build, about the quotes the curves ignore."""
        ditched = self.ditched_quotes()
        if not ditched:
            return
        lines = [
            f"  {_describe(q, self.plan.instrument[q], self.market)}: MTM {mtm:,.4f} {self.report_currency}"
            + (f" = {pct:.4%} of notional" if pct is not None else "")
            + (f" = {bp:+.4f} bp" if bp is not None else "")
            for q, mtm, pct, bp in ditched
        ]
        warnings.warn(
            RedundantQuoteWarning(
                f"{len(ditched)} quote(s) own no pillar and were not fitted (the curves they would "
                "pin are already built past their maturity by other quotes):\n"
                + "\n".join(lines),
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
    return abs(
        amount
        if ccy == currency
        else market.convert_value_currency(amount, ccy, currency)
    )


def _quote_dv01(quote, mtm: float, market: Market, currency: Currency) -> float | None:
    """MTM change, in `currency` (the one `mtm` is in), for +1bp on the quote's
    quoted rate.

    The fixed rate of a fixed-float swap quote, the receive-leg spread of a
    float-float one; None for a quote that is not a swap (an FX forward).
    """
    bp = 1e-4
    if isinstance(fixed := getattr(quote, "fixed_leg", None), FixedLegSpec):
        bumped = dataclasses.replace(
            quote,
            fixed_leg=dataclasses.replace(
                fixed, rate=Rate(fixed.rate.rate_convention, fixed.rate.rate + bp)
            ),
        )
    elif (
        isinstance(leg := getattr(quote, "receive_leg", None), FloatingLegSpec)
        and leg.spread is not None
    ):
        bumped = dataclasses.replace(
            quote,
            receive_leg=dataclasses.replace(
                leg, spread=Rate(leg.spread.rate_convention, leg.spread.rate + bp)
            ),
        )
    else:
        return None
    return Calculator.valuate(_instrument_of(bumped, market), market, currency) - mtm


def _describe(quote, instrument, market: Market) -> str:
    """Short label of a quote: type, tenor (or payment date) and currencies.

    One currency (``CLP``) for a single-currency swap; two as a pair (``USD-CLP``)
    for a cross-currency swap or an FX forward: its base-quote order for a forward,
    the risk-free currency last for a swap.
    """
    when = getattr(quote, "term", None) or getattr(quote, "payment_date", "")
    if isinstance(instrument, Forward):
        currencies = [
            instrument.currency_pair.base_currency,
            instrument.currency_pair.quote_currency,
        ]
    else:
        local = getattr(market.risk_free_index, "currency", None)
        legs = {instrument.receive_leg.currency, instrument.pay_leg.currency}
        currencies = sorted(legs, key=lambda c: (c == local, str(c)))
    return f"{type(quote).__name__} {when} {'-'.join(map(str, currencies))}"


def build_curves(
    quotes: list,
    market: Market,
    *,
    interp_method: InterpolationMethod = InterpolationMethod.PiecewiseConstant,
    report_currency: Currency = _REPORT_CURRENCY,
    reprice_tol: float = _REPRICE_TOL,
    sweep_tol: float = _SWEEP_TOL,
    group_tol: float = _GROUP_TOL,
) -> None:
    """Bootstrap the curves that reprice `quotes` into `market`, in place.

    Every index a quote reads gets a :class:`PillarCurve` under the index; a
    currency other than the risk-free index's gets one under the currency. Both go
    to ``market.curves``.

    Parameters
    ----------
    quotes : list of InstrumentQuote
        Quotes sharing one quote date, equal to ``market.t``.
    market : Market
        Market as of the quote date, with the FX and fixings valuation needs.
    interp_method : InterpolationMethod, optional
        Interpolation of the discount and index curves this build creates:
        ``PiecewiseConstant`` forwards by default, or any other pillar-curve method.
        A curve already in the market keeps its own method when the build extends it.
        With a ``CubicSpline`` curve every pillar is finally solved jointly, since
        a spline pillar moves the whole curve.
    report_currency : Currency, optional
        Currency the quotes' MTMs are valued in: the solve residuals, the repricing
        check and the MTMs of :class:`RedundantQuoteWarning` (CLP by default). The
        market needs the FX to it from every currency the quotes pay in.
    reprice_tol : float, optional
        Largest |MTM|, in `report_currency`, a quote that pinned a pillar may have
        once every curve is built (``1e-2`` by default); above it,
        :class:`RepricingError`. Absolute, so it depends on the quotes' notionals and
        on the currency.
    sweep_tol : float, optional
        Largest |MTM|, in `report_currency`, a date-by-date sweep of a group with
        piecewise-constant or piecewise-linear curves may leave on its quotes
        (``1e-4`` by default); above it the group is solved again jointly, warm-started
        from the sweep. It does not loosen the solves themselves. Keep it below
        `reprice_tol`: a sweep accepted above it fails the repricing check.
    group_tol : float, optional
        Largest sum of |MTM|, in `report_currency`, of the quotes that pinned a pillar
        in one solve group, once every curve is built (``1`` by default); above it,
        :class:`RepricingError`. Quotes that owned no pillar are not counted: they
        are only reported, in :class:`RedundantQuoteWarning`.

    Raises
    ------
    ValueError
        If no quote has a quote date, the dates differ, or they differ from
        ``market.t``; or `reprice_tol`, `sweep_tol` or `group_tol` is not a positive
        finite number.
    InsufficientQuotesError
        If the quotes under-determine a curve.
    RepricingError
        If a quote that pinned a pillar does not reprice once every curve is built
        (`reprice_tol`), or the quotes of a solve group do not together (`group_tol`).

    Warns
    -----
    RedundantQuoteWarning
        When quotes were ditched (they own no pillar): each with its MTM and, for a
        swap, that MTM in basis points of its DV01; for an FX forward, as a
        percentage of its notional.
    """
    # Derive the valuation date from the quotes and cross-check market.t, since
    # Calculator.valuate reads FX (and other) market data off market.t internally.
    quote_dates = {q.quote_date for q in quotes}
    if not quote_dates:
        raise ValueError("No quotes were provided; cannot derive a valuation date.")
    if len(quote_dates) > 1:
        raise ValueError(
            f"All quotes must share the same quote_date; got {sorted(quote_dates)}."
        )
    t = next(iter(quote_dates))
    if t != market.t:
        raise ValueError(
            f"Quotes' quote_date ({t}) does not match market.t ({market.t}); market must be "
            "valued as of the same date as the quotes (Calculator.valuate reads FX and other "
            "data off market.t)."
        )

    _CurveBuilder(
        quotes,
        market,
        interp_method=interp_method,
        report_currency=report_currency,
        reprice_tol=reprice_tol,
        sweep_tol=sweep_tol,
        group_tol=group_tol,
    ).build()
