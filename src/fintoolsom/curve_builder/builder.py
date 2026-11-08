"""Curve bootstrap from market quotes: :func:`build_curves`.

The build is split in private modules:

* :mod:`._keys`   — data arrangement: the curves each quote reads, its pillar dates;
* :mod:`._plan`   — planning: pillar ownership, solve layout, readiness, cycles;
* :mod:`._seeds`  — initial guesses and short-end anchors;
* :mod:`._solver` — the unknown layout, the atomic root, the date-by-date bootstrap.

:class:`_CurveBuilder` drives them: it solves one ready group at a time, merges
groups that wait on each other, and finally checks that every quote that pinned a
pillar still reprices.
"""
from __future__ import annotations

from ..derivatives.calculator import Calculator
from ..derivatives.swaps import Swap
from ..market.currencies import Currency, CurrencyName
from ..market.index import Index
from ..market.market import Market
from ..rates import InterpolationMethod, ProjectionInterpolationMethod
from ._errors import InsufficientQuotesError, RepricingError
from ._keys import _Proj
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
    """One curve build: every discount and projection curve, by one scheduler.

    Quotes are grouped by the curves they read; each quote owns at most one pillar.
    A group is solved once no other pending group still owns a pillar it reads, so
    every curve is complete where it is read; groups waiting on each other are
    merged and solved jointly. Discount curves go to ``market.curves``, the forward
    curves of projection-only indexes to ``market.projection_curves``. Mutates
    `market` in place.

    Parameters
    ----------
    quotes : list of InstrumentQuote
        Quotes to reprice.
    market : Market
        Market to build into; needs a `riskless_index`.
    discount_method : InterpolationMethod
        Interpolation of the discount curves.
    projection_method : ProjectionInterpolationMethod
        Interpolation of the projection curves.

    Raises
    ------
    ValueError
        If the market has no `riskless_index`.
    """

    # Bootstrap each group date by date when its interpolation allows it; False always
    # runs one joint solve per group (the reference the date-by-date sweep must match).
    date_by_date: bool = True

    def __init__(
        self,
        quotes: list,
        market: Market,
        discount_method: InterpolationMethod,
        projection_method: ProjectionInterpolationMethod,
    ):
        if market.riskless_index is None:
            raise ValueError("Market has no riskless_index; curve building needs one.")
        self.market = market
        self.t = market.t
        self.discount_method = discount_method
        self.projection_method = projection_method
        self.seeder = _Seeder(market, market.riskless_index)
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
        fixed_pillars, free_dates, free_seed_dfs = plan.partition_pillars(group, instruments, owners)
        if not free_dates:
            return "nothing"
        if not set(group) <= plan.existing_keys() | set(free_dates):
            return "defer"  # it reads a curve that is not built and gets no pillar from it.

        residual_instruments, pillar_dates = plan.residual_instruments(instruments, free_dates, owners)
        solve = _GroupSolve.from_free_pillars(
            self.market, self.t, self.discount_method, fixed_pillars, free_dates, free_seed_dfs,
            self.projection_method,
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

    def _bootstrap_projection_curve(self, index: Index, swaps: list[Swap]) -> None:
        """Solve the projection curve of `index` alone from `swaps`.

        The discount curves the swaps read must already be in the market.

        Raises
        ------
        InsufficientQuotesError
            If the swaps do not pin one distinct new segment each.
        """
        if self._solve_group(frozenset({_Proj(index)}), swaps, index.currency) != "solved":
            raise InsufficientQuotesError(
                f"Projection curve for '{index.name}' is under-determined by its swaps."
            )


def build_curves(
    quotes: list,
    market: Market,
    *,
    interpolation_method: InterpolationMethod | None = None,
) -> None:
    """Bootstrap the curves that reprice `quotes` into `market`, in place.

    Discount curves go to ``market.curves`` (keyed by Index, or by a non-riskless
    Currency) and projection-only indexes get a curve in
    ``market.projection_curves``.

    Parameters
    ----------
    quotes : list of InstrumentQuote
        Quotes sharing one quote date, equal to ``market.t``.
    market : Market
        Market as of the quote date, with the FX and fixings valuation needs.
    interpolation_method : InterpolationMethod, optional
        Discount-curve interpolation for this build; defaults to
        ``market.discount_interpolation_method``.

    Raises
    ------
    ValueError
        If no quote has a quote date, the dates differ, or they differ from
        ``market.t``.
    InsufficientQuotesError
        If the quotes under-determine a curve.
    RepricingError
        If a quote that pinned a pillar does not reprice once every curve is built.
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
        quotes,
        market,
        discount_method=interpolation_method or market.discount_interpolation_method,
        projection_method=market.projection_interpolation_method,
    ).build()
