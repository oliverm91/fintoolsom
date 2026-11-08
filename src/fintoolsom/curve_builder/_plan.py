"""Solver planning: which quote owns which pillar, and which group can solve when.

Quotes are grouped by the curves they read (:func:`_curves_needed`). Each quote owns
at most one pillar: the first open pillar among its candidate curves, at its
maturity (:meth:`_Planner.assign_pillars`). From that ownership:

* a group's solve layout follows (:meth:`_Planner.partition_pillars`): fixed
  pillars already built or anchored, one free pillar per owning quote;
* a group is ready when no other pending group still owns a pillar it reads
  (:meth:`_Planner.waits`), so a curve is never read before it is complete;
* groups waiting on each other form a cycle and are merged into one joint group
  (:meth:`_Planner.merge_cycles`).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from ..market.index import Index
from ..market.market import Market
from ..market.quotes import InstrumentQuote
from ._keys import CurveKey, _built, _curves_needed, _get_maturity, _instrument_of, _last_date, _primary_candidates
from ._seeds import _FLAT_GUESS, _IndexFixed, _Seeder

Group = frozenset  # of curve keys: the curves the group's quotes read


class _Planner:
    """Groups, pillar ownership and the solve order of one build.

    Parameters
    ----------
    quotes : list of InstrumentQuote
        Quotes of the build.
    market : Market
        Market being built; needs a `risk_free_index`.
    seeder : _Seeder
        Seeds and short-end anchors of the new curves.
    """

    def __init__(self, quotes: list[InstrumentQuote], market: Market, seeder: _Seeder):
        self.market = market
        self.t = market.t
        self.risk_free_index: Index = market.risk_free_index  # type: ignore[assignment]
        self.seeder = seeder
        self.pinned = {q: _curves_needed(q, self.risk_free_index) for q in quotes}
        self.groups: dict[Group, list[InstrumentQuote]] = defaultdict(list)
        for q in quotes:
            self.groups[self.pinned[q]].append(q)
        self._order = {g: i for i, g in enumerate(self.groups)}
        self.instrument = {q: _instrument_of(q, market) for q in quotes}
        self.candidates = self._resolve_candidates(quotes)

    def _resolve_candidates(self, quotes) -> dict:
        """Ordered candidate curves of each quote: its preferred curve, then fallbacks.

        A float-float basis swap prefers the index no single-candidate quote (an IRS)
        already prefers. A quote may fall back only to curves no quote prefers, so a
        redundant quote never takes a pillar that belongs to another quote type
        (e.g. a short XCCY on the SOFR curve of the OIS).
        """
        raw = {q: _primary_candidates(q, self.pinned[q]) for q in quotes}
        single = {c[0] for c in raw.values() if len(c) == 1}
        first = {q: next((k for k in c if k not in single), c[0]) for q, c in raw.items()}
        preferred = set(first.values())
        return {
            q: (first[q], *sorted((k for k in self.pinned[q] if k not in preferred), key=lambda k: (k not in raw[q], str(k))))
            for q in quotes
        }

    # -- groups ---------------------------------------------------------------

    def instruments_of(self, group: Group) -> list:
        return [self.instrument[q] for q in self.groups[group]]

    def candidates_of(self, group: Group) -> list[tuple]:
        return [self.candidates[q] for q in self.groups[group]]

    def priority(self, group: Group) -> tuple[int, int]:
        """Solve order among ready groups: fewer curves first, then quote order.

        It also settles which group owns a range of a curve two groups both extend
        (e.g. USD funding from FX forwards before the cross-currency swaps).
        """
        return len(group), self._order.get(group, len(self._order))

    def exists(self, curve: CurveKey) -> bool:
        """Whether the curve of `curve` is already in the market."""
        return _built(self.market, curve) is not None

    # -- pillar ownership -----------------------------------------------------

    def is_open(self, curve, d: date) -> bool:
        """Whether a pillar of `curve` at `d` would be new (curve unbuilt or `d` past its end)."""
        built = _built(self.market, curve)
        return built is None or d > built.last_date

    def assign_pillars(self, instruments, candidates) -> list:
        """Curve each instrument adds its one pillar to (None: redundant).

        The first candidate whose pillar at the instrument's maturity is open and not
        taken by another instrument. If every open one is taken the instrument
        duplicates a pillar: it keeps it, so the system is over-determined and fails.
        """
        owners, taken = [], set()
        for instr, cands in zip(instruments, candidates):
            open_ = [
                (c, m) for c in cands
                if (m := _get_maturity(c, instr, self.risk_free_index)) is not None and self.is_open(c, m)
            ]
            free = [p for p in open_ if p not in taken]
            pick = free[0] if free else (open_[0] if open_ else None)
            if pick is not None:
                taken.add(pick)
            owners.append(pick[0] if pick else None)
        return owners

    def owned_pillars(self, group: Group) -> set[tuple]:
        """``(curve, date)`` pillars the quotes of `group` would add now."""
        instruments = self.instruments_of(group)
        owners = self.assign_pillars(instruments, self.candidates_of(group))
        return {(c, _get_maturity(c, i, self.risk_free_index)) for i, c in zip(instruments, owners) if c is not None}

    def residual_instruments(self, instruments, free_dates, owners):
        """Instruments owning a free pillar, and that pillar's date.

        An instrument owns the pillar of its primary curve at its maturity; one with
        no open pillar (inside existing curves) is redundant.
        """
        out, dates = [], []
        for instr, curve in zip(instruments, owners):
            if curve is None:
                continue
            m = _get_maturity(curve, instr, self.risk_free_index)
            if m is not None and m in free_dates.get(curve, ()):
                out.append(instr)
                dates.append(m)
        return out, dates

    # -- readiness and cycles -------------------------------------------------

    def waits(self, group: Group, pending: list[Group], owned: dict[Group, set]) -> set[Group]:
        """Pending groups `group` must wait for.

        `group` reads every curve of its key up to its last date. It waits on H when
        H still owns a pillar on such a curve within that span, unless `group` itself
        extends that curve and goes first there: its own pillars start no later and
        it has priority. It also waits on the owners of a curve it reads that does
        not exist yet and gets no pillar from it.
        """
        mine = owned[group]
        first_own: dict = {}
        for c, d in mine:
            first_own[c] = min(d, first_own.get(c, d))
        last = max(_last_date(i) for i in self.instruments_of(group))
        owners: dict = defaultdict(list)  # curve it reads -> (other group, pillar date)
        for h in pending:
            if h != group:
                for c, d in owned[h]:
                    if c in group:
                        owners[c].append((h, d))
        out = set()
        for c, held in owners.items():
            within = [(h, d) for h, d in held if d <= last]
            if not within and not self.exists(c) and c not in first_own:
                out |= {h for h, _ in held}  # it must exist before it is read: wait for its builders
                continue
            out |= {
                h for h, d in within
                if c not in first_own or d < first_own[c] or self.priority(h) < self.priority(group)
            }
        return out

    def merge_cycles(self, pending: list[Group], waits: dict[Group, set]) -> bool:
        """Merge groups that wait on each other into one joint group, in place.

        Each strongly connected set of the `waits` graph is replaced, in `pending`
        and :attr:`groups`, by one group holding all their quotes; with one pillar
        per quote it is square. Returns False when there is no cycle.
        """
        def reach(g) -> set:
            seen, stack = set(), [g]
            while stack:
                for h in waits[stack.pop()]:
                    if h not in seen:
                        seen.add(h)
                        stack.append(h)
            return seen

        reachable = {g: reach(g) for g in pending}
        merged, done = False, set()
        for g in list(pending):
            cycle = {g} | {h for h in reachable[g] if g in reachable[h]}
            if g in done or len(cycle) < 2:
                continue
            done |= cycle
            key = frozenset().union(*cycle)
            order = min(self._order.get(h, len(self._order)) for h in cycle)
            quotes = [q for h in cycle for q in self.groups.pop(h)]
            for h in cycle:
                pending.remove(h)
            if key in pending:  # the union is another pending group's key: absorb it
                quotes += self.groups[key]
                pending.remove(key)
            self.groups[key] = quotes
            self._order[key] = order
            pending.append(key)
            merged = True
        return merged

    # -- solve layout ---------------------------------------------------------

    def partition_pillars(self, group, instruments, owners):
        """Split each curve's owned maturities into fixed and free pillars.

        Fixed: what is already solved, or a new curve's short-end anchor. Free: the
        new unknowns, past the fixed part. Seeds are read off the quotes in df space
        (see :meth:`_Seeder.seed_free_pillars`); an index curve's are turned into
        the forwards of its new segments.

        Parameters
        ----------
        group : frozenset of curve keys
            Curves the group reads.
        instruments : list
            The group's instruments.
        owners : list
            Curve each instrument owns a pillar on (None: none), see
            :meth:`assign_pillars`.

        Returns
        -------
        fixed_pillars : dict
            Curve key to its fixed part: ``(date, df)`` pillars for a currency's
            discount curve, an :class:`_IndexFixed` for an index curve.
        free_dates : dict
            Curve key to free pillar dates.
        free_seeds : dict
            Curve key to initial values of the free pillars: dfs for a discount
            curve, segment forwards for an index curve.
        """
        fixed_pillars: dict = {}
        free_dates: dict[CurveKey, list[date]] = {}
        flat_guess: dict[CurveKey, float] = {}
        for curve in group:
            maturities = {
                _get_maturity(curve, instr, self.risk_free_index)
                for instr, owner in zip(instruments, owners) if owner == curve
            }
            maturities.discard(None)
            if not maturities:
                continue
            built = _built(self.market, curve)
            if isinstance(curve, Index):
                if built is None:
                    fixed, flat_guess[curve] = self.seeder.index_anchor(curve)
                else:
                    fixed = _IndexFixed(list(built.knot_dates), [float(f) for f in built.forward_rates], built.spot_rate)
                fixed_pillars[curve] = fixed
                if new_dates := sorted(m for m in maturities if m > fixed.knots[-1]):
                    free_dates[curve] = new_dates
            elif built is None:
                fixed_pillars[curve], flat_guess[curve] = [], _FLAT_GUESS
                free_dates[curve] = sorted(maturities)
            else:  # extend an existing discount curve past its last pillar
                fixed_pillars[curve] = built.pillars
                if new_dates := sorted(m for m in maturities if m > built.last_date):
                    free_dates[curve] = new_dates
        points = {
            c: f.knot_dfs(self.t, c) if isinstance(c, Index) else f for c, f in fixed_pillars.items()
        }
        seed_dfs = self.seeder.seed_free_pillars(instruments, points, free_dates, flat_guess)
        free_seeds = {
            c: fixed_pillars[c].forwards_to(self.t, c, dates, seed_dfs[c]) if isinstance(c, Index) else seed_dfs[c]
            for c, dates in free_dates.items()
        }
        return fixed_pillars, free_dates, free_seeds
