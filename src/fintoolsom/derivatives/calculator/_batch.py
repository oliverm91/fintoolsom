"""Many instruments valued at once on one market, the curves read once per call.

:class:`ValuationBatch` is compiled once for a list of instruments and a market, then
:meth:`ValuationBatch.values` gives every MTM, as often as the market's curves change
(the curve builder's solver calls it once per iteration). Each value is, to the bit,
what :meth:`Calculator.valuate` gives for that instrument alone:

* Swaps (``Swap`` itself, legs settled in their own currency or not) and NDFs
  are compiled: what depends only on the market's date, histories and risk-free index
  (which coupons are unpaid, which legs are valued at par, the flows of fixed legs, the
  spot conversion) is computed once, and each call evaluates the curves once per
  source (a discounting, an index curve, a projection or an FX pair) on every date the
  batch needs from it. The flows, FX and discount factors are then multiplied
  elementwise and each leg summed with :func:`math.fsum` on its own slice: the
  arithmetic of the calculator, so the same values.
* Anything else (another instrument type, a coupon already running or fixed) is
  valued by :meth:`Calculator.valuate`.

The batch assumes only the curves change between calls: not the instruments, nor the
market's date, FX or index histories or risk-free index. A call that raises anything
is redone instrument by instrument, so the error raised is the calculator's.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ...market.currencies import CurrencyPair
from ..forwards.forwards import NDF
from ..swaps.legs import (
    FixedLeg,
    FloatingLeg,
    TermRateLeg,
    _interest_from_window,
    _interest_from_window_slope,
)
from ..swaps.swaps import Swap
from ._swap_pv import _discounted_on_index_curve, _nothing_realised

if TYPE_CHECKING:
    from ...market import Market
    from ...market.currencies import Currency

__all__ = ["ValuationBatch"]


class _Source:
    """One curve lookup of the market, evaluated once per call on every date the batch
    asks of it (``_dfs_at_ordinals`` and its kin are elementwise)."""

    __slots__ = ("kind", "args", "chunks", "size", "ordinals")

    def __init__(self, kind: str, args: tuple):
        self.kind, self.args = kind, args
        self.chunks: list = []
        self.size = 0

    def add(self, *ordinals: np.ndarray) -> tuple[int, int]:
        start = self.size
        self.chunks.append(ordinals)
        self.size += len(ordinals[0])
        return start, len(ordinals[0])

    def freeze(self) -> None:
        self.ordinals = tuple(
            np.concatenate([c[k] for c in self.chunks]).astype(np.int64, copy=False)
            for k in range(len(self.chunks[0]))
        )
        self.chunks = []

    def evaluate(self, market: Market) -> np.ndarray:
        kind, args = self.kind, self.args
        if kind == "discount":  # Market._dfs_at_ordinals
            currency, collateral = args
            return market._dfs_at_ordinals(
                currency, *self.ordinals, collateral=collateral
            )
        if kind == "index":  # the index curve's dfs (_par_leg_pv)
            return market.get_index_curve(args[0])._dfs_at_ordinals(*self.ordinals)
        if kind == "accrual":  # Market._accrual_at_ordinals (_projected_flows)
            return market._accrual_at_ordinals(args[0], *self.ordinals)
        return market._future_fx_rates_at_ordinals(args[0], *self.ordinals)  # "fx"


class _Uncompiled(Exception):
    """An instrument left to Calculator.valuate."""


class _Leg:
    """One compiled leg: its terms ``flow * fx * df`` are ``[start, stop)`` of the
    batch's products, plus the head ``residual_0 * df(start_0)`` of a par leg; its PV
    is converted to the reporting currency at ``spot``."""

    __slots__ = ("start", "stop", "head", "spot")

    def __init__(self, start: int, stop: int, head: int | None, spot: float):
        self.start, self.stop, self.head, self.spot = start, stop, head, spot


class _Ndf:
    """One compiled NDF: ``a * (s_t - strike) * df * spot``, ``s_t`` fixed or the
    batch's FX value ``fx``."""

    __slots__ = ("a", "strike", "s_t", "fx", "df", "spot")

    def __init__(self, a, strike, s_t, fx, df, spot):
        self.a, self.strike, self.s_t, self.fx, self.df, self.spot = (
            a,
            strike,
            s_t,
            fx,
            df,
            spot,
        )


class ValuationBatch:
    """The MTMs of `instruments` on `market` in `currency`, compiled once.

    Parameters
    ----------
    instruments : list
        Instruments :meth:`Calculator.valuate` values.
    market : Market
        Market they are valued on; only its curves may change between calls.
    currency : Currency
        Reporting currency.
    """

    def __init__(self, instruments, market: Market, currency: Currency):
        self.instruments = list(instruments)
        self.market, self.currency = market, currency
        self._sources: dict = {}
        self._spots: dict = {}  # spot rate by settlement currency (identity)
        self._terms: list = []  # (flows, fx refs, df refs) per leg, before freezing
        self._projected: list = []  # (positions in the flows, residuals, fixed, refs)
        self._heads: list = []  # (residual_0, df ref) per par leg
        self._compiled: list = []  # (position, plan) per compiled instrument
        self._fallback: list = []  # positions left to Calculator.valuate
        self._n_terms = 0
        for k, instrument in enumerate(self.instruments):
            checkpoint = self._checkpoint()
            try:
                self._compiled.append((k, self._compile(instrument)))
            except Exception:  # _Uncompiled, or an error the calculator will raise
                self._rollback(checkpoint)
                self._fallback.append(k)
        self._freeze()

    def _checkpoint(self) -> tuple:
        return (
            len(self._terms),
            len(self._projected),
            len(self._heads),
            self._n_terms,
            {key: (len(s.chunks), s.size) for key, s in self._sources.items()},
        )

    def _rollback(self, checkpoint: tuple) -> None:
        """Drop what a failed compilation added."""
        n_terms, n_projected, n_heads, self._n_terms, sources = checkpoint
        del self._terms[n_terms:], self._projected[n_projected:], self._heads[n_heads:]
        for key in [k for k in self._sources if k not in sources]:
            del self._sources[key]
        for key, (n_chunks, size) in sources.items():
            source = self._sources[key]
            del source.chunks[n_chunks:]
            source.size = size

    # ── Compilation ────────────────────────────────────────────────────────────

    def _source(self, kind: str, *args) -> _Source:
        # Indexes and currencies by identity (equal objects at worst split a source);
        # a pair by value, as the market keys its FX histories (one is built per leg).
        key = (kind, args[0]) if kind == "fx" else (kind, *(id(a) for a in args))
        source = self._sources.get(key)
        if source is None:
            source = self._sources[key] = _Source(kind, args)
        return source

    def _ref(self, source: _Source, *ordinals: np.ndarray) -> tuple:
        return (source, *source.add(*ordinals))

    def _spot(self, settlement) -> float:
        """Market.convert_value_currency's rate, ``value * rate`` (once per currency)."""
        spot = self._spots.get(id(settlement))
        if spot is None:
            market = self.market
            spot = self._spots[id(settlement)] = market.get_fx_rate(
                market.t, CurrencyPair(settlement, self.currency)
            ).value
        return spot

    def _compile(self, instrument):
        if type(instrument) is NDF:
            return self._compile_ndf(instrument)
        if type(instrument) is not Swap:
            raise _Uncompiled
        receive, pay = instrument.receive_leg, instrument.pay_leg
        collateral = instrument.collateral_index
        return tuple(
            self._compile_leg(leg, collateral, instrument.leg_settlement_currency(leg))
            for leg in (receive, pay)
        )

    def _compile_leg(self, leg, collateral, settlement) -> _Leg:
        """_swap_pv._leg_pv_non_deliverable, its static part."""
        market = self.market
        t = market.t.toordinal()
        start = self._n_terms
        spot = self._spot(settlement)
        if leg.currency == settlement:  # _leg_pv
            if (
                isinstance(leg, FloatingLeg)
                and leg._par_eligible
                and _nothing_realised(leg, t)
                and _discounted_on_index_curve(leg, collateral, market)
            ):  # _par_leg_pv
                index = self._source("index", leg.index)
                self._heads.append(
                    (leg.residuals[0], self._ref(index, leg._start_ordinals[:1]))
                )
                head = len(self._heads) - 1
                if leg._has_spread:
                    self._add_terms(
                        leg.spreads_values, None, self._ref(index, leg._end_ordinals)
                    )
                return _Leg(start, self._n_terms, head, spot)
            future = self._unpaid_flows(leg, t)
            if future is None:
                return _Leg(start, start, None, spot)
            payment = leg._payment_ordinals[future]
            discount = self._ref(
                self._source("discount", leg.currency, collateral), payment
            )
            self._add_terms(self._flows, None, discount)
            return _Leg(start, self._n_terms, None, spot)
        future = self._unpaid_flows(leg, t)
        if future is None:
            return _Leg(start, start, None, spot)
        pair = CurrencyPair(leg.currency, settlement)
        fx = self._ref(self._source("fx", pair), leg._fx_fixing_ordinals[future])
        payment = leg._payment_ordinals[future]
        discount = self._ref(self._source("discount", settlement, collateral), payment)
        self._add_terms(self._flows, fx, discount)
        return _Leg(start, self._n_terms, None, spot)

    def _unpaid_flows(self, leg, t: int):
        """_swap_pv._leg_flows when nothing is realised: the mask of the unpaid coupons
        (None if there is none), their flows left in ``self._flows`` (projected ones
        filled in by each call)."""
        future = leg._payment_ordinals > t
        if not future.any():
            return None
        if isinstance(leg, FixedLeg):
            self._flows = leg.flows[future]
            return future
        started = (
            leg._fixing_ordinals
            if isinstance(leg, TermRateLeg)
            else leg._start_ordinals
        )
        if (future & (started <= t)).any():  # a running or fixed coupon
            raise _Uncompiled
        n = int(future.sum())
        self._flows = np.zeros(n)
        accrual = self._source("accrual", leg.index)
        self._projected.append(
            (
                self._n_terms + np.arange(n),
                leg.residuals[future],
                leg.fixed_flows[future],
                leg._window_ratios[future],
                leg._window_power[future],
                self._ref(
                    accrual,
                    leg._window_start_ordinals[future],
                    leg._window_end_ordinals[future],
                ),
            )
        )
        return future

    def _add_terms(self, flows, fx, df) -> None:
        self._terms.append((np.asarray(flows, dtype=float), fx, df))
        self._n_terms += len(flows)

    def _compile_ndf(self, ndf: NDF) -> _Ndf:
        """_forwards.get_ndf_mtm, its static part."""
        market = self.market
        cp = ndf.currency_pair
        if ndf.payment_date <= market.t:
            return _Ndf(None, None, None, None, None, None)
        fx = None
        s_t = None
        if ndf.fixing_date <= market.t:
            s_t = market.get_fx_rate(ndf.fixing_date, cp).value
        elif ndf.fixing_date <= market.get_fx_known_until(cp):  # get_future_fx_rate
            s_t = market.get_fx_rate(ndf.fixing_date, cp).value
        else:
            fx = self._ref(
                self._source("fx", cp), np.array([ndf.fixing_date.toordinal()])
            )
        discount = self._source("discount", cp.quote_currency, ndf.collateral_index)
        df = self._ref(discount, np.array([ndf.payment_date.toordinal()]))
        return _Ndf(
            ndf.sign * ndf.notional,
            ndf.strike,
            s_t,
            fx,
            df,
            self._spot(cp.quote_currency),
        )

    def _freeze(self) -> None:
        """Lay the sources end to end, then turn every reference into its positions in
        the values of one call (the last one is a 1, the FX of a deliverable flow)."""
        offset = {}
        n = 0
        for source in self._sources.values():
            source.freeze()
            offset[source] = n
            n += source.size
        one = n

        def at(ref):
            source, start, size = ref
            return offset[source] + start + np.arange(size)

        terms = self._terms
        self._static_flows = (
            np.concatenate([f for f, _, _ in terms]) if terms else np.zeros(0)
        )
        self._fx_at = np.concatenate(
            [at(fx) if fx is not None else np.full(len(f), one) for f, fx, _ in terms]
            or [np.zeros(0, dtype=np.int64)]
        ).astype(np.intp)
        self._df_at = np.concatenate(
            [at(df) for _, _, df in terms] or [np.zeros(0, dtype=np.int64)]
        ).astype(np.intp)
        projected = self._projected
        self._projected_pos = np.concatenate(
            [p for p, *_ in projected] or [np.zeros(0, dtype=np.int64)]
        ).astype(np.intp)
        self._residuals = np.concatenate([p[1] for p in projected] or [np.zeros(0)])
        self._fixed = np.concatenate([p[2] for p in projected] or [np.zeros(0)])
        # Term-rate coupons projected over a fixing window other than their period:
        # their interest is _interest_from_window of the window's accrual.
        self._window_ratios = np.concatenate([p[3] for p in projected] or [np.zeros(0)])
        self._window_power = np.concatenate(
            [p[4] for p in projected] or [np.zeros(0, dtype=bool)]
        )
        self._on_accrual = bool(
            np.all(self._window_ratios == 1.0) and not self._window_power.any()
        )
        self._accrual_at = np.concatenate(
            [at(a) for *_, a in projected] or [np.zeros(0, dtype=np.int64)]
        ).astype(np.intp)
        self._head_residuals = np.array([r for r, _ in self._heads], dtype=float)
        self._head_at = np.concatenate(
            [at(d) for _, d in self._heads] or [np.zeros(0, dtype=np.int64)]
        ).astype(np.intp)
        for _, plan in self._compiled:
            if isinstance(plan, _Ndf):
                plan.fx = None if plan.fx is None else int(at(plan.fx)[0])
                plan.df = None if plan.df is None else int(at(plan.df)[0])
        self._sources_list = list(self._sources.values())
        # log(value + shift) of every value is affine in the curves' forwards on
        # piecewise-constant/linear or spline curves: dfs, FX (anchor * df ratios) and
        # 1 + accrual (a df ratio); the trailing 1 is constant.
        self._log_shift = np.concatenate(
            [
                np.full(s.size, 1.0 if s.kind == "accrual" else 0.0)
                for s in self._sources_list
            ]
            + [np.zeros(1)]
        )
        self._gradient_rows = None
        del self._terms, self._projected, self._heads

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def _interest(self, accruals: np.ndarray) -> np.ndarray:
        """Interest per unit notional of every projected coupon from its window's
        accrual: the accrual itself, unless a term-rate coupon's fixing window is not
        its period (:func:`_interest_from_window`, as the calculator computes it)."""
        if self._on_accrual:
            return accruals
        return _interest_from_window(accruals, self._window_ratios, self._window_power)

    def values(self) -> np.ndarray:
        """Every MTM, in the order of the instruments."""
        try:
            return self._values()
        except Exception:
            # The calculator, one instrument at a time, raises its own error.
            from . import Calculator

            return np.array(
                [
                    Calculator.valuate(i, self.market, self.currency)
                    for i in self.instruments
                ]
            )

    def _values(self) -> np.ndarray:
        market, currency = self.market, self.currency
        out = np.empty(len(self.instruments))
        if self._compiled:
            values = np.concatenate(
                [s.evaluate(market) for s in self._sources_list] + [np.ones(1)]
            )
            flows = self._static_flows.copy()
            if self._projected_pos.size:
                # _projected_flows: residual * interest + fixed flows
                flows[self._projected_pos] = (
                    self._residuals * self._interest(values[self._accrual_at])
                    + self._fixed
                )
            # flows * fx * dfs (deliverable: fx = 1, which leaves flows * dfs)
            products = (flows * values[self._fx_at] * values[self._df_at]).tolist()
            heads = (self._head_residuals * values[self._head_at]).tolist()
            fsum = math.fsum
            for k, plan in self._compiled:
                if isinstance(plan, _Ndf):
                    out[k] = _ndf_value(plan, values)
                    continue
                mtm = []
                for leg in plan:
                    pv = fsum(products[leg.start : leg.stop])
                    if leg.head is not None:
                        head = heads[leg.head]
                        pv = head + pv if leg.stop > leg.start else head
                    mtm.append(pv * leg.spot)
                out[k] = mtm[0] - mtm[1]
        if self._fallback:
            from . import Calculator

            for k in self._fallback:
                out[k] = Calculator.valuate(self.instruments[k], market, currency)
        return out

    # ── Sensitivities ──────────────────────────────────────────────────────────

    @property
    def differentiable(self) -> bool:
        """Whether every instrument is compiled, so :meth:`gradient` covers them all."""
        return not self._fallback

    @property
    def log_shift(self) -> np.ndarray:
        """Per value, the shift making ``log(value + shift)`` affine in the forwards
        of piecewise-constant/linear and spline curves (1 for an accrual, else 0)."""
        return self._log_shift

    def source_values(self) -> np.ndarray:
        """The values one call reads off the market: every source end to end, then 1."""
        market = self.market
        return np.concatenate(
            [s.evaluate(market) for s in self._sources_list] + [np.ones(1)]
        )

    def gradient(self, values: np.ndarray) -> np.ndarray:
        """``d MTM / d values``: one row per instrument, one column per value of
        :meth:`source_values` (the trailing constant 1 included), at `values`.

        Only for a :attr:`differentiable` batch. The MTMs are sums of products of the
        values (``flow * fx * df`` per term, ``residual_0 * df`` per par head, ``a *
        (s_t - strike) * df * spot`` per NDF) with constant coefficients, so each
        derivative is a product of the others.
        """
        if self._gradient_rows is None:
            self._gradient_rows = self._gradient_layout()
        term_row, term_scale, head_row, head_scale = self._gradient_rows
        grad = np.zeros((len(self.instruments), len(values)))
        flows = self._static_flows.copy()
        projected = self._projected_pos
        accruals = values[self._accrual_at]
        if projected.size:
            flows[projected] = self._residuals * self._interest(accruals) + self._fixed
        fx, df = values[self._fx_at], values[self._df_at]
        np.add.at(grad, (term_row, self._df_at), flows * fx * term_scale)
        np.add.at(grad, (term_row, self._fx_at), flows * df * term_scale)
        if projected.size:  # flow = residual * interest(accrual) + fixed
            slope = self._residuals
            if not self._on_accrual:
                slope = slope * _interest_from_window_slope(
                    accruals, self._window_ratios, self._window_power
                )
            np.add.at(
                grad,
                (term_row[projected], self._accrual_at),
                slope * fx[projected] * df[projected] * term_scale[projected],
            )
        if head_row.size:
            np.add.at(
                grad, (head_row, self._head_at), self._head_residuals * head_scale
            )
        for k, plan in self._compiled:
            if not isinstance(plan, _Ndf) or plan.a is None:
                continue
            s_t = plan.s_t if plan.s_t is not None else values[plan.fx]
            grad[k, plan.df] += plan.a * (s_t - plan.strike) * plan.spot
            if plan.fx is not None:
                grad[k, plan.fx] += plan.a * values[plan.df] * plan.spot
        return grad

    def _gradient_layout(self) -> tuple:
        """Instrument row and ``sign * spot`` of every term and par head."""
        term_row = np.zeros(self._static_flows.size, dtype=np.intp)
        term_scale = np.zeros(self._static_flows.size)
        head_row = np.zeros(self._head_residuals.size, dtype=np.intp)
        head_scale = np.zeros(self._head_residuals.size)
        for k, plan in self._compiled:
            if isinstance(plan, _Ndf):
                continue
            for leg, sign in zip(plan, (1.0, -1.0)):  # receive - pay
                term_row[leg.start : leg.stop] = k
                term_scale[leg.start : leg.stop] = sign * leg.spot
                if leg.head is not None:
                    head_row[leg.head] = k
                    head_scale[leg.head] = sign * leg.spot
        return term_row, term_scale, head_row, head_scale


def _ndf_value(plan: _Ndf, values: np.ndarray) -> float:
    """_forwards.get_ndf_mtm on the values of one call."""
    if plan.a is None:  # paid
        return 0.0
    if plan.s_t is not None:
        s_t = plan.s_t
    else:
        rate = float(values[plan.fx])
        if rate <= 0:  # FX_Rate rejects it, with its own message
            raise ValueError(rate)
        s_t = rate
    fv = plan.a * (s_t - plan.strike)
    return fv * float(values[plan.df]) * plan.spot
