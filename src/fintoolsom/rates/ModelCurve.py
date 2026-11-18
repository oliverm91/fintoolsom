"""A :class:`~fintoolsom.rates.Curve` backed by a term-structure model."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from datetime import date
from typing import Protocol, Self, Sequence, runtime_checkable

import numpy as np

from .Curve import DAY_BASE, Curve, year_fractions


@runtime_checkable
class TermStructureModel(Protocol):
    """What a :class:`ModelCurve` needs from a model: discount factors in time.

    Any object with this method qualifies, without inheriting from anything, e.g.
    :class:`~fintoolsom.models.NelsonSiegel` and
    :class:`~fintoolsom.models.NelsonSiegelSvensson`.
    """

    def get_df(self, t: np.ndarray) -> np.ndarray:
        """Discount factors at Act/365 year fractions `t` (vectorised); 1 at t = 0."""
        ...


@dataclass
class ModelCurve(Curve):
    """Curve whose discount factors come from a :class:`TermStructureModel`.

    ``df(d) = model.get_df(t_d) / model.get_df(t_c) · exp(-spread · (t_d - t_c))``,
    with ``t`` the Act/365 years from `model_date` and ``t_c`` that of `curve_date`.
    Everything else (zero and forward rates, accruals, bumps, aging) follows from
    the dfs, so the model only needs ``get_df``.

    Attributes
    ----------
    curve_date : date
        Date the dfs are measured from (df = 1 there).
    model : TermStructureModel
        The model, copied on construction: recalibrating the original later does not
        move the curve.
    horizon : date
        Last date the model is meant for, e.g. the last maturity it was fitted to; it
        is :attr:`last_date`. Past it the model keeps evaluating its own formula.
    spread : float
        Continuously compounded Act/365 shift of every zero rate from `curve_date`;
        :meth:`parallel_bump_bps` adds to it.
    model_date : date, optional
        The model's t = 0; `curve_date` by default. An aged curve keeps it and moves
        `curve_date` only.
    """

    curve_date: date
    model: TermStructureModel
    horizon: date
    spread: float = 0.0
    model_date: date | None = None

    def __post_init__(self):
        if not isinstance(self.model, TermStructureModel):
            raise TypeError(
                f"model must have get_df(t), got {type(self.model).__name__}."
            )
        self.model = copy.deepcopy(self.model)
        if self.model_date is None:
            self.model_date = self.curve_date
        if self.model_date > self.curve_date:
            raise ValueError(
                f"model_date {self.model_date} is after the curve date {self.curve_date}."
            )
        if self.horizon <= self.curve_date:
            raise ValueError(
                f"horizon {self.horizon} must be after the curve date {self.curve_date}."
            )
        # Years from the model's t = 0 to the curve date, and the model's df there.
        self._t0: float = (self.curve_date - self._model_date).days / DAY_BASE
        self._model_df0: float = (
            float(np.asarray(self.model.get_df(np.array(self._t0))))
            if self._t0
            else 1.0
        )

    @property
    def _model_date(self) -> date:
        return self.model_date if self.model_date is not None else self.curve_date

    def _dfs_at_years(self, t: np.ndarray) -> np.ndarray:
        """Discount factors at Act/365 years `t` from `model_date`."""
        dfs = np.asarray(self.model.get_df(t), dtype=float) / self._model_df0
        if self.spread:
            dfs = dfs * np.exp(-self.spread * (t - self._t0))
        return dfs

    def get_dfs(self, dates: Sequence[date]) -> np.ndarray:
        return self._dfs_at_years(year_fractions(self._model_date, dates))

    def _dfs_at_ordinals(self, ordinals: np.ndarray) -> np.ndarray:
        days = np.asarray(ordinals, dtype=np.int64) - self._model_date.toordinal()
        return self._dfs_at_years(days / DAY_BASE)

    @property
    def last_date(self) -> date:
        """The model's `horizon`."""
        return self.horizon

    def parallel_bump_bps(self, bps: float) -> Self:
        return replace(self, spread=self.spread + bps / 10_000)

    def get_aged_curve(self, date: date) -> Self:
        """The curve seen from `date`: the same model and spread, dfs divided by the
        df at `date`. Exact, since the model is not refitted."""
        self._check_aging_date(date)
        return replace(self, curve_date=date, model_date=self._model_date)

    def copy(self) -> Self:
        return replace(self)  # __post_init__ copies the model
