"""Which implementation, Python or C++, each subpackage uses.

Every subpackage has a Python implementation. Those listed in :data:`PORTED` also
have a C++ one, in the compiled extension ``fintoolsom._native``; both give the same
results, so switching only changes speed. Use the Python one to step through a
calculation in a debugger.

What each backend covers:

- ``"dates"``: calendars, business-day conventions, ``Tenor``, ``Term``, schedule
  generation, day counts, time fractions and ``date_ordinals``, on C++ copies of the
  calendars' days (loaded from ``holidays`` in one call per calendar).
- ``"rates"``: the curves (``IndexCurve``, ``DiscountCurve``), ``Rate`` factors,
  curve year fractions and knot checks.
- ``"derivatives"``: swap, forward, NDF and option valuation arithmetic, swap leg
  construction (date ordinals and checks, amortizations, the par check; a bullet
  leg's schedule and arrays in one call with ``"dates"``), leg duration.
- ``"market"``: the market lookups (curves, FX rates, realised fixings) of a
  valuation: an instrument with a C++ core valued on a C++ market runs in one C++
  call; ``Market.get_dfs``, its FX projection and accrual loops; the index
  histories' day walks (with ``"dates"``) and the UF extension.
- ``"curve_builder"``: ``build_curves``. Being ported (branch ``cpp-curve-builder``):
  the group solves run in C++ (block Newton sweeps, the joint solve by Newton with the
  analytic Jacobian) when the curves and instruments are C++ (``"rates"``,
  ``"market"``, ``"derivatives"``), and so do the instruments' pillar dates and the
  fixed-leg seeds; the forward rates may differ from the Python builder's within 1e-8.

The backend is read when an object is created (a curve, leg, forward, option or
market built under ``"cpp"`` keeps its C++ path after the setting changes), or, for
``dates``, ``Rate`` factors, the index histories, ``get_strike_from_delta`` and
``build_curves``, when called.

Set it from the environment before importing fintoolsom::

    FINTOOLSOM_BACKEND=cpp            # every ported subpackage
    FINTOOLSOM_BACKEND_RATES=python   # one subpackage, overriding the above

or in code::

    from fintoolsom import settings

    settings.set_backend("cpp")                 # every ported subpackage
    settings.set_backend("python", "rates")     # one subpackage
    with settings.use_backend("python"):        # temporarily
        ...
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

Backend = Literal["python", "cpp"]

BACKENDS: tuple[str, ...] = ("python", "cpp")
SUBPACKAGES: tuple[str, ...] = (
    "dates",
    "rates",
    "market",
    "derivatives",
    "curve_builder",
)
# Subpackages with a C++ implementation.
PORTED: frozenset[str] = frozenset(
    {"dates", "rates", "market", "derivatives", "curve_builder"}
)

DEFAULT_BACKEND: Backend = "python"

_backends: dict[str, str] = {}


def _check(backend: str, subpackage: str) -> None:
    if subpackage not in SUBPACKAGES:
        raise ValueError(
            f"Unknown subpackage '{subpackage}'; expected one of {', '.join(SUBPACKAGES)}."
        )
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'; expected one of {', '.join(BACKENDS)}."
        )
    if backend == "cpp" and subpackage not in PORTED:
        raise ValueError(
            f"'{subpackage}' has no C++ implementation yet; ported: "
            f"{', '.join(sorted(PORTED))}."
        )


def get_backend(subpackage: str) -> str:
    """Backend `subpackage` uses: ``"python"`` or ``"cpp"``."""
    if subpackage not in SUBPACKAGES:
        raise ValueError(
            f"Unknown subpackage '{subpackage}'; expected one of {', '.join(SUBPACKAGES)}."
        )
    return _backends[subpackage]


def set_backend(backend: Backend, *subpackages: str) -> None:
    """Use `backend` for `subpackages`, or for every ported one when none is given.

    Raises
    ------
    ValueError
        For an unknown backend or subpackage, or ``"cpp"`` for a subpackage with no
        C++ implementation.
    ImportError
        For ``"cpp"`` when the compiled extension is not installed.
    """
    targets = subpackages or (
        tuple(sorted(PORTED)) if backend == "cpp" else SUBPACKAGES
    )
    for subpackage in targets:
        _check(backend, subpackage)
    if backend == "cpp":
        native()
    for subpackage in targets:
        _backends[subpackage] = backend


@contextmanager
def use_backend(backend: Backend, *subpackages: str) -> Generator[None, None, None]:
    """Use `backend` inside the ``with`` block, as :func:`set_backend` does."""
    previous = dict(_backends)
    set_backend(backend, *subpackages)
    try:
        yield
    finally:
        _backends.clear()
        _backends.update(previous)


def native():
    """The compiled extension module ``fintoolsom._native``.

    Raises
    ------
    ImportError
        If the extension is not built (e.g. an install from source without a C++23
        compiler).
    """
    global _native_module
    if _native_module is not None:
        return _native_module
    try:
        from . import _native
    except ImportError as e:
        raise ImportError(
            "The C++ backend of fintoolsom is not available: the extension "
            "fintoolsom._native is not built. Install a wheel, or build from source "
            "with a C++23 compiler, or use the Python backend."
        ) from e
    _native_module = _native
    return _native


_native_module = None


def is_cpp(subpackage: str) -> bool:
    """Whether `subpackage` uses the C++ backend now (a fast check for hot paths)."""
    return _backends[subpackage] == "cpp"


def _from_environment() -> None:
    default = os.environ.get("FINTOOLSOM_BACKEND", DEFAULT_BACKEND).lower()
    for subpackage in SUBPACKAGES:
        backend = os.environ.get(f"FINTOOLSOM_BACKEND_{subpackage.upper()}")
        if backend is None:
            # The global default only reaches the ported subpackages.
            backend = default if subpackage in PORTED else DEFAULT_BACKEND
        backend = backend.lower()
        _check(backend, subpackage)
        _backends[subpackage] = backend
    if "cpp" in _backends.values():
        native()


_from_environment()
