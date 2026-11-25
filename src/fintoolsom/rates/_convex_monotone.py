"""Monotone convex forward: the function integrated between knots."""

# Ported from QuantLib, ql/math/interpolations/convexmonotoneinterpolation.hpp,
# which is distributed under these terms:
#
# Copyright (C) 2008 Simon Ibbotson
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#     Neither the names of the copyright holders nor the names of the QuantLib
#     Group and its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import numpy as np
from scipy.interpolate import PPoly

# Weight of the plain quadratic in the blend with the monotone convex shape.
QUADRATICITY = 0.3
# 1 is the basic Hagan & West shape; lower values keep its kink off the segment ends.
MONOTONICITY = 0.7

_ETA_MIN = (1 - MONOTONICITY) / 2
_ETA_MAX = (1 + MONOTONICITY) / 2
_FLAT_TOLERANCE = 1e-14


def convex_monotone_forward(knots: np.ndarray, averages: np.ndarray) -> PPoly:
    """Continuous forward with the given average over every segment.

    Monotone convex method of Patrick S. Hagan and Graeme West, "Interpolation Methods
    for Curve Construction", Applied Mathematical Finance 13(2), 2006, with the
    ``quadraticity`` (0.3) and ``monotonicity`` (0.7) enhancements of Simon Ibbotson's
    ``ConvexMonotone`` interpolation in QuantLib, at its default values.

    QuantLib's ``forcePositive`` floor is left out: it assumes positive averages, and
    basis and real-rate curves can have negative ones. Forwards can therefore go
    negative between knots.

    Parameters
    ----------
    knots : np.ndarray
        Times of the knots, strictly increasing.
    averages : np.ndarray
        Average forward over each segment: ``len(knots) - 1`` values.

    Returns
    -------
    PPoly
        The instantaneous forward between the first and the last knot.
    """
    at_knots = _forwards_at_knots(np.diff(knots), averages)
    n_segments = len(averages)
    knots_l = knots.tolist()
    at_l = at_knots.tolist()
    avg_l = averages.tolist()

    breaks = [0.0] * (2 * n_segments + 1)
    c0 = [0.0] * (2 * n_segments)
    c1 = [0.0] * (2 * n_segments)
    c2 = [0.0] * (2 * n_segments)
    breaks[-1] = knots_l[-1]

    q = QUADRATICITY
    one_minus_q = 1.0 - q

    for i in range(n_segments):
        start = knots_l[i]
        end = knots_l[i + 1]
        average = avg_l[i]
        g0 = at_l[i] - average
        g1 = at_l[i + 1] - average
        width = end - start

        qa = 3.0 * (g0 + g1)
        qb = -(4.0 * g0 + 2.0 * g1)
        qc = g0

        if (abs(g0) < _FLAT_TOLERANCE and abs(g1) < _FLAT_TOLERANCE) or (
            (g0 > 0.0 and -2.0 * g0 <= g1 <= -0.5 * g0)
            or (g0 < 0.0 and -0.5 * g0 <= g1 <= -2.0 * g0)
        ):
            eta = 0.5
            la, lb, lc = qa, qb, qc + average
            ra, rb, rc = qa, qb, qc + average
        else:
            eta, vertex = _shape(g0, g1)
            if g0 != vertex:
                kl = (g0 - vertex) / (eta * eta)
                pla, plb, plc = kl, -2.0 * kl * eta, kl * eta * eta + vertex
            else:
                pla, plb, plc = 0.0, 0.0, vertex
            reach_r = 1.0 - eta
            if g1 != vertex:
                kr = (g1 - vertex) / (reach_r * reach_r)
                pra, prb, prc = kr, -2.0 * kr * eta, kr * eta * eta + vertex
            else:
                pra, prb, prc = 0.0, 0.0, vertex
            la = q * qa + one_minus_q * pla
            lb = q * qb + one_minus_q * plb
            lc = q * qc + one_minus_q * plc + average
            ra = q * qa + one_minus_q * pra
            rb = q * qb + one_minus_q * prb
            rc = q * qc + one_minus_q * prc + average

        j = 2 * i
        breaks[j] = start
        breaks[j + 1] = start + eta * width

        inv_w = 1.0 / width
        inv_w2 = inv_w * inv_w
        c0[j] = la * inv_w2
        c1[j] = lb * inv_w
        c2[j] = lc

        c0[j + 1] = ra * inv_w2
        c1[j + 1] = (2.0 * ra * eta + rb) * inv_w
        c2[j + 1] = (ra * eta + rb) * eta + rc

    # A shape's inner break can round onto the end of its segment.
    keep_b: list[float] = []
    keep_c0: list[float] = []
    keep_c1: list[float] = []
    keep_c2: list[float] = []
    for k in range(2 * n_segments):
        if breaks[k + 1] > breaks[k]:
            keep_b.append(breaks[k])
            keep_c0.append(c0[k])
            keep_c1.append(c1[k])
            keep_c2.append(c2[k])
    keep_b.append(breaks[-1])

    coeffs = np.array([keep_c0, keep_c1, keep_c2], dtype=float)
    x = np.array(keep_b, dtype=float)
    return PPoly.construct_fast(coeffs, x)


def _forwards_at_knots(widths: np.ndarray, averages: np.ndarray) -> np.ndarray:
    """Inner knots weigh the averages around them; the ends extrapolate."""
    weight = widths[1:] / (widths[:-1] + widths[1:])
    inner = weight * averages[:-1] + (1 - weight) * averages[1:]
    first = 1.5 * averages[0] - 0.5 * inner[0]
    last = 1.5 * averages[-1] - 0.5 * inner[-1]
    return np.concatenate(([first], inner, [last]))


def _segment_polynomials(g0: float, g1: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Forward minus the average, on a segment scaled to ``[0, 1]``.

    `g0` and `g1` are its values at the two ends. Returns the point where the
    polynomials meet and both, as ``(a, b, c)`` of ``a s^2 + b s + c``. All of them
    integrate to zero, so the segment keeps its average.
    """
    quadratic = np.array([3 * (g0 + g1), -(4 * g0 + 2 * g1), g0])
    is_flat = abs(g0) < _FLAT_TOLERANCE and abs(g1) < _FLAT_TOLERANCE
    is_monotone = (g0 > 0 and -2 * g0 <= g1 <= -0.5 * g0) or (
        g0 < 0 and -0.5 * g0 <= g1 <= -2 * g0
    )
    if is_flat or is_monotone:
        return 0.5, quadratic, quadratic
    eta, vertex = _shape(g0, g1)
    left = _parabola(eta, vertex, g0, eta)
    right = _parabola(eta, vertex, g1, 1 - eta)
    return (
        eta,
        QUADRATICITY * quadratic + (1 - QUADRATICITY) * left,
        QUADRATICITY * quadratic + (1 - QUADRATICITY) * right,
    )


def _shape(g0: float, g1: float) -> tuple[float, float]:
    """Where the two parabolas meet and their common value there."""
    if (g0 < 0 and g1 > -2 * g0) or (g0 > 0 and g1 < -2 * g0):
        # Flat from the start, then the whole move at the end.
        eta = (g1 + 2 * g0) / (g1 - g0)
        return (eta, g0) if eta < _ETA_MAX else (_ETA_MAX, _balanced(g0, g1, _ETA_MAX))
    if (g0 > 0 and -0.5 * g0 < g1 < 0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
        # The whole move at the start, then flat.
        eta = 3 * g1 / (g1 - g0)
        return (eta, g1) if eta > _ETA_MIN else (_ETA_MIN, _balanced(g0, g1, _ETA_MIN))
    # Both ends on the same side of the average.
    eta = min(max(g1 / (g0 + g1), _ETA_MIN), _ETA_MAX)
    return eta, _balanced(g0, g1, eta)


def _balanced(g0: float, g1: float, eta: float) -> float:
    """Value at `eta` that makes two parabolas from `g0` and to `g1` integrate to 0."""
    return -0.5 * (eta * g0 + (1 - eta) * g1)


def _parabola(eta: float, vertex: float, end: float, reach: float) -> np.ndarray:
    """``a s^2 + b s + c`` with its vertex at ``(eta, vertex)``, `end` `reach` away."""
    curvature = (end - vertex) / reach**2 if end != vertex else 0.0
    return np.array([curvature, -2 * curvature * eta, curvature * eta**2 + vertex])


def _around(polynomial: np.ndarray, s: float, width: float) -> np.ndarray:
    """A polynomial of the segment's ``[0, 1]`` as powers of the time since `s`."""
    a, b, c = polynomial.tolist()
    return np.array([a / width**2, (2 * a * s + b) / width, a * s**2 + b * s + c])
