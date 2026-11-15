"""Standard normal cdf and its inverse for scalars, in pure Python.

``ndtr`` is ``0.5 * erfc(-x / sqrt(2))`` with the C library's ``erfc`` (``math.erfc``)
and ``ndtri`` a port of the Cephes code behind ``scipy.special.ndtri``. Both agree with
scipy to a few ulps; scipy's compiled kernels are not used because their rounding
depends on how scipy was built, while these give the same bits as the C++ backend
(``fintoolsom._native.ndtr`` / ``ndtri``), which runs the same operations.
"""

from __future__ import annotations

import math

_SQRT1_2 = 0.70710678118654752440
_SQRT2PI = 2.506628274631000502415765284811045253007
_EXP_M2 = 0.13533528323661269189  # exp(-2)

# Cephes ndtri.h coefficients.
# approximation for 0 <= |y - 0.5| <= 3/8
_P0 = (
    -5.99633501014107895267e1,
    9.80010754185999661536e1,
    -5.66762857469070293439e1,
    1.39312609387279679503e1,
    -1.23916583867381258016e0,
)
_Q0 = (
    1.95448858338141759834e0,
    4.67627912898881538453e0,
    8.63602421390890590575e1,
    -2.25462687854119370527e2,
    2.00260212380060660359e2,
    -8.20372256168333339912e1,
    1.59056225126211695515e1,
    -1.18331621121330003142e0,
)
# approximation for 2 <= sqrt(-2 log y) < 8
_P1 = (
    4.05544892305962419923e0,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539e0,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4,
)
_Q1 = (
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979e0,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4,
)
# approximation for 8 <= sqrt(-2 log y) < 64
_P2 = (
    3.23774891776946035970e0,
    6.91522889068984211695e0,
    3.93881025292474443415e0,
    1.33303460815807542389e0,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9,
)
_Q2 = (
    6.02427039364742014255e0,
    3.67983563856160859403e0,
    1.37702099489081330271e0,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9,
)


def _polevl(x: float, coef: tuple[float, ...]) -> float:
    """Cephes polevl: ``coef[0] x^N + ... + coef[N]`` by Horner."""
    ans = coef[0]
    for c in coef[1:]:
        ans = ans * x + c
    return ans


def _p1evl(x: float, coef: tuple[float, ...]) -> float:
    """Cephes p1evl: :func:`_polevl` with an implied leading coefficient of 1."""
    ans = x + coef[0]
    for c in coef[1:]:
        ans = ans * x + c
    return ans


def ndtr(x: float) -> float:
    """Standard normal cumulative distribution at `x`."""
    return 0.5 * math.erfc(-x * _SQRT1_2)


def ndtri(y0: float) -> float:
    """Inverse of :func:`ndtr`: -inf at 0, inf at 1, nan outside [0, 1]."""
    if y0 == 0.0:
        return -math.inf
    if y0 == 1.0:
        return math.inf
    if not 0.0 < y0 < 1.0:
        return math.nan
    code = True
    y = y0
    if y > 1.0 - _EXP_M2:
        y = 1.0 - y
        code = False
    if y > _EXP_M2:
        y = y - 0.5
        y2 = y * y
        x = y + y * (y2 * _polevl(y2, _P0) / _p1evl(y2, _Q0))
        return x * _SQRT2PI
    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x
    z = 1.0 / x
    if x < 8.0:  # y > exp(-32)
        x1 = z * _polevl(z, _P1) / _p1evl(z, _Q1)
    else:
        x1 = z * _polevl(z, _P2) / _p1evl(z, _Q2)
    x = x0 - x1
    return -x if code else x
