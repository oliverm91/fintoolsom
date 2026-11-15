#include "curves/fits.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace fts::fits {
namespace {

// numpy.sign: -1, 0 or 1 (NaN propagates as NaN, which compares unequal).
double sign(double v) noexcept { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : (v == 0.0 ? 0.0 : v)); }

// scipy CubicHermiteSpline coefficients from values y and slopes dydx at x.
PiecewisePoly cubic_hermite(std::span<const double> x, std::span<const double> y,
                            const std::vector<double>& dydx) {
    const std::size_t n = x.size();
    PiecewisePoly p;
    p.breaks.assign(x.begin(), x.end());
    p.pieces.resize(n - 1);
    p.degree = 3;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const double dx = x[i + 1] - x[i];
        const double slope = (y[i + 1] - y[i]) / dx;
        const double t = (dydx[i] + dydx[i + 1] - 2 * slope) / dx;
        Cubic& c = p.pieces[i];
        c.c[3] = t / dx;
        c.c[2] = (slope - dydx[i]) / dx - t;
        c.c[1] = dydx[i];
        c.c[0] = y[i];
    }
    return p;
}

// PchipInterpolator._edge_case: one-sided three-point slope, kept shape preserving.
double pchip_edge(double h0, double h1, double m0, double m1) noexcept {
    double d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    const bool mask = sign(d) != sign(m0);
    const bool mask2 = (sign(m0) != sign(m1)) && (std::abs(d) > 3. * std::abs(m0));
    if (mask) {
        d = 0.;
    } else if (mask2) {
        d = 3. * m0;
    }
    return d;
}

// ── Convex monotone (rates/_convex_monotone.py) ──────────────────────────────

constexpr double kQuadraticity = 0.3;
constexpr double kMonotonicity = 0.7;
constexpr double kEtaMin = (1 - kMonotonicity) / 2;
constexpr double kEtaMax = (1 + kMonotonicity) / 2;
constexpr double kFlatTolerance = 1e-14;

struct Quad {  // a s^2 + b s + c on the segment scaled to [0, 1]
    double a, b, c;
};

double balanced(double g0, double g1, double eta) noexcept {
    return -0.5 * (eta * g0 + (1 - eta) * g1);
}

void shape(double g0, double g1, double& eta, double& vertex) noexcept {
    if ((g0 < 0 && g1 > -2 * g0) || (g0 > 0 && g1 < -2 * g0)) {
        eta = (g1 + 2 * g0) / (g1 - g0);
        if (eta < kEtaMax) {
            vertex = g0;
        } else {
            eta = kEtaMax;
            vertex = balanced(g0, g1, kEtaMax);
        }
        return;
    }
    if ((g0 > 0 && -0.5 * g0 < g1 && g1 < 0) || (g0 < 0 && 0 < g1 && g1 < -0.5 * g0)) {
        eta = 3 * g1 / (g1 - g0);
        if (eta > kEtaMin) {
            vertex = g1;
        } else {
            eta = kEtaMin;
            vertex = balanced(g0, g1, kEtaMin);
        }
        return;
    }
    eta = std::min(std::max(g1 / (g0 + g1), kEtaMin), kEtaMax);
    vertex = balanced(g0, g1, eta);
}

Quad parabola(double eta, double vertex, double end, double reach) noexcept {
    const double curvature = end != vertex ? (end - vertex) / std::pow(reach, 2.0) : 0.0;
    return {curvature, -2 * curvature * eta, curvature * std::pow(eta, 2.0) + vertex};
}

Quad blend(const Quad& quadratic, const Quad& shaped) noexcept {
    return {kQuadraticity * quadratic.a + (1 - kQuadraticity) * shaped.a,
            kQuadraticity * quadratic.b + (1 - kQuadraticity) * shaped.b,
            kQuadraticity * quadratic.c + (1 - kQuadraticity) * shaped.c};
}

// _around: the [0, 1] polynomial as powers of the time since `s`, scaled by `width`;
// returned in PiecewisePoly order (constant first).
Cubic around(const Quad& q, double s, double width) noexcept {
    Cubic out;
    out.c[2] = q.a / std::pow(width, 2.0);
    out.c[1] = (2 * q.a * s + q.b) / width;
    out.c[0] = q.a * std::pow(s, 2.0) + q.b * s + q.c;
    return out;
}

}  // namespace

PiecewisePoly natural_cubic_spline(std::span<const double> x, std::span<const double> y) {
    const std::size_t n = x.size();
    std::vector<double> dx(n - 1), slope(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        dx[i] = x[i + 1] - x[i];
        slope[i] = (y[i + 1] - y[i]) / dx[i];
    }
    // Tridiagonal system for the slopes s, laid out as scipy's solve_banded hands it to
    // dgtsv: dl[i] = M[i+1, i], d[i] = M[i, i], du[i] = M[i, i+1].
    std::vector<double> dl(n - 1), d(n), du(n - 1), b(n);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        d[i] = 2 * (dx[i - 1] + dx[i]);
        b[i] = 3 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
    }
    for (std::size_t i = 2; i < n; ++i) du[i - 1] = dx[i - 2];  // A[0, 2:] = dx[:-1]
    for (std::size_t j = 0; j + 2 < n; ++j) dl[j] = dx[j + 1];  // A[-1, :-2] = dx[1:]
    // Natural ends (second derivative 0): b = -0.5*0*dx^2 + 3*dy, i.e. 3*dy.
    d[0] = 2 * dx[0];
    du[0] = dx[0];
    b[0] = -0.5 * 0.0 * std::pow(dx[0], 2.0) + 3 * (y[1] - y[0]);
    d[n - 1] = 2 * dx[n - 2];
    dl[n - 2] = dx[n - 2];
    b[n - 1] = 0.5 * 0.0 * std::pow(dx[n - 2], 2.0) + 3 * (y[n - 1] - y[n - 2]);

    // LAPACK dgtsv, one right-hand side: elimination with partial pivoting. A row
    // interchange creates a second super-diagonal, kept in dl.
    for (std::size_t i = 0; i + 2 < n; ++i) {
        if (std::abs(d[i]) >= std::abs(dl[i])) {
            const double fact = dl[i] / d[i];
            d[i + 1] = d[i + 1] - fact * du[i];
            b[i + 1] = b[i + 1] - fact * b[i];
            dl[i] = 0.0;
        } else {
            const double fact = d[i] / dl[i];
            d[i] = dl[i];
            double temp = d[i + 1];
            d[i + 1] = du[i] - fact * temp;
            dl[i] = du[i + 1];
            du[i + 1] = -fact * dl[i];
            du[i] = temp;
            temp = b[i];
            b[i] = b[i + 1];
            b[i + 1] = temp - fact * b[i + 1];
        }
    }
    if (n > 1) {
        const std::size_t i = n - 2;
        if (std::abs(d[i]) >= std::abs(dl[i])) {
            const double fact = dl[i] / d[i];
            d[i + 1] = d[i + 1] - fact * du[i];
            b[i + 1] = b[i + 1] - fact * b[i];
        } else {
            const double fact = d[i] / dl[i];
            d[i] = dl[i];
            const double temp = d[i + 1];
            d[i + 1] = du[i] - fact * temp;
            du[i] = temp;
            const double tb = b[i];
            b[i] = b[i + 1];
            b[i + 1] = tb - fact * b[i + 1];
        }
    }
    b[n - 1] = b[n - 1] / d[n - 1];
    if (n > 1) b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / d[n - 2];
    for (std::size_t k = n - 2; k-- > 0;) {
        b[k] = (b[k] - du[k] * b[k + 1] - dl[k] * b[k + 2]) / d[k];
    }
    return cubic_hermite(x, y, b);
}

PiecewisePoly pchip(std::span<const double> x, std::span<const double> y) {
    const std::size_t n = x.size();
    std::vector<double> hk(n - 1), mk(n - 1), dk(n, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        hk[i] = x[i + 1] - x[i];
        mk[i] = (y[i + 1] - y[i]) / hk[i];
    }
    if (n == 2) {
        dk[0] = dk[1] = mk[0];
        return cubic_hermite(x, y, dk);
    }
    for (std::size_t i = 0; i + 2 < n; ++i) {  // interior point i + 1
        const bool condition =
            (sign(mk[i + 1]) != sign(mk[i])) || (mk[i + 1] == 0) || (mk[i] == 0);
        if (!condition) {
            const double w1 = 2 * hk[i + 1] + hk[i];
            const double w2 = hk[i + 1] + 2 * hk[i];
            const double whmean = (w1 / mk[i] + w2 / mk[i + 1]) / (w1 + w2);
            dk[i + 1] = 1.0 / whmean;
        }
    }
    dk[0] = pchip_edge(hk[0], hk[1], mk[0], mk[1]);
    dk[n - 1] = pchip_edge(hk[n - 2], hk[n - 3], mk[n - 2], mk[n - 3]);
    return cubic_hermite(x, y, dk);
}

std::size_t QuadraticBSpline::interval(double x, std::size_t hint) const noexcept {
    const std::size_t n = t.size() - k - 1;
    std::size_t l = (k < hint && hint < n) ? hint : k;
    while (x < t[l] && l != k) --l;
    ++l;
    while (x >= t[l] && l != n) ++l;
    return l - 1;
}

double QuadraticBSpline::evaluate(double x, std::size_t ell, int nu) const noexcept {
    // fitpack _deBoor_D: h receives the k + 1 non-zero basis values (or derivatives).
    double h[k + 1] = {};
    double hh[k + 1] = {};
    h[0] = 1.0;
    for (int j = 1; j <= k - nu; ++j) {
        for (int i = 0; i < j; ++i) hh[i] = h[i];
        h[0] = 0.0;
        for (int n = 1; n <= j; ++n) {
            const std::size_t ind = ell + n;
            const double xb = t[ind];
            const double xa = t[ind - j];
            if (xb == xa) {
                h[n] = 0.0;
                continue;
            }
            const double w = hh[n - 1] / (xb - xa);
            h[n - 1] += w * (xb - x);
            h[n] = w * (x - xa);
        }
    }
    for (int j = k - nu + 1; j <= k; ++j) {
        for (int i = 0; i < j; ++i) hh[i] = h[i];
        h[0] = 0.0;
        for (int n = 1; n <= j; ++n) {
            const std::size_t ind = ell + n;
            const double xb = t[ind];
            const double xa = t[ind - j];
            if (xb == xa) {
                h[nu] = 0.0;
                continue;
            }
            const double w = j * hh[n - 1] / (xb - xa);
            h[n - 1] -= w;
            h[n] = w;
        }
    }
    double out = 0.0;
    for (int a = 0; a <= k; ++a) out = out + c[ell + a - k] * h[a];
    return out;
}

QuadraticBSpline linear_forward_integral(std::span<const double> x,
                                         std::span<const double> averages) {
    const std::size_t n = x.size();
    // _linear_forwards_at_knots
    std::vector<double> at_knots(n);
    at_knots[0] = at_knots[1] = averages[0];
    for (std::size_t i = 1; i < averages.size(); ++i) {
        at_knots[i + 1] = 2 * averages[i] - at_knots[i];
    }
    // make_interp_spline(k=1): t = r_[x[0], x, x[-1]], c = at_knots.
    std::vector<double> t1(n + 2);
    t1[0] = x[0];
    std::copy(x.begin(), x.end(), t1.begin() + 1);
    t1[n + 1] = x[n - 1];
    // splantider: c = cumsum(c * (t[k+1:] - t[:-k-1])) / (k + 1), padded; knots extended.
    QuadraticBSpline s;
    s.c.assign(n + 4, 0.0);
    double running = 0.0;
    std::vector<double> cum(n);
    for (std::size_t j = 0; j < n; ++j) {
        running = running + at_knots[j] * (t1[j + 2] - t1[j]);
        cum[j] = running;
    }
    for (std::size_t j = 0; j < n; ++j) s.c[j + 1] = cum[j] / 2;
    for (std::size_t j = n + 1; j < n + 4; ++j) s.c[j] = s.c[n];
    s.t.resize(n + 4);
    s.t[0] = t1[0];
    std::copy(t1.begin(), t1.end(), s.t.begin() + 1);
    s.t[n + 3] = t1[n + 1];
    return s;
}

PiecewisePoly convex_monotone_integral(std::span<const double> x,
                                       std::span<const double> averages) {
    const std::size_t n_seg = averages.size();
    // _forwards_at_knots
    std::vector<double> at_knots(n_seg + 1);
    for (std::size_t i = 0; i + 1 < n_seg; ++i) {
        const double w0 = x[i + 1] - x[i];
        const double w1 = x[i + 2] - x[i + 1];
        const double weight = w1 / (w0 + w1);
        at_knots[i + 1] = weight * averages[i] + (1 - weight) * averages[i + 1];
    }
    at_knots[0] = 1.5 * averages[0] - 0.5 * at_knots[1];
    at_knots[n_seg] = 1.5 * averages[n_seg - 1] - 0.5 * at_knots[n_seg - 1];

    // The forward: two quadratic pieces per segment, split where the shapes meet.
    std::vector<double> breaks(2 * n_seg + 1);
    std::vector<Cubic> forward(2 * n_seg);
    breaks[2 * n_seg] = x[n_seg];
    for (std::size_t i = 0; i < n_seg; ++i) {
        const double start = x[i];
        const double width = x[i + 1] - start;
        const double average = averages[i];
        const double g0 = at_knots[i] - average;
        const double g1 = at_knots[i + 1] - average;
        const Quad quadratic{3 * (g0 + g1), -(4 * g0 + 2 * g1), g0};
        const bool is_flat = std::abs(g0) < kFlatTolerance && std::abs(g1) < kFlatTolerance;
        const bool is_monotone = (g0 > 0 && -2 * g0 <= g1 && g1 <= -0.5 * g0) ||
                                 (g0 < 0 && -0.5 * g0 <= g1 && g1 <= -2 * g0);
        double eta = 0.5;
        Quad left = quadratic, right = quadratic;
        if (!(is_flat || is_monotone)) {
            double vertex = 0.0;
            shape(g0, g1, eta, vertex);
            left = blend(quadratic, parabola(eta, vertex, g0, eta));
            right = blend(quadratic, parabola(eta, vertex, g1, 1 - eta));
        }
        left.a += 0.0;
        left.b += 0.0;
        left.c += average;
        right.a += 0.0;
        right.b += 0.0;
        right.c += average;
        breaks[2 * i] = start;
        breaks[2 * i + 1] = start + eta * width;
        forward[2 * i] = around(left, 0.0, width);
        forward[2 * i + 1] = around(right, eta, width);
    }
    // A shape's inner break can round onto the end of its segment: drop empty pieces.
    PiecewisePoly integral;
    integral.degree = 3;
    for (std::size_t j = 0; j < forward.size(); ++j) {
        if (breaks[j + 1] - breaks[j] > 0) {
            integral.breaks.push_back(breaks[j]);
            // PPoly.antiderivative: divide by the rising factorials (3, 2, 1).
            Cubic c;
            c.c[3] = forward[j].c[2] / 3;
            c.c[2] = forward[j].c[1] / 2;
            c.c[1] = forward[j].c[0] / 1;
            integral.pieces.push_back(c);
        }
    }
    integral.breaks.push_back(breaks.back());
    // _ppoly.fix_continuity: each constant is the previous piece at its right end.
    for (std::size_t i = 1; i < integral.pieces.size(); ++i) {
        integral.pieces[i].c[0] = PiecewisePoly::value(
            integral.pieces[i - 1], integral.breaks[i] - integral.breaks[i - 1], 3);
    }
    return integral;
}

}  // namespace fts::fits
