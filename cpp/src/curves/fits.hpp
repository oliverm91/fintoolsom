#pragma once

#include <span>

#include "curves/piecewise_poly.hpp"

// The functions a PillarCurve integrates between knots, ported from the scipy calls
// the Python implementation makes, with the same arithmetic in the same order.
namespace fts::fits {

// scipy.interpolate.CubicSpline(x, y, bc_type="natural"): solved like scipy, with
// LAPACK dgtsv's elimination (partial pivoting) on the tridiagonal system.
PiecewisePoly natural_cubic_spline(std::span<const double> x, std::span<const double> y);

// scipy.interpolate.PchipInterpolator(x, y).
PiecewisePoly pchip(std::span<const double> x, std::span<const double> y);

// Quadratic B-spline (knots t, coefficients c), evaluated as scipy's BSpline: de Boor's
// recursion (fitpack _deBoor_D) and the same sum over the k + 1 basis values.
struct QuadraticBSpline {
    static constexpr int k = 2;
    std::vector<double> t;
    std::vector<double> c;

    // scipy _bspl.find_interval: t[l] <= x < t[l + 1], the last one including te.
    [[nodiscard]] std::size_t interval(double x, std::size_t hint) const noexcept;
    // Value (nu = 0) or first derivative (nu = 1) at x, using the interval `ell`.
    [[nodiscard]] double evaluate(double x, std::size_t ell, int nu) const noexcept;
    [[nodiscard]] double operator()(double x) const noexcept { return evaluate(x, interval(x, k), 0); }
};

// make_interp_spline(x, at_knots, k=1).antiderivative(), with at_knots the forwards at
// the knots of the continuous line with these segment averages.
QuadraticBSpline linear_forward_integral(std::span<const double> x,
                                         std::span<const double> averages);

// convex_monotone_forward(x, averages).antiderivative() (rates/_convex_monotone.py).
PiecewisePoly convex_monotone_integral(std::span<const double> x,
                                       std::span<const double> averages);

}  // namespace fts::fits
