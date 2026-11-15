#include "curves/pillar_curve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "curves/fits.hpp"

namespace fts {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double tau_of(std::int64_t day) noexcept { return static_cast<double>(day) / kDayBase; }

// First day whose tau is >= `b`, with tau computed exactly as the lookups compute it.
std::int64_t first_day_at_or_after(double b) noexcept {
    auto d = static_cast<std::int64_t>(std::ceil(b * kDayBase));
    while (tau_of(d - 1) >= b) --d;
    while (tau_of(d) < b) ++d;
    return d;
}

// First day whose tau is > `b`.
std::int64_t first_day_after(double b) noexcept {
    auto d = first_day_at_or_after(b);
    while (tau_of(d) <= b) ++d;
    return d;
}

// One piece of a fitted curve over consecutive days. Coefficients are loop invariant
// and the degree is a template parameter, so the body has no branches and the
// compiler can vectorise the arithmetic.
template <int Degree, bool NegativeLog>
void fitted_run(const Cubic& p, double left_break, double log_df_effective,
                std::int64_t d0, std::int64_t d1, double* out) noexcept {
    for (std::int64_t d = d0; d < d1; ++d) {
        const double s = tau_of(d) - left_break;
        double res = 0.0;
        double z = 1.0;
        for (int k = 0; k <= Degree; ++k) {
            res = res + p.c[k] * z;
            if (k < Degree) z *= s;
        }
        if constexpr (NegativeLog) res = -std::log(res);
        out[d] = std::exp(log_df_effective - res);
    }
}

// The quadratic B-spline over consecutive days of one knot interval `ell`: de Boor's
// recursion (QuadraticBSpline::evaluate, nu = 0) unrolled for k = 2, with the same
// operations in the same order. The knots are loop invariant and the body branch-free,
// so the compiler can vectorise it. Repeated knots take the generic path.
void bspline_run(const fits::QuadraticBSpline& s, std::size_t ell, double log_df_effective,
                 std::int64_t d0, std::int64_t d1, double* out) noexcept {
    const double* t = s.t.data();
    const double t0 = t[ell - 1], t1 = t[ell], t2 = t[ell + 1], t3 = t[ell + 2];
    if (t2 == t1 || t2 == t0 || t3 == t1) {
        for (std::int64_t d = d0; d < d1; ++d) {
            out[d] = std::exp(log_df_effective - s.evaluate(tau_of(d), ell, 0));
        }
        return;
    }
    const double c0 = s.c[ell - 2], c1 = s.c[ell - 1], c2 = s.c[ell];
    const double w1 = 1.0 / (t2 - t1);
    const double d20 = t2 - t0, d31 = t3 - t1;
    for (std::int64_t d = d0; d < d1; ++d) {
        const double x = tau_of(d);
        // j = 1
        const double a0 = 0.0 + w1 * (t2 - x);
        const double a1 = w1 * (x - t1);
        // j = 2
        const double wa = a0 / d20;
        const double h0 = 0.0 + wa * (t2 - x);
        double h1 = wa * (x - t0);
        const double wb = a1 / d31;
        h1 = h1 + wb * (t3 - x);
        const double h2 = wb * (x - t1);
        double res = 0.0 + c0 * h0;
        res = res + c1 * h1;
        res = res + c2 * h2;
        out[d] = std::exp(log_df_effective - res);
    }
}

}  // namespace

PillarCurveCore::PillarCurveCore(std::span<const double> knot_tau,
                                 std::span<const double> forwards, Method method,
                                 double log_df_effective, std::span<const double> spot_dfs)
    : method_(method),
      knot_tau_(knot_tau.begin(), knot_tau.end()),
      forwards_(forwards.begin(), forwards.end()),
      log_df_effective_(log_df_effective),
      spot_dfs_(spot_dfs.begin(), spot_dfs.end()) {
    const std::size_t n = knot_tau_.size();
    if (n < 3 || forwards_.size() != n - 1) {
        throw std::invalid_argument("PillarCurveCore needs n >= 3 knots and n - 1 forwards.");
    }
    // np.concatenate(([0.0], np.cumsum(forward_rates * np.diff(knot_tau)))): sequential.
    cum_.resize(n);
    cum_[0] = 0.0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        cum_[i + 1] = cum_[i] + forwards_[i] * (knot_tau_[i + 1] - knot_tau_[i]);
    }
    first_ = knot_tau_.front();
    last_ = knot_tau_.back();
    first_forward_ = forwards_.front();
    last_forward_ = forwards_.back();
    switch (method_) {
        case Method::PiecewiseConstant:
            break;
        case Method::CubicSpline:
            fit_ = fits::natural_cubic_spline(knot_tau_, cum_);
            last_forward_ = fit_.derivative_at(last_);
            break;
        case Method::PiecewiseLinear:
            bspline_ = fits::linear_forward_integral(knot_tau_, forwards_);
            last_forward_ = bspline_.evaluate(last_, bspline_.interval(last_, 2), 1);
            break;
        case Method::HermiteCubicSpline: {
            std::vector<double> dfs(n);
            for (std::size_t i = 0; i < n; ++i) dfs[i] = std::exp(-cum_[i]);
            fit_ = fits::pchip(knot_tau_, dfs);
            break;  // the last forward stays the last segment average, as in Python
        }
        case Method::ConvexMonotone:
            fit_ = fits::convex_monotone_integral(knot_tau_, forwards_);
            last_forward_ = fit_.derivative_at(last_);
            break;
        default:
            throw std::invalid_argument("No forward integral defined for interpolation method " +
                                        std::to_string(static_cast<int>(method_)) + ".");
    }
}

double PillarCurveCore::forward_integral(double tau) const noexcept {
    if (method_ == Method::PiecewiseConstant) {
        // Segment searched among the inner knots only, as np.searchsorted(side="right").
        const auto first_inner = knot_tau_.begin() + 1;
        const auto idx = static_cast<std::size_t>(
            std::upper_bound(first_inner, knot_tau_.end() - 1, tau) - first_inner);
        return cum_[idx] + forwards_[idx] * (tau - knot_tau_[idx]);
    }
    const double x = std::min(std::max(tau, first_), last_);  // np.clip
    double value = method_ == Method::PiecewiseLinear ? bspline_(x) : fit_(x);
    if (method_ == Method::HermiteCubicSpline) value = -std::log(value);
    return value + first_forward_ * std::min(tau - first_, 0.0) +
           last_forward_ * std::max(tau - last_, 0.0);
}

double PillarCurveCore::df_formula(std::int64_t day) const noexcept {
    if (!spot_dfs_.empty() && day < static_cast<std::int64_t>(spot_dfs_.size())) {
        return day >= 0 ? spot_dfs_[static_cast<std::size_t>(day)] : kNaN;
    }
    return std::exp(log_df_effective_ - forward_integral(tau_of(day)));
}

std::size_t PillarCurveCore::dfs_from_days(const std::int64_t* days, std::size_t n,
                                           std::int64_t shift, double* out) const noexcept {
    std::size_t unresolved = 0;
    const bool spot = has_spot();
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t d = days[i] - shift;
        out[i] = df_formula(d);
        unresolved += static_cast<std::size_t>(spot && d < 0);
    }
    return unresolved;
}

std::size_t PillarCurveCore::dfs_at_days(const std::int64_t* days, std::size_t n,
                                         std::int64_t shift, double* out) const noexcept {
    if (!grid_) return dfs_from_days(days, n, shift, out);
    const double* grid = grid_->data();
    // Unsigned, a negative offset is huge: one comparison checks both ends.
    const auto horizon = static_cast<std::uint64_t>(horizon_);
    std::size_t unresolved = 0;
    const bool spot = has_spot();
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t d = days[i] - shift;
        if (static_cast<std::uint64_t>(d) <= horizon) {
            out[i] = grid[d];
        } else {
            out[i] = df_formula(d);
            unresolved += static_cast<std::size_t>(spot && d < 0);
        }
    }
    return unresolved;
}

std::shared_ptr<const std::vector<double>> PillarCurveCore::precalculate(std::int64_t horizon) {
    auto grid = std::make_shared<std::vector<double>>(static_cast<std::size_t>(horizon + 1));
    double* out = grid->data();
    std::int64_t start = 0;
    if (has_spot()) {
        start = std::min<std::int64_t>(static_cast<std::int64_t>(spot_dfs_.size()), horizon + 1);
        std::copy_n(spot_dfs_.begin(), start, out);
    }
    fill_formula(start, horizon + 1, out);
    grid_ = grid;
    horizon_ = horizon;
    return grid_;
}

void PillarCurveCore::fill_formula(std::int64_t lo, std::int64_t hi, double* out) const noexcept {
    if (lo >= hi) return;
    if (method_ == Method::PiecewiseConstant) {
        fill_constant_segments(lo, hi, out);
    } else if (method_ == Method::PiecewiseLinear) {
        fill_bspline(lo, hi, out);
    } else {
        fill_fitted(lo, hi, out);
    }
}

void PillarCurveCore::fill_constant_segments(std::int64_t lo, std::int64_t hi,
                                             double* out) const noexcept {
    const std::size_t n_segments = forwards_.size();
    const auto first_inner = knot_tau_.begin() + 1;
    auto idx = static_cast<std::size_t>(
        std::upper_bound(first_inner, knot_tau_.end() - 1, tau_of(lo)) - first_inner);
    std::int64_t d = lo;
    while (d < hi) {
        const std::int64_t end =
            idx + 1 < n_segments ? std::min(hi, first_day_at_or_after(knot_tau_[idx + 1])) : hi;
        const double c = cum_[idx];
        const double f = forwards_[idx];
        const double k = knot_tau_[idx];
        const double l = log_df_effective_;
        for (; d < end; ++d) out[d] = std::exp(l - (c + f * (tau_of(d) - k)));
        ++idx;
    }
}

void PillarCurveCore::fill_bspline(std::int64_t lo, std::int64_t hi,
                                   double* out) const noexcept {
    // De Boor's recursion per day; consecutive days reuse the interval as a hint.
    const std::int64_t in_lo = std::max(lo, first_day_at_or_after(first_));
    const std::int64_t in_hi = std::min(hi, first_day_after(last_));
    for (std::int64_t d = lo; d < std::min(in_lo, hi); ++d) out[d] = df_formula(d);
    for (std::int64_t d = std::max(in_hi, lo); d < hi; ++d) out[d] = df_formula(d);
    // One run per knot interval: days with t[ell] <= tau < t[ell + 1], the last interval
    // taking the rest (scipy's find_interval).
    const std::size_t last_interval = bspline_.t.size() - fits::QuadraticBSpline::k - 2;
    std::size_t ell = fits::QuadraticBSpline::k;
    for (std::int64_t d = in_lo; d < in_hi;) {
        ell = bspline_.interval(tau_of(d), ell);
        const std::int64_t end =
            ell == last_interval
                ? in_hi
                : std::min(in_hi, first_day_at_or_after(bspline_.t[ell + 1]));
        bspline_run(bspline_, ell, log_df_effective_, d, end, out);
        d = end;
    }
}

void PillarCurveCore::fill_fitted(std::int64_t lo, std::int64_t hi, double* out) const noexcept {
    // Before the first knot and past the last one: the scalar formula (a few days).
    const std::int64_t in_lo = std::max(lo, first_day_at_or_after(first_));
    const std::int64_t in_hi = std::min(hi, first_day_after(last_));
    for (std::int64_t d = lo; d < std::min(in_lo, hi); ++d) out[d] = df_formula(d);
    for (std::int64_t d = std::max(in_hi, lo); d < hi; ++d) out[d] = df_formula(d);
    if (in_lo >= in_hi) return;

    const bool negative_log = method_ == Method::HermiteCubicSpline;
    const std::size_t pieces = fit_.size();
    std::size_t i = fit_.piece_of(tau_of(in_lo));
    std::int64_t d = in_lo;
    while (d < in_hi) {
        const std::int64_t end =
            i + 1 < pieces ? std::min(in_hi, first_day_at_or_after(fit_.breaks[i + 1])) : in_hi;
        const Cubic& p = fit_.pieces[i];
        const double b = fit_.breaks[i];
        if (fit_.degree == 2) {
            fitted_run<2, false>(p, b, log_df_effective_, d, end, out);
        } else if (negative_log) {
            fitted_run<3, true>(p, b, log_df_effective_, d, end, out);
        } else {
            fitted_run<3, false>(p, b, log_df_effective_, d, end, out);
        }
        d = std::max(d, end);
        ++i;
    }
}

}  // namespace fts
