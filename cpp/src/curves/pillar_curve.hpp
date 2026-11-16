#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "curves/fits.hpp"
#include "curves/piecewise_poly.hpp"

namespace fts {

// fintoolsom.rates.InterpolationMethod values.
enum class Method : int {
    HermiteCubicSpline = 2,
    PiecewiseConstant = 3,
    PiecewiseLinear = 4,
    CubicSpline = 5,
    ConvexMonotone = 6,
};

// Day base of the curves' time axis (Act/365), as rates.Curve.DAY_BASE.
inline constexpr double kDayBase = 365.0;

// Numerical core of a fintoolsom PillarCurve: everything that turns day offsets into
// discount factors. Validation, dates and the curve's identity stay in Python.
//
// df(t) = exp(log_df_effective - I(t)), with I the integral of the forward from the
// first knot: I is linear in each segment (PiecewiseConstant) or given by a fitted
// piecewise polynomial, clipped to the knots and extended with the first forward before
// them and the last forward after them. A spot segment (IndexCurve with a spot rate)
// overrides days [0, first knot): their dfs come precomputed in `spot_dfs`; dfs of
// negative days on such a curve need the spot rate's own convention and are reported
// back to the caller instead.
class PillarCurveCore {
public:
    PillarCurveCore(std::span<const double> knot_tau, std::span<const double> forwards,
                    Method method, double log_df_effective, std::span<const double> spot_dfs);

    // Dfs at day offsets days[i] - shift (shift lets callers pass date ordinals). Uses the
    // daily grid when one is set. Returns the number of negative days on a curve with a
    // spot segment, whose dfs are left as NaN for the caller.
    std::size_t dfs_at_days(const std::int64_t* days, std::size_t n, std::int64_t shift,
                            double* out) const noexcept;

    // Same as dfs_at_days, always from the formula (never the grid).
    std::size_t dfs_from_days(const std::int64_t* days, std::size_t n, std::int64_t shift,
                              double* out) const noexcept;

    // New segment forwards (and the log df at the first knot), refitted as the
    // constructor fits them; the grid is dropped.
    void set_forwards(std::span<const double> forwards, double log_df_effective);

    // Daily dfs for days [0, horizon]: becomes the grid of later lookups.
    std::shared_ptr<const std::vector<double>> precalculate(std::int64_t horizon);

    [[nodiscard]] const std::shared_ptr<const std::vector<double>>& grid() const noexcept {
        return grid_;
    }
    [[nodiscard]] std::int64_t grid_horizon() const noexcept { return horizon_; }
    [[nodiscard]] bool has_spot() const noexcept { return !spot_dfs_.empty(); }

    // Integral of the forward from the first knot at `tau` (Act/365 years).
    [[nodiscard]] double forward_integral(double tau) const noexcept;

private:
    // cum_, the fit and the first and last forwards, from knot_tau_ and forwards_.
    void fit();
    [[nodiscard]] double df_formula(std::int64_t day) const noexcept;
    // Formula dfs for every day of [lo, hi), in order: one pass per segment.
    void fill_formula(std::int64_t lo, std::int64_t hi, double* out) const noexcept;
    void fill_constant_segments(std::int64_t lo, std::int64_t hi, double* out) const noexcept;
    void fill_fitted(std::int64_t lo, std::int64_t hi, double* out) const noexcept;
    void fill_bspline(std::int64_t lo, std::int64_t hi, double* out) const noexcept;

    Method method_;
    // Per-knot and per-segment state as flat arrays (structure of arrays).
    std::vector<double> knot_tau_;  // n knots
    std::vector<double> forwards_;  // n - 1 segment averages
    std::vector<double> cum_;       // n: integral at each knot
    PiecewisePoly fit_;             // integral (or df for Hermite) between knots
    fits::QuadraticBSpline bspline_;  // the integral for PiecewiseLinear, as scipy
    double log_df_effective_;
    double first_ = 0.0, last_ = 0.0, first_forward_ = 0.0, last_forward_ = 0.0;
    std::vector<double> spot_dfs_;  // dfs of days [0, first knot) when there is a spot rate
    std::shared_ptr<const std::vector<double>> grid_;
    std::int64_t horizon_ = -1;
};

}  // namespace fts
