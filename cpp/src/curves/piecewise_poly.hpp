#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace fts {

// One polynomial piece p(s) = c[0] + c[1] s + c[2] s^2 + c[3] s^3, with s the distance
// from the piece's left break. Pieces are stored as an array of 32-byte, aligned
// records (AoS): a point lookup reads one cache line, and the in-order sweep of the
// daily grid walks the records sequentially.
struct alignas(32) Cubic {
    double c[4] = {0.0, 0.0, 0.0, 0.0};
};

// Piecewise polynomial in the layout of scipy.interpolate.PPoly, evaluated with the
// same operation order (a power sum with a running power, not Horner), so values
// match scipy's to the last bit when the coefficients do.
struct PiecewisePoly {
    std::vector<double> breaks;  // n + 1, strictly increasing
    std::vector<Cubic> pieces;   // n
    int degree = 3;              // highest power with a coefficient

    [[nodiscard]] std::size_t size() const noexcept { return pieces.size(); }
    [[nodiscard]] double front() const noexcept { return breaks.front(); }
    [[nodiscard]] double back() const noexcept { return breaks.back(); }

    // Piece of x in [front, back]: breaks[i] <= x < breaks[i + 1]; back is in the last.
    [[nodiscard]] std::size_t piece_of(double x) const noexcept {
        const auto first_inner = breaks.begin() + 1;
        const auto it = std::upper_bound(first_inner, breaks.end() - 1, x);
        return static_cast<std::size_t>(it - first_inner);
    }

    // scipy _ppoly.evaluate_poly1 with dx = 0.
    [[nodiscard]] static double value(const Cubic& p, double s, int degree) noexcept {
        double res = 0.0;
        double z = 1.0;
        for (int k = 0; k <= degree; ++k) {
            res = res + p.c[k] * z;
            if (k < degree) z *= s;
        }
        return res;
    }

    // scipy _ppoly.evaluate_poly1 with dx = 1 (prefactor k, zero for the constant).
    [[nodiscard]] static double derivative(const Cubic& p, double s, int degree) noexcept {
        double res = 0.0;
        double z = 1.0;
        for (int k = 0; k <= degree; ++k) {
            const double prefactor = static_cast<double>(k);
            res = res + p.c[k] * z * prefactor;
            if (k < degree && k >= 1) z *= s;
        }
        return res;
    }

    [[nodiscard]] double operator()(double x) const noexcept {
        const std::size_t i = piece_of(x);
        return value(pieces[i], x - breaks[i], degree);
    }

    [[nodiscard]] double derivative_at(double x) const noexcept {
        const std::size_t i = piece_of(x);
        return derivative(pieces[i], x - breaks[i], degree);
    }
};

}  // namespace fts
