// Python bindings of the C++ kernels (module fintoolsom._native).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curves/pillar_curve.hpp"

namespace nb = nanobind;
using fts::Method;
using fts::PillarCurveCore;

namespace {

using DoublesIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using DaysIn = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using DoublesOut = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using GridOut = nb::ndarray<nb::numpy, const double, nb::ndim<1>>;

std::span<const double> view(const DoublesIn& a) { return {a.data(), a.shape(0)}; }

// A new numpy array owning a C++ buffer of n doubles.
std::pair<DoublesOut, double*> new_array(std::size_t n) {
    auto* buffer = new double[n];
    nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return {DoublesOut(buffer, {n}, owner), buffer};
}

// A read-only numpy view of a shared grid; the capsule keeps the grid alive.
GridOut grid_view(const std::shared_ptr<const std::vector<double>>& grid) {
    auto* keep = new std::shared_ptr<const std::vector<double>>(grid);
    nb::capsule owner(keep, [](void* p) noexcept {
        delete static_cast<std::shared_ptr<const std::vector<double>>*>(p);
    });
    return GridOut(grid->data(), {grid->size()}, owner);
}

// PillarCurveCore plus the Python fallback for the dfs it cannot resolve (negative days
// on a curve with a spot rate, which need the spot rate's own day count).
struct PyPillarCurve {
    PillarCurveCore core;
    nb::object spot_fallback;  // callable(days: int64 array) -> dfs, or None
    std::int64_t origin;       // ordinal of the curve date

    template <class Fn>
    DoublesOut evaluate(const DaysIn& days, std::int64_t shift, Fn&& fn) const {
        const std::size_t n = days.shape(0);
        auto [out, data] = new_array(n);
        const std::size_t unresolved = fn(days.data(), n, shift, data);
        if (unresolved > 0) resolve(days.data(), n, shift, data, unresolved);
        return out;
    }

    void resolve(const std::int64_t* days, std::size_t n, std::int64_t shift, double* out,
                 std::size_t unresolved) const {
        if (spot_fallback.is_none()) return;
        auto [negative, nd] = new_int_array(unresolved);
        std::vector<std::size_t> at;
        at.reserve(unresolved);
        for (std::size_t i = 0; i < n; ++i) {
            const std::int64_t d = days[i] - shift;
            if (d < 0) {
                nd[at.size()] = d;
                at.push_back(i);
            }
        }
        const auto dfs = nb::cast<DoublesIn>(spot_fallback(negative));
        for (std::size_t k = 0; k < at.size(); ++k) out[at[k]] = dfs.data()[k];
    }

    static std::pair<nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>, std::int64_t*>
    new_int_array(std::size_t n) {
        auto* buffer = new std::int64_t[n];
        nb::capsule owner(buffer,
                          [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
        return {nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>(buffer, {n}, owner), buffer};
    }
};

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "C++ kernels of fintoolsom (C++23, nanobind).";

    nb::class_<PyPillarCurve>(m, "PillarCurveCore",
                              "Numerical core of a PillarCurve: day offsets to discount factors.")
        .def(
            "__init__",
            [](PyPillarCurve* self, DoublesIn knot_tau, DoublesIn forwards, int method,
               double log_df_effective, DoublesIn spot_dfs, std::int64_t origin,
               nb::object spot_fallback) {
                new (self) PyPillarCurve{
                    PillarCurveCore(view(knot_tau), view(forwards), static_cast<Method>(method),
                                    log_df_effective, view(spot_dfs)),
                    std::move(spot_fallback), origin};
            },
            nb::arg("knot_tau"), nb::arg("forwards"), nb::arg("method"),
            nb::arg("log_df_effective"), nb::arg("spot_dfs"), nb::arg("origin"),
            nb::arg("spot_fallback").none())
        .def(
            "dfs_at_days",
            [](const PyPillarCurve& c, DaysIn days) {
                return c.evaluate(days, 0, [&](auto... a) { return c.core.dfs_at_days(a...); });
            },
            "Dfs at day offsets from the curve date (the grid when set).")
        .def(
            "dfs_at_ordinals",
            [](const PyPillarCurve& c, DaysIn ordinals) {
                return c.evaluate(ordinals, c.origin,
                                  [&](auto... a) { return c.core.dfs_at_days(a...); });
            },
            "Dfs at date ordinals (the grid when set).")
        .def(
            "dfs_from_days",
            [](const PyPillarCurve& c, DaysIn days) {
                return c.evaluate(days, 0,
                                  [&](auto... a) { return c.core.dfs_from_days(a...); });
            },
            "Dfs at day offsets from the formula, never the grid.")
        .def(
            "precalculate",
            [](PyPillarCurve& c, std::int64_t horizon) {
                return grid_view(c.core.precalculate(horizon));
            },
            nb::arg("horizon"), "Compute the daily grid of days [0, horizon] and return it.")
        .def(
            "forward_integral",
            [](const PyPillarCurve& c, DoublesIn tau) {
                auto [out, data] = new_array(tau.shape(0));
                for (std::size_t i = 0; i < tau.shape(0); ++i) {
                    data[i] = c.core.forward_integral(tau.data()[i]);
                }
                return out;
            },
            "Integral of the forward from the first knot at each tau.");
}
