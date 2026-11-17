// Python bindings of the C++ kernels (module fintoolsom._native).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "py_curve.hpp"
#include "py_market.hpp"

namespace nb = nanobind;
using fts::Method;
using fts::PillarCurveCore;
using namespace fts::py;

namespace {

using GridOut = nb::ndarray<nb::numpy, const double, nb::ndim<1>>;

// A read-only numpy view of a shared grid; the capsule keeps the grid alive.
GridOut grid_view(const std::shared_ptr<const std::vector<double>>& grid) {
    auto* keep = new std::shared_ptr<const std::vector<double>>(grid);
    nb::capsule owner(keep, [](void* p) noexcept {
        delete static_cast<std::shared_ptr<const std::vector<double>>*>(p);
    });
    return GridOut(grid->data(), {grid->size()}, owner);
}

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "C++ kernels of fintoolsom (C++23, nanobind).";
    // Objects held by Python globals (a shared Calendar's core, a module-level curve)
    // can outlive the interpreter's last collection; that is not a leak to report.
    nb::set_leak_warnings(false);

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
            "set_forwards",
            [](PyPillarCurve& c, DoublesIn forwards, double log_df_effective) {
                c.core.set_forwards(view(forwards), log_df_effective);
            },
            nb::arg("forwards"), nb::arg("log_df_effective"),
            "Refit on new segment forwards, in place; drops the grid.")
        .def(
            "precalculate",
            [](PyPillarCurve& c, std::int64_t horizon) {
                return grid_view(c.core.precalculate(horizon));
            },
            nb::arg("horizon"), "Compute the daily grid of days [0, horizon] and return it.")
        .def(
            "forward_integral",
            [](const PyPillarCurve& c, DoublesIn tau) {
                auto [out, data] = new_array<double>(tau.shape(0));
                for (std::size_t i = 0; i < tau.shape(0); ++i) {
                    data[i] = c.core.forward_integral(tau.data()[i]);
                }
                return out;
            },
            "Integral of the forward from the first knot at each tau.");

    bind_swaps(m);
    bind_fx(m);
    bind_market(m);
    bind_dates(m);
    bind_rates(m);
    bind_histories(m);
    bind_solver(m);
}
