#pragma once
// Shared by the binding units: the Python-facing curve core and numpy helpers.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <span>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curves/pillar_curve.hpp"
#include "swaps/swap_leg.hpp"

namespace fts::py {

namespace nb = nanobind;

using DoublesIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using DaysIn = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using DoublesOut = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using IntsOut = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>;

inline std::span<const double> view(const DoublesIn& a) { return {a.data(), a.shape(0)}; }
inline std::span<const std::int64_t> view(const DaysIn& a) { return {a.data(), a.shape(0)}; }

// A new numpy array owning a C++ buffer of n values.
template <class T>
std::pair<nb::ndarray<nb::numpy, T, nb::ndim<1>>, T*> new_array(std::size_t n) {
    auto* buffer = new T[n];
    nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return {nb::ndarray<nb::numpy, T, nb::ndim<1>>(buffer, {n}, owner), buffer};
}

// A numpy array holding a copy of `values`.
template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> to_array(std::span<const T> values) {
    auto [out, data] = new_array<T>(values.size());
    std::copy(values.begin(), values.end(), data);
    return out;
}

// PillarCurveCore plus the Python fallback for the dfs it cannot resolve (negative days
// on a curve with a spot rate, which need the spot rate's own day count).
struct PyPillarCurve {
    PillarCurveCore core;
    nb::object spot_fallback;  // callable(days: int64 array) -> dfs, or None
    std::int64_t origin;       // ordinal of the curve date

    // Dfs at days[i] - shift into out, through `fn` (grid or formula).
    template <class Fn>
    void evaluate_into(const std::int64_t* days, std::size_t n, std::int64_t shift, double* out,
                       Fn&& fn) const {
        const std::size_t unresolved = fn(days, n, shift, out);
        if (unresolved > 0) resolve(days, n, shift, out, unresolved);
    }

    template <class Fn>
    DoublesOut evaluate(const DaysIn& days, std::int64_t shift, Fn&& fn) const {
        const std::size_t n = days.shape(0);
        auto [out, data] = new_array<double>(n);
        evaluate_into(days.data(), n, shift, data, fn);
        return out;
    }

    // Dfs at date ordinals (the grid when set).
    void dfs_at_ordinals(std::span<const std::int64_t> ordinals, std::span<double> out) const {
        evaluate_into(ordinals.data(), ordinals.size(), origin, out.data(),
                      [this](auto... a) { return core.dfs_at_days(a...); });
    }

    void resolve(const std::int64_t* days, std::size_t n, std::int64_t shift, double* out,
                 std::size_t unresolved) const {
        if (spot_fallback.is_none()) return;
        auto [negative, nd] = new_array<std::int64_t>(unresolved);
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
};

// Dfs of a Python curve at date ordinals: its C++ core when it has one, else its
// _dfs_at_ordinals method (Python curves, legacy curves).
inline swaps::DfSource df_source(nb::handle curve) {
    nb::object core = nb::getattr(curve, "_core", nb::none());
    if (!core.is_none() && nb::isinstance<PyPillarCurve>(core)) {
        const PyPillarCurve* c = nb::inst_ptr<PyPillarCurve>(core);
        return [c, keep = std::move(core)](std::span<const std::int64_t> ordinals,
                                           std::span<double> out) {
            c->dfs_at_ordinals(ordinals, out);
        };
    }
    nb::object method = curve.attr("_dfs_at_ordinals");
    return [method = std::move(method)](std::span<const std::int64_t> ordinals,
                                        std::span<double> out) {
        auto arg = to_array<std::int64_t>(ordinals);
        const auto dfs = nb::cast<DoublesIn>(method(arg));
        if (dfs.shape(0) != out.size()) {
            throw std::runtime_error("_dfs_at_ordinals returned the wrong number of dfs.");
        }
        std::copy(dfs.data(), dfs.data() + out.size(), out.data());
    };
}

inline std::optional<swaps::DfSource> optional_source(nb::handle curve) {
    if (curve.is_none()) return std::nullopt;
    return df_source(curve);
}

// The curves of Market._dfs_at_ordinals: `a`, or a * b / c.
struct DiscountCurves {
    swaps::DfSource a;
    std::optional<swaps::DfSource> b, c;
    DiscountCurves(nb::handle a_, nb::handle b_, nb::handle c_)
        : a(df_source(a_)), b(optional_source(b_)), c(optional_source(c_)) {
        if (b.has_value() != c.has_value()) {
            throw std::invalid_argument("A collateral discount needs both b and c.");
        }
    }
    [[nodiscard]] swaps::Discount get() const {
        return swaps::Discount{&a, b ? &*b : nullptr, c ? &*c : nullptr};
    }
};

// A leg's SwapLegCore plus the leg's currency and index (None for a fixed leg), which
// the C++ market valuation reads without calling the leg's Python properties. Legs are
// immutable, so they never go stale.
struct PySwapLeg : swaps::SwapLegCore {
    nb::object currency;
    nb::object index;
    PySwapLeg(swaps::SwapLegCore core, nb::object currency_, nb::object index_)
        : swaps::SwapLegCore(std::move(core)),
          currency(std::move(currency_)),
          index(std::move(index_)) {}
};

void bind_swaps(nb::module_& m);
void bind_fx(nb::module_& m);

}  // namespace fts::py
