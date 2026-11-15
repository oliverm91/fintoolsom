// Bindings of the swap leg valuation (fintoolsom._native.SwapLegCore).
#include <cstdint>
#include <optional>
#include <utility>

#include <nanobind/stl/pair.h>

#include "py_curve.hpp"

namespace fts::py {

namespace {

using swaps::FxProjection;
using swaps::Known;
using swaps::LegKind;
using swaps::SwapLegCore;

Known known_of(const DaysIn& index, const DoublesIn& flows) {
    if (index.shape(0) != flows.shape(0)) {
        throw std::invalid_argument("known_index and known_flows differ in length.");
    }
    return Known{view(index), view(flows)};
}

}  // namespace

void bind_swaps(nb::module_& m) {
    nb::class_<PySwapLeg>(m, "SwapLegCore",
                          "Per-coupon data of a swap leg and its valuation arithmetic.")
        .def(
            "__init__",
            [](PySwapLeg* self, int kind, DaysIn start, DaysIn end, DaysIn pay,
               DaysIn fx_fixing, DaysIn fixing, DoublesIn residual, DoublesIn flows,
               DoublesIn spreads, bool par_eligible, bool has_spread, nb::object currency,
               nb::object index) {
                new (self) PySwapLeg(
                    SwapLegCore(static_cast<LegKind>(kind), view(start), view(end), view(pay),
                                view(fx_fixing), view(fixing), view(residual), view(flows),
                                view(spreads), par_eligible, has_spread),
                    std::move(currency), std::move(index));
            },
            nb::arg("kind"), nb::arg("start"), nb::arg("end"), nb::arg("pay"),
            nb::arg("fx_fixing"), nb::arg("fixing"), nb::arg("residual"), nb::arg("flows"),
            nb::arg("spreads"), nb::arg("par_eligible"), nb::arg("has_spread"),
            nb::arg("currency").none() = nb::none(), nb::arg("index").none() = nb::none())
        .def_prop_ro("size", [](const PySwapLeg& leg) { return leg.size(); })
        .def(
            "has_unpaid", [](const PySwapLeg& leg, std::int64_t t) { return leg.has_unpaid(t); },
            nb::arg("t"))
        .def(
            "nothing_realised",
            [](const PySwapLeg& leg, std::int64_t t) { return leg.nothing_realised(t); },
            nb::arg("t"))
        .def(
            "has_realised",
            [](const PySwapLeg& leg, std::int64_t t) { return leg.has_realised(t); },
            nb::arg("t"))
        .def(
            "known_fx_fixings",
            [](const PySwapLeg& leg, std::int64_t t, std::int64_t last_known) {
                bool any_unknown = false;
                const auto known = leg.known_fx_fixings(t, last_known, any_unknown);
                return std::make_pair(to_array<std::int64_t>(known), any_unknown);
            },
            nb::arg("t"), nb::arg("last_known"),
            "FX fixing ordinals of the unpaid coupons on or before last_known, and whether "
            "any unpaid coupon fixes later.")
        .def(
            "unpaid_flows",
            [](const PySwapLeg& leg, std::int64_t t, nb::handle proj, DaysIn known_index,
               DoublesIn known_flows) {
                const auto source = optional_source(proj);
                const auto flows = leg.unpaid_flows(t, source ? &*source : nullptr,
                                                    known_of(known_index, known_flows));
                return to_array<double>(flows);
            },
            nb::arg("t"), nb::arg("proj").none(), nb::arg("known_index"),
            nb::arg("known_flows"), "Flows of the coupons paid after t (_leg_flows).")
        .def(
            "par_pv",
            [](const PySwapLeg& leg, nb::handle curve) { return leg.par_pv(df_source(curve)); },
            nb::arg("curve"), "PV at par on the curve projecting and discounting the leg.")
        .def(
            "pv",
            [](const PySwapLeg& leg, std::int64_t t, nb::handle proj, DaysIn known_index,
               DoublesIn known_flows, nb::handle a, nb::handle b, nb::handle c) {
                const auto source = optional_source(proj);
                const DiscountCurves discount(a, b, c);
                return leg.pv(t, source ? &*source : nullptr, known_of(known_index, known_flows),
                              discount.get());
            },
            nb::arg("t"), nb::arg("proj").none(), nb::arg("known_index"),
            nb::arg("known_flows"), nb::arg("a"), nb::arg("b").none() = nb::none(),
            nb::arg("c").none() = nb::none(),
            "PV of the unpaid flows in the leg currency, discounted on a (or a * b / c).")
        .def(
            "pv_non_deliverable",
            [](const PySwapLeg& leg, std::int64_t t, nb::handle proj, DaysIn known_index,
               DoublesIn known_flows, std::int64_t last_known, DoublesIn known_fx,
               double anchor, nb::handle base, nb::handle quote, nb::handle a, nb::handle b,
               nb::handle c) {
                const auto source = optional_source(proj);
                const auto base_source = optional_source(base);
                const auto quote_source = optional_source(quote);
                const DiscountCurves discount(a, b, c);
                const FxProjection fx{last_known, view(known_fx), anchor,
                                      base_source ? &*base_source : nullptr,
                                      quote_source ? &*quote_source : nullptr};
                bool any_unknown = false;
                const auto known = leg.known_fx_fixings(t, last_known, any_unknown);
                if (known.size() != known_fx.shape(0) ||
                    (any_unknown && (fx.base == nullptr || fx.quote == nullptr))) {
                    throw std::invalid_argument(
                        "known_fx must hold the known fixings, and base and quote are "
                        "needed for the later ones.");
                }
                return leg.pv_non_deliverable(t, source ? &*source : nullptr,
                                              known_of(known_index, known_flows), fx,
                                              discount.get());
            },
            nb::arg("t"), nb::arg("proj").none(), nb::arg("known_index"),
            nb::arg("known_flows"), nb::arg("last_known"), nb::arg("known_fx"),
            nb::arg("anchor"), nb::arg("base").none(), nb::arg("quote").none(), nb::arg("a"),
            nb::arg("b").none() = nb::none(), nb::arg("c").none() = nb::none(),
            "PV in the settlement currency of the unpaid flows converted at their FX "
            "fixings and discounted on a (or a * b / c).")
        .def(
            "non_started_pv",
            [](const PySwapLeg& leg, nb::handle proj, nb::handle a, nb::handle b,
               nb::handle c) {
                const auto source = optional_source(proj);
                const DiscountCurves discount(a, b, c);
                return leg.non_started_pv(source ? &*source : nullptr, discount.get());
            },
            nb::arg("proj").none(), nb::arg("a"), nb::arg("b").none() = nb::none(),
            nb::arg("c").none() = nb::none(),
            "PV of every coupon projected (floating) and discounted, none realised.");

    m.def(
        "fsum", [](DoublesIn values) { return swaps::fsum(view(values)); }, nb::arg("values"),
        "math.fsum of a float64 array (the C++ port, for tests).");
}

}  // namespace fts::py
