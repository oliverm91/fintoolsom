// Bindings of the swap leg valuation (fintoolsom._native.SwapLegCore).
#include <cstdint>
#include <optional>
#include <utility>

#include <cmath>
#include <string>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include "dates/schedule.hpp"
#include "py_curve.hpp"
#include "py_dates.hpp"

namespace fts::py {

namespace {

// Fills `out` with the ordinals of `list`, a list of `n` exact dates; false otherwise.
bool fill_ordinals(nb::handle list, std::int64_t* out, Py_ssize_t n) {
    if (!PyList_CheckExact(list.ptr()) || PyList_GET_SIZE(list.ptr()) != n) return false;
    for (Py_ssize_t i = 0; i < n; ++i) {
        const auto o = dt::exact_ordinal(PyList_GET_ITEM(list.ptr(), i));
        if (!o) return false;
        out[i] = *o;
    }
    return true;
}

// SwapLeg.__post_init__'s arrays in one call: the ordinals of the start, end and (when
// given) payment and FX fixing dates, the amortizations (each residual minus the next,
// the last residual on the last coupon) and whether the leg pays on its end dates with
// each coupon starting where the previous one ends. None when the inputs are not plain
// (lists of exact dates of the residuals' length) or a date check fails, for the
// Python code to handle them and raise its errors.
nb::object leg_arrays(nb::handle start, nb::handle end, nb::handle pay, nb::handle fx,
                      DoublesIn residuals) {
    if (!PyList_CheckExact(start.ptr())) return nb::none();
    const Py_ssize_t n = PyList_GET_SIZE(start.ptr());
    if (static_cast<Py_ssize_t>(residuals.shape(0)) != n) return nb::none();
    const auto size = static_cast<std::size_t>(n);
    auto [start_arr, s] = new_array<std::int64_t>(size);
    auto [end_arr, e] = new_array<std::int64_t>(size);
    if (!fill_ordinals(start, s, n) || !fill_ordinals(end, e, n)) return nb::none();
    nb::object pay_obj = nb::none(), fx_obj = nb::none();
    const std::int64_t* p = e;
    if (!pay.is_none()) {
        auto [arr, buf] = new_array<std::int64_t>(size);
        if (!fill_ordinals(pay, buf, n)) return nb::none();
        pay_obj = nb::cast(std::move(arr));
        p = buf;
    }
    const std::int64_t* f = p;
    if (!fx.is_none()) {
        auto [arr, buf] = new_array<std::int64_t>(size);
        if (!fill_ordinals(fx, buf, n)) return nb::none();
        fx_obj = nb::cast(std::move(arr));
        f = buf;
    }
    bool par = n > 0;
    for (std::size_t i = 0; i < size; ++i) {
        if (s[i] >= e[i] || e[i] > p[i] || f[i] > p[i]) return nb::none();
        if (i > 0 && (s[i] < s[i - 1] || p[i] < p[i - 1])) return nb::none();
        par = par && p[i] == e[i] && (i == 0 || s[i] == e[i - 1]);
    }
    const double* r = residuals.data();
    auto [amort_arr, a] = new_array<double>(size);
    for (std::size_t i = 0; i + 1 < size; ++i) a[i] = r[i] - r[i + 1];
    if (size > 0) a[size - 1] = r[size - 1];
    return nb::make_tuple(std::move(start_arr), std::move(end_arr), pay_obj, fx_obj,
                          std::move(amort_arr), par);
}

// A list of the dates of `ordinals[first, last)`.
nb::list date_list(const std::vector<std::int64_t>& ordinals, std::size_t first, std::size_t last) {
    nb::list out;
    for (std::size_t i = first; i < last; ++i) out.append(dt::date_of(ordinals[i]));
    return out;
}

// A copy of `values` as a new numpy array.
nb::object array_of(const std::vector<std::int64_t>& values) {
    auto [array, out] = new_array<std::int64_t>(values.size());
    std::copy(values.begin(), values.end(), out);
    return nb::cast(std::move(array));
}

// A bullet leg's from_term in one call: the schedule (ScheduleGenerator.generate_schedule),
// the payment and FX fixing dates (SwapLeg._schedule_dates), then SwapLeg.__post_init__'s
// arrays (leg_arrays) and, given its rate, the fixed interest and flows. Returns
// (start dates, end dates, payment dates or None, FX fixing dates or None, residuals,
// (start, end, pay or None, fx or None, amortizations, par, interest or None, flows or
// None)); None when the Python code must run (rolling never reaches the end, a date
// check fails), interest None when `rate` is None, not one C++ reproduces, or gives a
// value that is not finite.
nb::object bullet_leg(nb::handle start, std::optional<std::string> maturity_unit,
                      std::int64_t maturity_value, const std::string& frequency_unit,
                      std::int64_t frequency_value, int adj_kind,
                      const dates::CalendarCore* adj_calendar, int maturity_adj_kind,
                      const dates::CalendarCore* maturity_calendar, nb::handle end_date,
                      bool stub_first, bool long_stub, std::int64_t payment_lag,
                      std::int64_t fx_fixing_lag, double notional, nb::handle rate) {
    auto unit_ok = [](const std::string& u) {
        return u.size() == 1 && std::string("dwmy").find(u[0]) != std::string::npos;
    };
    if (!unit_ok(frequency_unit) || (maturity_unit && !unit_ok(*maturity_unit)) ||
        adj_kind < 0 || adj_kind > 3 || maturity_adj_kind < 0 || maturity_adj_kind > 3 ||
        payment_lag < 0 || fx_fixing_lag < 0 || adj_calendar == nullptr ||
        maturity_calendar == nullptr) {
        return nb::none();
    }
    std::optional<dates::Tenor> maturity;
    if (maturity_unit) maturity = dates::Tenor{(*maturity_unit)[0], maturity_value, maturity_calendar};
    std::optional<std::int64_t> end;
    if (!end_date.is_none()) end = dt::ordinal_of(end_date);
    if (!maturity && !end) return nb::none();
    const dates::Convention adj{static_cast<dates::Adjustment>(adj_kind), adj_calendar};
    const auto schedule = dates::generate_schedule(
        dt::ordinal_of(start), maturity, dates::Tenor{frequency_unit[0], frequency_value, adj_calendar},
        adj, dates::Convention{static_cast<dates::Adjustment>(maturity_adj_kind), maturity_calendar},
        end, stub_first, long_stub);
    if (!schedule || schedule->empty()) return nb::none();
    const std::size_t n = schedule->size() - 1;
    const std::vector<std::int64_t> s(schedule->begin(), schedule->end() - 1),
        e(schedule->begin() + 1, schedule->end());
    std::vector<std::int64_t> p, f;
    if (payment_lag != 0) {
        p.reserve(n);
        for (const auto d : e) p.push_back(adj_calendar->add_business_days(d, payment_lag));
    }
    const std::vector<std::int64_t>& pay = payment_lag != 0 ? p : e;
    if (fx_fixing_lag != 0) {
        f.reserve(n);
        for (const auto d : pay) f.push_back(adj_calendar->add_business_days(d, -fx_fixing_lag));
    }
    const std::vector<std::int64_t>& fx = fx_fixing_lag != 0 ? f : pay;
    bool par = n > 0;
    for (std::size_t i = 0; i < n; ++i) {  // leg_arrays' checks
        if (s[i] >= e[i] || e[i] > pay[i] || fx[i] > pay[i]) return nb::none();
        if (i > 0 && (s[i] < s[i - 1] || pay[i] < pay[i - 1])) return nb::none();
        par = par && pay[i] == e[i] && (i == 0 || s[i] == e[i - 1]);
    }
    auto [residual_arr, r] = new_array<double>(n);
    auto [amort_arr, a] = new_array<double>(n);
    for (std::size_t i = 0; i < n; ++i) r[i] = notional;
    for (std::size_t i = 0; i + 1 < n; ++i) a[i] = r[i] - r[i + 1];
    if (n > 0) a[n - 1] = r[n - 1];
    nb::object interest_obj = nb::none(), flows_obj = nb::none();
    const auto terms = rate.is_none() ? std::nullopt : dt::rate_terms(rate);
    if (terms && n > 0) {  // Rate.get_accrued_interest(residuals, starts, ends)
        auto [interest_arr, in] = new_array<double>(n);
        auto [flows_arr, fl] = new_array<double>(n);
        bool finite = true;
        for (std::size_t i = 0; i < n && finite; ++i) {
            const double wf = terms->wealth_factor(s[i], e[i]);
            finite = std::isfinite(wf);
            in[i] = r[i] * (wf - 1.0);
            fl[i] = in[i] + a[i];
        }
        if (finite) {
            interest_obj = nb::cast(std::move(interest_arr));
            flows_obj = nb::cast(std::move(flows_arr));
        }
    }
    nb::object start_arr = array_of(s), end_arr = array_of(e);
    return nb::make_tuple(
        date_list(*schedule, 0, n), date_list(*schedule, 1, n + 1),
        payment_lag != 0 ? nb::object(date_list(p, 0, n)) : nb::none(),
        fx_fixing_lag != 0 ? nb::object(date_list(f, 0, n)) : nb::none(),
        std::move(residual_arr),
        nb::make_tuple(start_arr, end_arr,
                       payment_lag != 0 ? array_of(p) : nb::none(),
                       fx_fixing_lag != 0 ? array_of(f) : nb::none(), std::move(amort_arr), par,
                       interest_obj, flows_obj));
}

// curve_builder._seeds._Seeder.fixed_leg_zero's compounding: the product, left to
// right, of the wealth factors of the coupons ending after `t` (from max(start, t)),
// after the one from `t` to the first such start when it is later; with the ordinal of
// the last end. None when no coupon is left, the rate is not one C++ reproduces or a
// factor is not finite (Python does it then).
nb::object fixed_leg_wealth(std::int64_t t, nb::handle rate, DaysIn starts, DaysIn ends) {
    const auto terms = dt::rate_terms(rate);
    if (!terms || starts.shape(0) != ends.shape(0)) return nb::none();
    const std::int64_t* s = starts.data();
    const std::int64_t* e = ends.data();
    const std::size_t n = ends.shape(0);
    double wf = 1.0;
    bool any = false;
    std::int64_t last = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (e[i] <= t) continue;
        if (!any && s[i] > t) {
            const double gap = terms->wealth_factor(t, s[i]);
            if (!std::isfinite(gap)) return nb::none();
            wf *= gap;
        }
        any = true;
        const double f = terms->wealth_factor(std::max(s[i], t), e[i]);
        if (!std::isfinite(f)) return nb::none();
        wf *= f;
        last = e[i];
    }
    if (!any) return nb::none();
    return nb::make_tuple(wf, last);
}

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
    dt::import();
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
        "repays_at_par",
        [](DaysIn pay, DaysIn start, DaysIn end, DoublesIn residuals,
           DoublesIn amortizations) -> nb::object {
            const std::size_t n = pay.shape(0);
            if (start.shape(0) != n || end.shape(0) != n || residuals.shape(0) != n ||
                amortizations.shape(0) != n) {
                return nb::none();
            }
            return nb::bool_(swaps::repays_at_par(view(pay), view(start), view(end),
                                                  view(residuals), view(amortizations)));
        },
        nb::arg("pay"), nb::arg("start"), nb::arg("end"), nb::arg("residuals"),
        nb::arg("amortizations"),
        "FloatingLeg._repays_at_par on the leg's ordinals and notionals; None when the "
        "lengths differ.");

    m.def("fixed_leg_wealth", &fixed_leg_wealth, nb::arg("t"), nb::arg("rate"),
          nb::arg("start_ordinals"), nb::arg("end_ordinals"),
          "The compounded wealth factor of _Seeder.fixed_leg_zero and the last end "
          "ordinal, or None for Python to compute it.");

    m.def("bullet_leg", &bullet_leg, nb::arg("start"), nb::arg("maturity_unit").none(),
          nb::arg("maturity_value"), nb::arg("frequency_unit"), nb::arg("frequency_value"),
          nb::arg("adj_kind"), nb::arg("adj_calendar"), nb::arg("maturity_adj_kind"),
          nb::arg("maturity_calendar"), nb::arg("end_date").none(), nb::arg("stub_first"),
          nb::arg("long_stub"), nb::arg("payment_lag"), nb::arg("fx_fixing_lag"),
          nb::arg("notional"), nb::arg("rate").none(),
          "A bullet leg's schedule, dates and SwapLeg arrays in one call (see "
          "SwapLeg._bullet_parts), or None for the Python code to build it.");

    m.def("leg_arrays", &leg_arrays, nb::arg("start"), nb::arg("end"),
          nb::arg("pay").none(), nb::arg("fx").none(), nb::arg("residuals"),
          "SwapLeg's ordinals, amortizations and par-schedule flag in one call: "
          "(start, end, pay or None, fx or None, amortizations, par), or None when the "
          "inputs are not plain or a date check fails.");

    m.def(
        "fsum", [](DoublesIn values) { return swaps::fsum(view(values)); }, nb::arg("values"),
        "math.fsum of a float64 array (the C++ port, for tests).");
}

}  // namespace fts::py
