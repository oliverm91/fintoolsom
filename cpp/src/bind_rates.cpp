// Bindings of fintoolsom.rates beyond the pillar curves: Rate factors (interest
// conventions on day counts), curve year fractions and knot checks.
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "py_dates.hpp"

namespace fts::py {

namespace {

using dt::Ordinal;

// Rate.get_wealth_factor / get_discount_factor / get_accrued_interest on dates or
// lists of dates (as DayCountConventionBase.get_time_fraction maps them): a float or
// an array; None when the inputs are left to Python (types, empty or unequal lists, a
// result that is not finite, where Python may raise).
nb::object rate_factors(int what, int interest, double rate, int day_count, double base,
                        nb::handle start, nb::handle end, double notional) {
    const auto f = static_cast<rates::Factor>(what);
    const rates::Interest k = rates::interest_of(interest);
    if (day_count < 0 || day_count > 4) throw std::invalid_argument("Unknown day count.");
    const auto dc = static_cast<dates::DayCount>(day_count);
    const auto starts = dt::exact_ordinals(start), ends = dt::exact_ordinals(end);
    if (!starts || !ends) return nb::none();
    const bool start_list = PyList_CheckExact(start.ptr()), end_list = PyList_CheckExact(end.ptr());
    const auto value = [&](Ordinal a, Ordinal b) {
        const double tf = static_cast<double>(dates::day_count(dc, a, b)) / base;
        return rates::factor(f, rates::wealth_factor(k, rate, tf), notional);
    };
    if (!start_list && !end_list) {
        const double v = value(starts->front(), ends->front());
        if (!std::isfinite(v)) return nb::none();
        return nb::float_(v);
    }
    if (start_list && end_list && starts->size() != ends->size()) return nb::none();
    const std::size_t n = std::max(starts->size(), ends->size());
    auto [array, out] = new_array<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = value((*starts)[starts->size() == 1 ? 0 : i], (*ends)[ends->size() == 1 ? 0 : i]);
        if (!std::isfinite(out[i])) return nb::none();
    }
    return nb::cast(array);
}

// The exact dates of a list or tuple, or nullopt.
std::optional<std::vector<Ordinal>> date_sequence(nb::handle seq) {
    PyObject* s = seq.ptr();
    const bool list = PyList_CheckExact(s);
    if (!list && !PyTuple_CheckExact(s)) return std::nullopt;
    const Py_ssize_t n = list ? PyList_GET_SIZE(s) : PyTuple_GET_SIZE(s);
    std::vector<Ordinal> out(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        const auto o = dt::exact_ordinal(list ? PyList_GET_ITEM(s, i) : PyTuple_GET_ITEM(s, i));
        if (!o) return std::nullopt;
        out[static_cast<std::size_t>(i)] = *o;
    }
    return out;
}

// Curve.year_fractions: days from the curve date over 365.
nb::object year_fractions(nb::handle curve_date, nb::handle dates) {
    const auto origin = dt::exact_ordinal(curve_date);
    const auto days = date_sequence(dates);
    if (!origin || !days) return nb::none();
    auto [array, out] = new_array<double>(days->size());
    for (std::size_t i = 0; i < days->size(); ++i) {
        out[i] = static_cast<double>((*days)[i] - *origin) / 365.0;
    }
    return nb::cast(array);
}

// PillarCurve._check_knots (ordering part) and year_fractions in one call: the knots'
// year fractions when there are at least `min_knots`, strictly increasing, the first
// on or after the curve date; None otherwise (Python then raises its error).
nb::object knot_taus(nb::handle curve_date, nb::handle knots, std::size_t min_knots) {
    const auto origin = dt::exact_ordinal(curve_date);
    const auto days = date_sequence(knots);
    if (!origin || !days || days->size() < min_knots || days->front() < *origin) {
        return nb::none();
    }
    for (std::size_t i = 1; i < days->size(); ++i) {
        if ((*days)[i] <= (*days)[i - 1]) return nb::none();
    }
    auto [array, out] = new_array<double>(days->size());
    for (std::size_t i = 0; i < days->size(); ++i) {
        out[i] = static_cast<double>((*days)[i] - *origin) / 365.0;
    }
    return nb::cast(array);
}

}  // namespace

void bind_rates(nb::module_& m) {
    dt::import();

    m.def("rate_factors", &rate_factors, nb::arg("what"), nb::arg("interest"),
          nb::arg("rate"), nb::arg("day_count"), nb::arg("base"), nb::arg("start"),
          nb::arg("end"), nb::arg("notional") = 1.0,
          "Rate.get_wealth_factor (what 0), get_discount_factor (1) or "
          "get_accrued_interest (2, on `notional`) of a rate in interest convention "
          "`interest` (0 linear, 1 compounded, 2 exponential) and day count `day_count` "
          "over `base`; None for inputs left to Python.");
    m.def(
        "rate_from_wf",
        [](int interest, double wf, double time_fraction) {
            return rates::rate_from_wf(rates::interest_of(interest), wf, time_fraction);
        },
        nb::arg("interest"), nb::arg("wf"), nb::arg("time_fraction"),
        "InterestConventionBase.get_rate_from_wf, scalar.");
    m.def("year_fractions", &year_fractions, nb::arg("curve_date"), nb::arg("dates"),
          "Curve.year_fractions of a list or tuple of dates; None for other inputs.");
    m.def("knot_taus", &knot_taus, nb::arg("curve_date"), nb::arg("knots"),
          nb::arg("min_knots"),
          "The knots' year fractions when PillarCurve accepts their order and count, "
          "else None.");
}

}  // namespace fts::py
