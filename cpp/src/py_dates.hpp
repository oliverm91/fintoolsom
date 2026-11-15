#pragma once
// Dates and Rate objects read from Python, for the binding units that need them.
//
// datetime.h gives each translation unit its own PyDateTimeAPI, so everything here has
// internal linkage, and each unit calls dt::import() in its bind function.

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

#include "dates/day_counts.hpp"
#include "py_curve.hpp"
#include "rates/interest.hpp"

#include <datetime.h>  // after Python.h

namespace fts::py::dt {
namespace {

using Ordinal = std::int64_t;

void import() {
    PyDateTime_IMPORT;
    if (PyDateTimeAPI == nullptr) throw nb::python_error();
}

// date.toordinal() of a date (or datetime: its date), else of d.toordinal().
Ordinal ordinal_of(nb::handle d) {
    if (!PyDate_Check(d.ptr())) return nb::cast<Ordinal>(d.attr("toordinal")());
    return dates::ordinal_of(std::chrono::year_month_day{
        std::chrono::year{PyDateTime_GET_YEAR(d.ptr())},
        std::chrono::month{static_cast<unsigned>(PyDateTime_GET_MONTH(d.ptr()))},
        std::chrono::day{static_cast<unsigned>(PyDateTime_GET_DAY(d.ptr()))}});
}

nb::object date_of(Ordinal o) {
    const auto ymd = dates::ymd_of(o);
    return nb::steal(PyDate_FromDate(static_cast<int>(ymd.year()),
                                     static_cast<int>(static_cast<unsigned>(ymd.month())),
                                     static_cast<int>(static_cast<unsigned>(ymd.day()))));
}

// The ordinal of an exact `date`, or nullopt (a datetime, a string...).
std::optional<Ordinal> exact_ordinal(nb::handle d) {
    if (!PyDate_CheckExact(d.ptr())) return std::nullopt;
    return ordinal_of(d);
}

// The ordinals of a date, or of a non-empty list of exact dates; nullopt otherwise.
std::optional<std::vector<Ordinal>> exact_ordinals(nb::handle v) {
    if (!PyList_CheckExact(v.ptr())) {
        const auto o = exact_ordinal(v);
        if (!o) return std::nullopt;
        return std::vector<Ordinal>{*o};
    }
    const Py_ssize_t n = PyList_GET_SIZE(v.ptr());
    if (n == 0) return std::nullopt;
    std::vector<Ordinal> out(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        const auto o = exact_ordinal(PyList_GET_ITEM(v.ptr(), i));
        if (!o) return std::nullopt;
        out[static_cast<std::size_t>(i)] = *o;
    }
    return out;
}

// A float from a Python int or float (exact types only), else nullopt.
std::optional<double> plain_number(nb::handle x) {
    if (PyFloat_CheckExact(x.ptr())) return PyFloat_AS_DOUBLE(x.ptr());
    if (PyLong_CheckExact(x.ptr())) {
        const double v = PyLong_AsDouble(x.ptr());
        if (v == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return std::nullopt;
        }
        return v;
    }
    return std::nullopt;
}

// ── Rates ───────────────────────────────────────────────────────────────────

// A Rate's terms: Rate.get_wealth_factor without Python.
struct RateTerms {
    rates::Interest interest;
    dates::DayCount day_count;
    double base;
    double rate;

    [[nodiscard]] double time_fraction(Ordinal start, Ordinal end) const {
        return static_cast<double>(dates::day_count(day_count, start, end)) / base;
    }
    [[nodiscard]] double wealth_factor(Ordinal start, Ordinal end) const {
        return rates::wealth_factor(interest, rate, time_fraction(start, end));
    }
};

struct RateRefs {
    PyObject* rate_type;       // fintoolsom.rates.Rates.Rate
    PyObject* interest_kinds;  // fintoolsom.rates.Rates._INTEREST_KINDS
    PyObject* day_count_kinds; // fintoolsom.dates.date_counts._KINDS
};

const RateRefs& rate_refs() {
    static const RateRefs r = [] {
        auto get = [](const char* module, const char* name) {
            return nb::object(nb::module_::import_(module).attr(name)).release().ptr();
        };
        return RateRefs{get("fintoolsom.rates.Rates", "Rate"),
                        get("fintoolsom.rates.Rates", "_INTEREST_KINDS"),
                        get("fintoolsom.dates.date_counts", "_KINDS")};
    }();
    return r;
}

std::optional<int> kind_in(PyObject* kinds, nb::handle cls) {
    PyObject* k = PyDict_GetItemWithError(kinds, cls.ptr());
    if (k == nullptr) {
        if (PyErr_Occurred()) throw nb::python_error();
        return std::nullopt;
    }
    return static_cast<int>(PyLong_AsLong(k));
}

// The terms of an exact Rate whose conventions, base and value C++ reproduces (an int
// or float rate, a nonzero int or float base, conventions of the ported classes);
// nullopt for anything else, which stays in Python.
std::optional<RateTerms> rate_terms(nb::handle rate) {
    const RateRefs& refs = rate_refs();
    if (Py_TYPE(rate.ptr()) != reinterpret_cast<PyTypeObject*>(refs.rate_type)) {
        return std::nullopt;
    }
    const auto value = plain_number(rate.attr("rate"));
    if (!value) return std::nullopt;
    nb::object convention = rate.attr("rate_convention");
    const auto base = plain_number(convention.attr("time_fraction_base"));
    if (!base || *base == 0.0) return std::nullopt;
    const auto interest = kind_in(refs.interest_kinds, convention.attr("interest_convention"));
    const auto day_count = kind_in(refs.day_count_kinds, convention.attr("day_count_convention"));
    if (!interest || !day_count) return std::nullopt;
    return RateTerms{static_cast<rates::Interest>(*interest),
                     static_cast<dates::DayCount>(*day_count), *base, *value};
}

}  // namespace
}  // namespace fts::py::dt
