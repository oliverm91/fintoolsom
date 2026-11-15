// Bindings of the calendars and schedule generation (fintoolsom._native.CalendarCore,
// generate_schedule).
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include "dates/day_counts.hpp"
#include "dates/schedule.hpp"
#include "py_curve.hpp"

#include <datetime.h>  // after Python.h

namespace fts::py {

namespace {

using dates::Adjustment;
using dates::CalendarCore;
using dates::Convention;
using dates::Ordinal;
using dates::Tenor;

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

Tenor tenor_of(const std::string& unit, std::int64_t value, const CalendarCore* calendar) {
    if (unit.size() != 1 || std::string("dwmy").find(unit[0]) == std::string::npos) {
        throw std::invalid_argument("Tenor unit must be one of d, w, m, y.");
    }
    return Tenor{unit[0], value, calendar};
}

Convention convention_of(int kind, const CalendarCore* calendar) {
    if (kind < 0 || kind > 3) throw std::invalid_argument("Unknown adjustment convention.");
    return Convention{static_cast<Adjustment>(kind), calendar};
}

// The ordinal of an exact `date`, or nullopt (a datetime, a string...).
std::optional<Ordinal> exact_ordinal(PyObject* d) {
    if (!PyDate_CheckExact(d)) return std::nullopt;
    return ordinal_of(nb::handle(d));
}

// The ordinals of `start` / `end`, a date or a non-empty list of exact dates; nullopt
// for anything else.
std::optional<std::vector<Ordinal>> exact_ordinals(nb::handle v) {
    if (!PyList_CheckExact(v.ptr())) {
        const auto o = exact_ordinal(v.ptr());
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

dates::DayCount day_count_of(int kind) {
    if (kind < 0 || kind > 4) throw std::invalid_argument("Unknown day count convention.");
    return static_cast<dates::DayCount>(kind);
}

// DayCountConventionBase.get_day_count for (date | list, date | list), or with `base`
// get_time_fraction (the counts divided by it): an int / float, or an array. None for
// inputs the Python mapper handles otherwise (other types, empty lists, unequal
// lengths), for the Python code to handle them and raise its errors.
nb::object day_counts(int kind, nb::handle start, nb::handle end, std::optional<double> base) {
    const dates::DayCount k = day_count_of(kind);
    const auto starts = exact_ordinals(start), ends = exact_ordinals(end);
    if (!starts || !ends) return nb::none();
    const bool start_list = PyList_CheckExact(start.ptr()), end_list = PyList_CheckExact(end.ptr());
    if (!start_list && !end_list) {
        const std::int64_t days = dates::day_count(k, starts->front(), ends->front());
        if (base) return nb::float_(static_cast<double>(days) / *base);
        return nb::int_(days);
    }
    if (start_list && end_list && starts->size() != ends->size()) return nb::none();
    const std::size_t n = std::max(starts->size(), ends->size());
    const auto at = [](const std::vector<Ordinal>& v, std::size_t i) {
        return v.size() == 1 ? v[0] : v[i];
    };
    if (base) {
        auto [array, out] = new_array<double>(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = static_cast<double>(dates::day_count(k, at(*starts, i), at(*ends, i))) / *base;
        }
        return nb::cast(array);
    }
    auto [array, out] = new_array<std::int64_t>(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = dates::day_count(k, at(*starts, i), at(*ends, i));
    return nb::cast(array);
}

}  // namespace

void bind_dates(nb::module_& m) {
    PyDateTime_IMPORT;
    if (PyDateTimeAPI == nullptr) throw nb::python_error();

    nb::class_<CalendarCore>(m, "CalendarCore",
                             "A Calendar's days, loaded from Python many years per call.")
        .def(
            "__init__",
            [](CalendarCore* self, nb::callable load, unsigned weekend_mask) {
                new (self) CalendarCore(
                    [load = std::move(load)](int first_year, int last_year) {
                        const nb::bytes days = nb::cast<nb::bytes>(load(first_year, last_year));
                        const auto* p = static_cast<const std::uint8_t*>(days.data());
                        return std::vector<std::uint8_t>(p, p + days.size());
                    },
                    weekend_mask);
            },
            nb::arg("load"), nb::arg("weekend_mask"),
            "load(first_year, last_year) -> bytes of per-day flags over those years (1 "
            "holiday, 2 weekend, 4 weekend workday); weekend_mask: bits of "
            "Calendar._weekend_weekdays.")
        .def_prop_ro("years", &CalendarCore::years,
                     "The years loaded, (first, last), or (0, -1) before any.")
        .def(
            "is_holiday",
            [](const CalendarCore& c, nb::handle d) { return c.is_holiday(ordinal_of(d)); },
            nb::arg("date"))
        .def(
            "add_business_days",
            [](const CalendarCore& c, nb::handle d, std::int64_t n) {
                return date_of(c.add_business_days(ordinal_of(d), n));
            },
            nb::arg("date"), nb::arg("business_days"))
        .def(
            "adjust",
            [](const CalendarCore& c, int kind, nb::handle d) {
                return date_of(convention_of(kind, &c).adjust(ordinal_of(d)));
            },
            nb::arg("kind"), nb::arg("date"),
            "The adjust method of convention `kind` (0 following, 1 modified following, 2 "
            "preceding, 3 modified preceding) on this calendar.")
        .def(
            "shift",
            [](const CalendarCore& c, const std::string& unit, std::int64_t value, nb::handle d,
               bool forward, std::int64_t periods) {
                const Tenor tenor = tenor_of(unit, value, &c);
                return date_of(tenor.unadjusted(ordinal_of(d), forward, periods));
            },
            nb::arg("unit"), nb::arg("value"), nb::arg("date"), nb::arg("forward") = true,
            nb::arg("periods") = 1,
            "Tenor.get_unadjusted_maturity: `date` moved by periods x value units (d: "
            "business days on this calendar, w, m, y).")
        .def(
            "advance",
            [](const CalendarCore& c, int kind, const std::string& unit, std::int64_t value,
               nb::handle d, bool forward) {
                const Ordinal raw = tenor_of(unit, value, &c).unadjusted(ordinal_of(d), forward, 1);
                return date_of(convention_of(kind, &c).adjust(raw));
            },
            nb::arg("kind"), nb::arg("unit"), nb::arg("value"), nb::arg("date"),
            nb::arg("forward") = true,
            "Tenor.get_adjusted_maturity and Term.advance: shift, then adjust by convention "
            "`kind` on this calendar.");

    m.def(
        "date_ordinals",
        [](nb::handle dates) {
            PyObject* seq = dates.ptr();
            const bool list = PyList_Check(seq);
            if (!list && !PyTuple_Check(seq)) throw nb::type_error("Expected a list or tuple.");
            const Py_ssize_t n = list ? PyList_GET_SIZE(seq) : PyTuple_GET_SIZE(seq);
            auto [array, out] = new_array<std::int64_t>(static_cast<std::size_t>(n));
            for (Py_ssize_t i = 0; i < n; ++i) {
                out[i] = ordinal_of(list ? PyList_GET_ITEM(seq, i) : PyTuple_GET_ITEM(seq, i));
            }
            return array;
        },
        nb::arg("dates"), "date_ordinals of a list or tuple: d.toordinal() of each, as int64.");

    m.def("day_counts", &day_counts, nb::arg("kind"), nb::arg("start"), nb::arg("end"),
          nb::arg("base").none() = nb::none(),
          "DayCountConventionBase.get_day_count of convention `kind` (0 actual, 1 30A, 2 30U, "
          "3 30E, 4 30E ISDA), or get_time_fraction with `base`; None for inputs left to "
          "Python.");

    m.def(
        "time_fraction",
        [](int kind, nb::handle start, nb::handle end, std::int64_t frequency,
           nb::handle period_start, nb::handle period_end) {
            if (kind < 0 || kind > 12) throw std::invalid_argument("Unknown time fraction.");
            const Ordinal a = ordinal_of(start), b = ordinal_of(end);
            const Ordinal ps = period_start.is_none() ? a : ordinal_of(period_start);
            const Ordinal pe = period_end.is_none() ? b : ordinal_of(period_end);
            const auto k = static_cast<dates::TimeFraction>(kind);
            if (k == dates::TimeFraction::ActualActualICMA && frequency * (pe - ps) == 0) {
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                throw nb::python_error();
            }
            return dates::time_fraction(k, a, b, frequency, ps, pe);
        },
        nb::arg("kind"), nb::arg("start"), nb::arg("end"), nb::arg("frequency") = 0,
        nb::arg("period_start").none() = nb::none(), nb::arg("period_end").none() = nb::none(),
        "TimeFractionBase.get_day_count_factor of time fraction `kind` (as numbered in "
        "fintoolsom.dates.time_fractions._KINDS).");

    m.def(
        "generate_schedule",
        [](nb::handle start, std::optional<std::string> maturity_unit,
           std::int64_t maturity_value, const std::string& frequency_unit,
           std::int64_t frequency_value, int adj_kind, const CalendarCore* adj_calendar,
           int maturity_adj_kind, const CalendarCore* maturity_calendar, nb::handle end_date,
           bool stub_first, bool long_stub) -> nb::object {
            std::optional<Tenor> maturity;
            if (maturity_unit) maturity = tenor_of(*maturity_unit, maturity_value, maturity_calendar);
            const Tenor frequency = tenor_of(frequency_unit, frequency_value, adj_calendar);
            std::optional<Ordinal> end;
            if (!end_date.is_none()) end = ordinal_of(end_date);
            if (!maturity && !end) {
                throw std::invalid_argument("A schedule needs a maturity tenor or an end date.");
            }
            const auto schedule = dates::generate_schedule(
                ordinal_of(start), maturity, frequency, convention_of(adj_kind, adj_calendar),
                convention_of(maturity_adj_kind, maturity_calendar), end, stub_first,
                long_stub);
            if (!schedule) return nb::none();
            nb::list out;
            for (const Ordinal o : *schedule) out.append(date_of(o));
            return out;
        },
        nb::arg("start"), nb::arg("maturity_unit").none(), nb::arg("maturity_value"),
        nb::arg("frequency_unit"), nb::arg("frequency_value"), nb::arg("adj_kind"),
        nb::arg("adj_calendar"), nb::arg("maturity_adj_kind"), nb::arg("maturity_calendar"),
        nb::arg("end_date").none(), nb::arg("stub_first"), nb::arg("long_stub"),
        "ScheduleGenerator.generate_schedule, or None where rolling never reaches the end "
        "(the Python one raises there).");
}

}  // namespace fts::py
