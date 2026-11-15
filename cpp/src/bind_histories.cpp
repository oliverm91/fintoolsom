// Bindings of the market's fixing histories (fintoolsom.market.index_history) and the
// UF reajuste (fintoolsom.market.uf): the day-by-day walks that fill and compound the
// Python dicts in place, in the order and with the values of the Python loops.
//
// Each function either does the whole walk or, where it meets a value the Python
// code computes differently (another numeric type, a rate convention not ported, a
// result where Python raises), stops before changing anything that Python would not
// have changed, and says so: the Python loop then carries on.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

#include "dates/schedule.hpp"
#include "py_dates.hpp"

namespace fts::py {

namespace {

using dates::CalendarCore;
using dt::Ordinal;

// A float, or an int that converts exactly (|v| <= 2**53): the numbers whose Python
// arithmetic (int / int included) is the double arithmetic.
std::optional<double> exact_double(PyObject* x) {
    if (PyFloat_CheckExact(x)) return PyFloat_AS_DOUBLE(x);
    if (!PyLong_CheckExact(x)) return std::nullopt;
    const double v = PyLong_AsDouble(x);
    if (v == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        return std::nullopt;
    }
    if (std::fabs(v) > 9007199254740992.0) return std::nullopt;
    return v;
}

// dict.get(date) without raising: a borrowed reference or nullptr.
PyObject* get(nb::handle dict, Ordinal o) {
    const nb::object key = dt::date_of(o);
    PyObject* v = PyDict_GetItemWithError(dict.ptr(), key.ptr());
    if (v == nullptr && PyErr_Occurred()) throw nb::python_error();
    return v;
}

void set(nb::handle dict, Ordinal o, nb::handle value) {
    const nb::object key = dt::date_of(o);
    if (PyDict_SetItem(dict.ptr(), key.ptr(), value.ptr()) != 0) throw nb::python_error();
}

// OvernightRateHistory._fill_gaps (not verbose): from t to end_t, each next business
// day missing from `rates` repeats the fixing of the day before and joins `filled`.
// Returns the date the walk ends on.
nb::object fill_rate_gaps(nb::handle rates, nb::handle filled, nb::handle t_date,
                          nb::handle end_date, const CalendarCore& calendar) {
    if (!PyDict_CheckExact(rates.ptr()) || !PySet_CheckExact(filled.ptr())) {
        throw std::invalid_argument("fill_rate_gaps takes a dict and a set.");
    }
    Ordinal t = dt::ordinal_of(t_date);
    const Ordinal end = dt::ordinal_of(end_date);
    if (!(t < end)) return nb::borrow(t_date);
    while (t < end) {
        const Ordinal next = calendar.add_business_days(t, 1);
        if (get(rates, next) == nullptr) {
            PyObject* rate = get(rates, t);
            if (rate == nullptr) {  // self.rates[t]
                const nb::object key = dt::date_of(t);
                PyErr_SetObject(PyExc_KeyError, key.ptr());
                throw nb::python_error();
            }
            const nb::object keep = nb::borrow(rate);
            set(rates, next, keep);
            const nb::object key = dt::date_of(next);
            if (PySet_Add(filled.ptr(), key.ptr()) != 0) throw nb::python_error();
        }
        t = next;
    }
    return dt::date_of(t);
}

// OvernightRateHistory._compound_levels from t to end_t: levels[next] = levels[t] *
// r.get_wealth_factor(t, next), r the latest fixing on or before t. Returns (t, r)
// where it stopped: end_t, or the first step whose level, rate or result Python
// computes differently (the step is left undone), for the Python loop to carry on.
nb::object compound_levels(nb::handle levels, nb::handle rates, nb::handle t_date,
                           nb::handle end_date, const CalendarCore& calendar) {
    Ordinal t = dt::ordinal_of(t_date);
    const Ordinal end = dt::ordinal_of(end_date);
    nb::object r = nb::none();
    if (!PyDict_CheckExact(levels.ptr()) || !PyDict_CheckExact(rates.ptr()) || !(t < end)) {
        return nb::make_tuple(nb::borrow(t_date), r);
    }
    std::optional<dt::RateTerms> terms;
    while (t < end) {
        const Ordinal next = calendar.add_business_days(t, 1);
        nb::object step_r = r;
        std::optional<dt::RateTerms> step_terms = terms;
        if (PyObject* fixing = get(rates, t); fixing != nullptr) {
            step_r = nb::borrow(fixing);
            step_terms = dt::rate_terms(step_r);
        }
        PyObject* level = get(levels, t);
        const auto value = level == nullptr ? std::nullopt : exact_double(level);
        // An exponential rate's factors are np.float64, so would the levels be.
        if (step_r.is_none() || !value || !step_terms ||
            step_terms->interest == rates::Interest::Exponential) {
            break;
        }
        const double next_level = *value * step_terms->wealth_factor(t, next);
        if (!std::isfinite(next_level)) break;
        set(levels, next, nb::float_(next_level));
        r = std::move(step_r);
        terms = step_terms;
        t = next;
    }
    return nb::make_tuple(dt::date_of(t), r);
}

// The rescaling of OvernightRateHistory._replace_fixing: levels[d] *= ratio for the
// dates after `after`, newest first. False, with nothing changed, unless those levels
// are floats on exact dates.
bool scale_levels_after(nb::handle levels, nb::handle after_date, double ratio) {
    if (!PyDict_CheckExact(levels.ptr())) return false;
    const Ordinal after = dt::ordinal_of(after_date);
    std::vector<std::pair<PyObject*, PyObject*>> items;
    items.reserve(static_cast<std::size_t>(PyDict_GET_SIZE(levels.ptr())));
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(levels.ptr(), &pos, &key, &value)) items.emplace_back(key, value);
    std::vector<std::pair<PyObject*, double>> scaled;
    for (auto it = items.rbegin(); it != items.rend(); ++it) {
        const auto o = dt::exact_ordinal(it->first);
        if (!o) return false;
        if (*o <= after) break;
        if (!PyFloat_CheckExact(it->second)) return false;
        scaled.emplace_back(it->first, PyFloat_AS_DOUBLE(it->second) * ratio);
    }
    for (const auto& [k, v] : scaled) {
        const nb::object level = nb::float_(v);
        if (PyDict_SetItem(levels.ptr(), k, level.ptr()) != 0) throw nb::python_error();
    }
    return true;
}

// OvernightInterestPriceHistory._check_for_gaps (not verbose): each business day
// missing between consecutive known levels gets level[t0] * (level[t1] / level[t0]) **
// (days(t0, t) / days(t0, t1)). False, with nothing changed, for keys that are not
// exact dates, levels that are not floats (or exact ints), or a gap where Python
// raises or computes another type.
bool fill_price_gaps(nb::handle values, const CalendarCore& calendar) {
    if (!PyDict_CheckExact(values.ptr())) return false;
    struct Known {
        Ordinal o;
        double v;
    };
    std::vector<Known> known;
    known.reserve(static_cast<std::size_t>(PyDict_GET_SIZE(values.ptr())));
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(values.ptr(), &pos, &key, &value)) {
        const auto o = dt::exact_ordinal(key);
        const auto v = exact_double(value);
        if (!o || !v) return false;
        known.push_back(Known{*o, *v});
    }
    std::sort(known.begin(), known.end(), [](const Known& a, const Known& b) { return a.o < b.o; });
    std::vector<std::pair<Ordinal, double>> filled;
    for (std::size_t i = 0; i + 1 < known.size(); ++i) {
        const auto [t0, v0] = known[i];
        const auto [t1, v1] = known[i + 1];
        const double days_total = static_cast<double>(t1 - t0);
        if (v0 == 0.0) return false;  // ZeroDivisionError in Python
        const double ratio = v1 / v0;
        for (Ordinal t = calendar.add_business_days(t0, 1); t < t1;
             t = calendar.add_business_days(t, 1)) {
            if (get(values, t) != nullptr) continue;
            // float ** float: a complex for a negative ratio, OverflowError past range.
            if (!(ratio > 0.0) || !std::isfinite(ratio)) return false;
            const double growth = std::pow(ratio, static_cast<double>(t - t0) / days_total);
            const double level = v0 * growth;
            if (!std::isfinite(growth) || !std::isfinite(level)) return false;
            filled.emplace_back(t, level);
        }
    }
    for (const auto& [t, level] : filled) set(values, t, nb::float_(level));
    return true;
}

// round(x, 2) of Python: the correctly rounded 2-decimal string read back.
std::optional<double> round2(double x) {
    if (!std::isfinite(x)) return std::nullopt;
    char buf[400];
    const auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::fixed, 2);
    if (ec != std::errc{}) return std::nullopt;
    double out = 0.0;
    const auto [p, ec2] = std::from_chars(buf, end, out);
    if (ec2 != std::errc{} || p != end) return std::nullopt;
    return out;
}

// UFConvention.reajuste: round(base * (1 + cpi) ** (elapsed / period), 2).
std::optional<double> reajuste(double base, double cpi, Ordinal elapsed, Ordinal period) {
    const double growth = 1.0 + cpi;
    if (!(growth > 0.0) || !std::isfinite(growth) || period == 0) return std::nullopt;
    const double factor =
        std::pow(growth, static_cast<double>(elapsed) / static_cast<double>(period));
    if (!std::isfinite(factor)) return std::nullopt;
    return round2(base * factor);
}

// The loop of uf.extend_with_cpi: each day after the base 9th up to the next one,
// missing from `out`, gets its reajuste. False, with nothing changed, where Python
// would raise or compute another value.
bool uf_extend(nb::handle out, double base_uf, double cpi, Ordinal base, Ordinal next) {
    if (!PyDict_CheckExact(out.ptr())) return false;
    std::vector<std::pair<Ordinal, double>> filled;
    for (Ordinal d = base + 1; d <= next; ++d) {
        if (get(out, d) != nullptr) continue;
        const auto level = reajuste(base_uf, cpi, d - base, next - base);
        if (!level) return false;
        filled.emplace_back(d, *level);
    }
    for (const auto& [d, level] : filled) set(out, d, nb::float_(level));
    return true;
}

}  // namespace

void bind_histories(nb::module_& m) {
    dt::import();

    m.def("fill_rate_gaps", &fill_rate_gaps, nb::arg("rates"), nb::arg("filled"),
          nb::arg("t"), nb::arg("end_t"), nb::arg("calendar"),
          "OvernightRateHistory._fill_gaps on the index calendar's core.");
    m.def("compound_levels", &compound_levels, nb::arg("levels"), nb::arg("rates"),
          nb::arg("t"), nb::arg("end_t"), nb::arg("calendar"),
          "OvernightRateHistory._compound_levels up to where Python must carry on: "
          "(t, r) of that step.");
    m.def("scale_levels_after", &scale_levels_after, nb::arg("levels"), nb::arg("after"),
          nb::arg("ratio"), "The level rescaling of OvernightRateHistory._replace_fixing.");
    m.def("fill_price_gaps", &fill_price_gaps, nb::arg("values"), nb::arg("calendar"),
          "OvernightInterestPriceHistory._check_for_gaps on the index calendar's core.");
    m.def(
        "uf_reajuste",
        [](double base, double cpi, Ordinal elapsed, Ordinal period) -> nb::object {
            const auto level = reajuste(base, cpi, elapsed, period);
            if (!level) return nb::none();
            return nb::float_(*level);
        },
        nb::arg("base_level"), nb::arg("cpi"), nb::arg("elapsed_days"),
        nb::arg("period_days"), "UFConvention.reajuste; None where Python raises.");
    m.def("uf_extend", &uf_extend, nb::arg("out"), nb::arg("base_uf"), nb::arg("cpi"),
          nb::arg("base"), nb::arg("next_ninth"), "The loop of uf.extend_with_cpi.");
}

}  // namespace fts::py
