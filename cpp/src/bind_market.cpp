// The market lookups of the valuation, in C++ (fintoolsom._native.market_*).
//
// A MarketView reads a Python fintoolsom.market.Market where it stands: its curves,
// FX histories, valuation date and risk-free index are the Python containers
// themselves, looked up through the C API, so the view can never go stale (callers
// mutate those dicts directly). Each lookup is the Python method's, in the same order,
// with the same errors and messages: Market.get_index_curve, get_discount_curve,
// _dfs_at_ordinals (its curves), get_fx_rate, get_fx_known_until and
// convert_value_currency. On top of it, a whole swap, simplified swap, forward or NDF is
// valued in one call, as the calculator functions do, and a batch of them in one call on
// one view (market_valuate_many).
//
// A view lives for one call. It remembers the curves, the known-until dates and the spot
// rates it has looked up (by the identity of the index, currency or pair asked for): the
// market cannot change during the call, so a batch looks each one up once.
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dates/schedule.hpp"
#include "fx/fx_derivatives.hpp"
#include "py_dates.hpp"  // datetime.h, after Python.h
#include "py_market.hpp"

namespace fts::py {

namespace {

using swaps::Discount;
using swaps::DfSource;
using swaps::FxProjection;
using swaps::Known;
using swaps::LegKind;

// ── Python objects used throughout (imported once, kept for the process) ────

struct Refs {
    PyObject* index_curve_type;  // fintoolsom.rates.IndexCurve
    PyObject* currency_type;     // fintoolsom.market.currencies.Currency
    PyObject* currency_pair;     // fintoolsom.market.currencies.CurrencyPair
    PyObject* fx_rate;           // fintoolsom.market.currencies.FX_Rate
    PyObject* leg_core;          // _native_swap_pv._core
    PyObject* projection;        // _native_swap_pv._projection
    PyObject* overnight_rate_history;   // fintoolsom.market.index_history
    PyObject* overnight_price_history;  // OvernightInterestPriceHistory
    PyObject* term_rate_history;        // TermRateHistory
};

const Refs& refs() {
    static const Refs r = [] {
        auto get = [](const char* module, const char* name) {
            return nb::object(nb::module_::import_(module).attr(name)).release().ptr();
        };
        const char* currencies = "fintoolsom.market.currencies";
        const char* glue = "fintoolsom.derivatives.calculator._native_swap_pv";
        const char* histories = "fintoolsom.market.index_history";
        return Refs{get("fintoolsom.rates", "IndexCurve"),
                    get(currencies, "Currency"),
                    get(currencies, "CurrencyPair"),
                    get(currencies, "FX_Rate"),
                    get(glue, "_core"),
                    get(glue, "_projection"),
                    get(histories, "OvernightRateHistory"),
                    get(histories, "OvernightInterestPriceHistory"),
                    get(histories, "TermRateHistory")};
    }();
    return r;
}

[[noreturn]] void throw_py(PyObject* type, const nb::str& message) {
    PyErr_SetObject(type, message.ptr());
    throw nb::python_error();
}

template <class... Args>
nb::str fmt_str(const char* fmt, Args&&... args) {
    return nb::borrow<nb::str>(nb::str(fmt).attr("format")(std::forward<Args>(args)...));
}

// ── Dates ───────────────────────────────────────────────────────────────────

constexpr std::int64_t kOrdinal1970 = 719163;  // date(1970, 1, 1).toordinal()

std::int64_t ordinal_of(nb::handle d) {
    if (!PyDate_Check(d.ptr())) return nb::cast<std::int64_t>(d.attr("toordinal")());
    const std::chrono::year_month_day ymd{std::chrono::year{PyDateTime_GET_YEAR(d.ptr())},
                                          std::chrono::month{static_cast<unsigned>(
                                              PyDateTime_GET_MONTH(d.ptr()))},
                                          std::chrono::day{static_cast<unsigned>(
                                              PyDateTime_GET_DAY(d.ptr()))}};
    return std::chrono::sys_days{ymd}.time_since_epoch().count() + kOrdinal1970;
}

nb::object date_of(std::int64_t ordinal) {
    const std::chrono::year_month_day ymd{
        std::chrono::sys_days{std::chrono::days{ordinal - kOrdinal1970}}};
    return nb::steal(PyDate_FromDate(static_cast<int>(ymd.year()),
                                     static_cast<int>(static_cast<unsigned>(ymd.month())),
                                     static_cast<int>(static_cast<unsigned>(ymd.day()))));
}

// ── Currencies ──────────────────────────────────────────────────────────────

// Currency.__eq__: another Currency of the same CurrencyName (an Enum: identity).
bool same_currency(nb::handle a, nb::handle b) {
    if (a.is(b)) return true;
    if (a.is_none() || b.is_none()) return false;
    if (!PyObject_IsInstance(b.ptr(), refs().currency_type)) return false;
    return a.attr("name").is(b.attr("name"));
}

// Currency.settlement_currency.
nb::object settlement_of(nb::handle currency) {
    nb::object settles_in = currency.attr("settles_in");
    if (settles_in.is_none()) return nb::borrow(currency);
    return settles_in;
}

// CurrencyPair(base, quote), one object per pair of currency names for the process.
nb::handle currency_pair(nb::handle base, nb::handle quote) {
    struct Hash {
        std::size_t operator()(const std::pair<PyObject*, PyObject*>& k) const noexcept {
            return std::hash<PyObject*>{}(k.first) * 31 + std::hash<PyObject*>{}(k.second);
        }
    };
    static std::unordered_map<std::pair<PyObject*, PyObject*>, PyObject*, Hash> pairs;
    const nb::object bn = base.attr("name"), qn = quote.attr("name");
    const auto key = std::make_pair(bn.ptr(), qn.ptr());
    if (const auto it = pairs.find(key); it != pairs.end()) return it->second;
    // Keep the names alive with the pair: the key holds their addresses.
    bn.inc_ref();
    qn.inc_ref();
    PyObject* pair = nb::handle(refs().currency_pair)(base, quote).release().ptr();
    pairs.emplace(key, pair);
    return pair;
}

// ── The market ──────────────────────────────────────────────────────────────

class MarketView {
public:
    // `risk_free_index`: the numeraire of the discount curves, None for the market's.
    explicit MarketView(nb::handle market, nb::handle risk_free_index = nb::none())
        : market_(market),
          curves_(market.attr("curves")),
          fx_history_(market.attr("fx_history")),
          t_date_(market.attr("t")),
          rfi_(risk_free_index.is_none() ? nb::object(market.attr("risk_free_index"))
                                         : nb::borrow(risk_free_index)),
          t_(ordinal_of(t_date_)) {}

    [[nodiscard]] nb::handle market() const { return market_; }
    [[nodiscard]] std::int64_t t() const { return t_; }
    [[nodiscard]] nb::handle t_date() const { return t_date_; }

    // Market.get_index_curve.
    [[nodiscard]] nb::object index_curve(nb::handle index) const {
        if (PyObject* hit = memo_find(index_curves_, index)) return nb::borrow(hit);
        nb::object curve = index_curve_lookup(index);
        index_curves_.emplace_back(nb::borrow(index), curve);
        return curve;
    }

    // Market.get_discount_curve, under the market's risk-free index.
    [[nodiscard]] nb::object discount_curve(nb::handle currency) const {
        if (PyObject* hit = memo_find(discount_curves_, currency)) return nb::borrow(hit);
        nb::object curve = discount_curve_lookup(currency);
        discount_curves_.emplace_back(nb::borrow(currency), curve);
        return curve;
    }

    [[nodiscard]] nb::object index_curve_lookup(nb::handle index) const {
        const nb::object name = index.attr("name");
        PyObject* curve = PyDict_GetItemWithError(curves_.ptr(), name.ptr());
        if (curve == nullptr && PyErr_Occurred()) throw nb::python_error();
        if (curve == nullptr || !PyObject_IsInstance(curve, refs().index_curve_type)) {
            throw_py(PyExc_KeyError, fmt_str("No index curve for '{}' in market.", name));
        }
        return nb::borrow(curve);
    }

    [[nodiscard]] nb::object discount_curve_lookup(nb::handle currency) const {
        if (rfi_.is_none()) {
            throw_py(PyExc_ValueError,
                  nb::str("Market has no risk_free_index; cannot resolve discount curves."));
        }
        if (same_currency(currency, rfi_.attr("currency"))) return index_curve(rfi_);
        const nb::str id = fmt_str("{}_{}", currency.attr("value"), rfi_.attr("name"));
        PyObject* curve = PyDict_GetItemWithError(curves_.ptr(), id.ptr());
        if (curve == nullptr) {
            if (PyErr_Occurred()) throw nb::python_error();
            throw_py(PyExc_KeyError, fmt_str("No discount curve '{}' in market.", id));
        }
        return nb::borrow(curve);
    }

    // The curves of Market._dfs_at_ordinals, resolved in its order.
    [[nodiscard]] DiscountCurves discount(nb::handle currency, nb::handle collateral) const {
        if (collateral.is_none()) {
            return DiscountCurves(discount_curve(currency), nb::none(), nb::none());
        }
        if (same_currency(collateral.attr("currency"), currency)) {
            return DiscountCurves(index_curve(collateral), nb::none(), nb::none());
        }
        nb::object a = discount_curve(currency);
        nb::object b = index_curve(collateral);
        nb::object c = discount_curve(collateral.attr("currency"));
        return DiscountCurves(a, b, c);
    }

    // _swap_pv._discounted_on_index_curve.
    [[nodiscard]] bool discounted_on_index_curve(nb::handle currency, nb::handle index,
                                                 nb::handle collateral) const {
        nb::handle on = collateral.is_none() ? nb::handle(rfi_) : collateral;
        if (on.is_none()) return false;
        if (!same_currency(on.attr("currency"), currency)) return false;
        return on.attr("name").equal(index.attr("name"));
    }

    // Market.get_fx_rate(t, pair).value.
    [[nodiscard]] double fx_rate(nb::handle t, nb::handle pair) const {
        if (same_currency(pair.attr("base_currency"), pair.attr("quote_currency"))) return 1.0;
        PyObject* data = PyDict_GetItemWithError(fx_history_.ptr(), pair.ptr());
        if (data == nullptr) {
            if (PyErr_Occurred()) throw nb::python_error();
            throw_py(PyExc_KeyError, fmt_str("No {} data loaded in market.", pair));
        }
        // FX_RateData.get_fx_rate(t)
        const nb::object history = nb::handle(data).attr("historic_data");
        PyObject* rate = PyDict_GetItemWithError(history.ptr(), t.ptr());
        if (rate == nullptr) {
            if (PyErr_Occurred()) throw nb::python_error();
            throw_py(PyExc_KeyError, fmt_str("Date {} not found in {} data.", t,
                                         nb::handle(data).attr("currency_pair")));
        }
        return nb::cast<double>(nb::handle(rate).attr("value"));
    }

    // Market.get_fx_known_until(pair).
    [[nodiscard]] nb::object fx_known_until(nb::handle pair) const {
        if (PyObject* hit = memo_find(known_until_, pair)) return nb::borrow(hit);
        nb::object until = fx_known_until_lookup(pair);
        known_until_.emplace_back(nb::borrow(pair), until);
        return until;
    }

    [[nodiscard]] nb::object fx_known_until_lookup(nb::handle pair) const {
        PyObject* data = PyDict_GetItemWithError(fx_history_.ptr(), pair.ptr());
        if (data == nullptr && PyErr_Occurred()) throw nb::python_error();
        if (data == nullptr) return nb::borrow(t_date_);
        const nb::object history = nb::handle(data).attr("historic_data");
        if (PyDict_GET_SIZE(history.ptr()) == 0) return nb::borrow(t_date_);
        // max(self.t, max(data.historic_data))
        nb::object latest = nb::borrow(t_date_);
        std::int64_t latest_ordinal = t_;
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        bool first = true;
        nb::object key_max;
        std::int64_t key_max_ordinal = 0;
        while (PyDict_Next(history.ptr(), &pos, &key, &value)) {
            const std::int64_t o = ordinal_of(key);
            if (first || o > key_max_ordinal) {
                key_max = nb::borrow(key);
                key_max_ordinal = o;
                first = false;
            }
        }
        if (key_max_ordinal > latest_ordinal) return key_max;
        return latest;
    }

    // Market.convert_value_currency(value, from, to).
    // The pair comes from currency_pair: one object per pair for the process.
    [[nodiscard]] double convert(double value, nb::handle from, nb::handle to) const {
        const nb::handle pair = currency_pair(from, to);
        for (const auto& [p, rate] : spot_) {
            if (p == pair.ptr()) return value * rate;
        }
        const double rate = fx_rate(t_date_, pair);
        spot_.emplace_back(pair.ptr(), rate);
        return value * rate;
    }

    // Market.get_index(index.name): the history holding the index's fixings.
    [[nodiscard]] nb::object index_history(nb::handle index) const {
        const nb::object name = index.attr("name").attr("upper")();
        const nb::object histories = market_.attr("indexes_history");
        PyObject* history = PyDict_GetItemWithError(histories.ptr(), name.ptr());
        if (history == nullptr) {
            if (PyErr_Occurred()) throw nb::python_error();
            throw_py(PyExc_KeyError, fmt_str("Index {} not found in market.", name));
        }
        return nb::borrow(history);
    }

    // Market.get_rate(t, name, use_closest_past_rate).
    [[nodiscard]] nb::object rate(std::int64_t t, nb::handle name, bool closest) const {
        const nb::object rates = market_.attr("interest_rates");
        PyObject* series = PyDict_GetItemWithError(rates.ptr(), name.ptr());
        if (series == nullptr) {
            if (PyErr_Occurred()) throw nb::python_error();
            throw_py(PyExc_ValueError, fmt_str("Rate {} not found.", name));
        }
        const nb::object day = date_of(t);
        PyObject* rate = PyDict_GetItemWithError(series, day.ptr());
        if (rate != nullptr) return nb::borrow(rate);
        if (PyErr_Occurred()) throw nb::python_error();
        if (!closest) throw_py(PyExc_ValueError, fmt_str("Rate {} not found for date {}.", name, day));
        PyObject *key, *value, *best = nullptr;
        std::int64_t best_ordinal = 0;
        Py_ssize_t pos = 0;
        while (PyDict_Next(series, &pos, &key, &value)) {
            const std::int64_t o = ordinal_of(key);
            if (o < t && (best == nullptr || o > best_ordinal)) {
                best = value;
                best_ordinal = o;
            }
        }
        if (best == nullptr) {
            throw_py(PyExc_ValueError,
                     fmt_str("Rate {} not found for date {} and there were no past dates.",
                             name, day));
        }
        return nb::borrow(best);
    }

    // Market._dfs_at_ordinals.
    [[nodiscard]] DoublesOut dfs(nb::handle currency, std::span<const std::int64_t> ordinals,
                                 nb::handle collateral) const {
        const DiscountCurves curves = discount(currency, collateral);
        auto [out, data] = new_array<double>(ordinals.size());
        curves.get()(ordinals, std::span(data, ordinals.size()));
        return out;
    }

    // Market._future_fx_rates_at_ordinals.
    [[nodiscard]] DoublesOut future_fx(nb::handle pair, std::span<const std::int64_t> ordinals) const {
        const std::size_t n = ordinals.size();
        auto [out, fx] = new_array<double>(n);
        const nb::object base = pair.attr("base_currency"), quote = pair.attr("quote_currency");
        if (same_currency(base, quote)) {
            std::fill(fx, fx + n, 1.0);
            return out;
        }
        const nb::object known_until = fx_known_until(pair);
        const std::int64_t last_known = ordinal_of(known_until);
        std::vector<std::int64_t> nodes{last_known};
        for (std::size_t i = 0; i < n; ++i) {
            if (ordinals[i] <= last_known) fx[i] = fx_rate(date_of(ordinals[i]), pair);
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (ordinals[i] > last_known) nodes.push_back(ordinals[i]);
        }
        if (nodes.size() > 1) {
            const double anchor = fx_rate(known_until, pair);
            std::vector<double> df_base(nodes.size()), df_quote(nodes.size());
            df_source(discount_curve(base))(nodes, df_base);
            df_source(discount_curve(quote))(nodes, df_quote);
            std::size_t k = 1;
            for (std::size_t i = 0; i < n; ++i) {
                if (ordinals[i] <= last_known) continue;
                fx[i] = anchor * (df_base[k] / df_base[0]) / (df_quote[k] / df_quote[0]);
                ++k;
            }
        }
        return out;
    }

private:
    using Memo = std::vector<std::pair<nb::object, nb::object>>;
    static PyObject* memo_find(const Memo& memo, nb::handle key) {
        for (const auto& [k, v] : memo) {
            if (k.is(key)) return v.ptr();
        }
        return nullptr;
    }

    nb::handle market_;
    nb::object curves_, fx_history_, t_date_, rfi_;
    std::int64_t t_;
    // Lookups already made, for the call this view lives in (few entries: linear scans).
    mutable Memo index_curves_, discount_curves_, known_until_;
    mutable std::vector<std::pair<const PyObject*, double>> spot_;
};

// ── Swap legs (_native_swap_pv, with the lookups here) ─────────────────────

struct Leg {
    nb::object core_obj;
    const PySwapLeg* core;
    [[nodiscard]] bool fixed() const { return core->kind() == LegKind::Fixed; }
};

Leg leg_of(nb::handle leg) {
    nb::object core = leg.attr("_core");
    if (core.is_none()) core = nb::handle(refs().leg_core)(leg);
    const PySwapLeg* p = nb::cast<const PySwapLeg*>(core);
    return Leg{std::move(core), p};
}

// _native_swap_pv._projection: the index curve, then the realised flows of the coupons
// started or fixed by t.
struct Projection {
    std::optional<DfSource> curve;
    std::vector<std::int64_t> index;  // coupons with realised flows
    std::vector<double> flows;
    [[nodiscard]] const DfSource* get() const { return curve ? &*curve : nullptr; }
    [[nodiscard]] Known known() const { return Known{index, flows}; }
};

// levels[o] of a history's level dict (exact float or int values); KeyError(date) when
// missing, as the Python lookup; nullopt for values left to Python.
std::optional<double> level_at(nb::handle levels, std::int64_t o) {
    const nb::object day = date_of(o);
    PyObject* v = PyDict_GetItemWithError(levels.ptr(), day.ptr());
    if (v == nullptr) {
        if (PyErr_Occurred()) throw nb::python_error();
        PyErr_SetObject(PyExc_KeyError, day.ptr());
        throw nb::python_error();
    }
    return dt::plain_number(v);
}

// The realised flows of _native_swap_pv._projection (_swap_pv._running_flows for an
// overnight leg, the fixed rates of a term-rate leg), read from the market's index
// history: an exact OvernightRateHistory or OvernightInterestPriceHistory (overnight),
// TermRateHistory or OvernightRateHistory (term rate). False, with nothing computed,
// for any other history, value or rate, which the Python callback then reads.
bool realised_flows(const PySwapLeg& core, const MarketView& m, nb::handle proj,
                    Projection& p) {
    if (!PyDate_CheckExact(m.t_date().ptr())) return false;
    const std::int64_t t = m.t();
    const nb::object history = m.index_history(core.index);  // Market.get_index
    const PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(history.ptr()));
    const Refs& r = refs();
    const auto start = core.start(), end = core.end(), pay = core.pay();
    const auto residual = core.residual(), fixed = core.flows();
    const std::size_t n = core.size();
    if (core.kind() == LegKind::TermRate) {
        if (type != r.term_rate_history && type != r.overnight_rate_history) return false;
        const nb::object rates = history.attr("rates");
        if (!PyDict_CheckExact(rates.ptr())) return false;
        const auto fixing = core.fixing();
        for (std::size_t i = 0; i < n; ++i) {
            if (!(pay[i] > t && fixing[i] <= t)) continue;
            // RateHistory.get_rate(fixing).get_accrued_interest(residual, start, end)
            const nb::object day = date_of(fixing[i]);
            PyObject* rate = PyDict_GetItemWithError(rates.ptr(), day.ptr());
            if (rate == nullptr) {
                if (PyErr_Occurred()) throw nb::python_error();
                throw_py(PyExc_KeyError, fmt_str("Date {} not found in {} rate history.", day,
                                                 history.attr("name")));
            }
            const auto terms = dt::rate_terms(rate);
            if (!terms) return false;
            const double wf = terms->wealth_factor(start[i], end[i]);
            if (!std::isfinite(wf)) return false;  // where Python may raise
            p.index.push_back(static_cast<std::int64_t>(i));
            p.flows.push_back(residual[i] * (wf - 1.0) + fixed[i]);
        }
        return true;
    }
    const char* attr = type == r.overnight_rate_history    ? "_index_values"
                       : type == r.overnight_price_history ? "values"
                                                           : nullptr;
    if (attr == nullptr) return false;
    const nb::object levels = history.attr(attr);
    if (!PyDict_CheckExact(levels.ptr())) return false;
    for (std::size_t i = 0; i < n; ++i) {
        if (!(pay[i] > t && start[i] <= t)) continue;
        const std::int64_t realised_end = std::min(t, end[i]);
        // get_accrued_interest: levels[end] / levels[start] - 1, end looked up first
        const auto last = level_at(levels, realised_end);
        if (!last) return false;
        const auto first = level_at(levels, start[i]);
        if (!first) return false;
        double interest = residual[i] * (*last / *first - 1.0);
        if (realised_end < end[i]) {
            const double accrual =
                nb::cast<double>(proj.attr("get_accrual")(m.t_date(), date_of(end[i])));
            interest += (residual[i] + interest) * accrual;
        }
        p.index.push_back(static_cast<std::int64_t>(i));
        p.flows.push_back(interest + fixed[i]);
    }
    return true;
}

Projection projection(nb::handle leg, const Leg& l, const MarketView& m) {
    Projection p;
    if (l.fixed()) return p;
    nb::object proj = m.index_curve(l.core->index);
    if (!l.core->has_realised(m.t())) {
        p.curve = df_source(proj);
        return p;
    }
    if (realised_flows(*l.core, m, proj, p)) {
        p.curve = df_source(proj);
        return p;
    }
    p.index.clear();
    p.flows.clear();
    const nb::object keep = nb::handle(refs().projection)(leg, l.core_obj, m.market(), m.t());
    const auto index = nb::cast<DaysIn>(nb::object(keep[1]));
    const auto flows = nb::cast<DoublesIn>(nb::object(keep[2]));
    p.curve = df_source(nb::object(keep[0]));
    p.index.assign(index.data(), index.data() + index.shape(0));
    p.flows.assign(flows.data(), flows.data() + flows.shape(0));
    return p;
}

// _native_swap_pv.leg_pv.
double leg_pv(nb::handle leg, nb::handle collateral, const MarketView& m, bool assume_par) {
    const Leg l = leg_of(leg);
    const PySwapLeg& core = *l.core;
    const std::int64_t t = m.t();
    if (!l.fixed() && (assume_par || core.par_eligible()) && core.nothing_realised(t) &&
        m.discounted_on_index_curve(core.currency, core.index, collateral)) {
        return core.par_pv(df_source(m.index_curve(core.index)));
    }
    if (!core.has_unpaid(t)) return 0.0;
    const Projection p = projection(leg, l, m);
    const DiscountCurves discount = m.discount(core.currency, collateral);
    return core.pv(t, p.get(), p.known(), discount.get());
}

// _native_swap_pv.leg_pv_non_deliverable.
double leg_pv_non_deliverable(nb::handle leg, nb::handle collateral, const MarketView& m,
                              nb::handle settlement) {
    const Leg l = leg_of(leg);
    const PySwapLeg& core = *l.core;
    const std::int64_t t = m.t();
    if (!core.has_unpaid(t)) return 0.0;
    const Projection p = projection(leg, l, m);
    const nb::handle pair = currency_pair(core.currency, settlement);
    const nb::object known_until = m.fx_known_until(pair);
    const std::int64_t last_known = ordinal_of(known_until);
    bool any_unknown = false;
    const auto known_ordinals = core.known_fx_fixings(t, last_known, any_unknown);
    std::vector<double> known_fx(known_ordinals.size());
    for (std::size_t i = 0; i < known_ordinals.size(); ++i) {
        known_fx[i] = m.fx_rate(date_of(known_ordinals[i]), pair);
    }
    double anchor = 0.0;
    std::optional<DfSource> base, quote;
    if (any_unknown) {
        anchor = m.fx_rate(known_until, pair);
        base = df_source(m.discount_curve(core.currency));
        quote = df_source(m.discount_curve(settlement));
    }
    const DiscountCurves discount = m.discount(settlement, collateral);
    const FxProjection fx{last_known, known_fx, anchor, base ? &*base : nullptr,
                          quote ? &*quote : nullptr};
    return core.pv_non_deliverable(t, p.get(), p.known(), fx, discount.get());
}

// _native_swap_pv.non_started_leg_pv.
double non_started_leg_pv(nb::handle leg, nb::handle collateral, const MarketView& m) {
    const Leg l = leg_of(leg);
    const PySwapLeg& core = *l.core;
    std::optional<DfSource> proj;
    if (!l.fixed()) {
        if (m.discounted_on_index_curve(core.currency, core.index, collateral)) {
            return core.par_pv(df_source(m.index_curve(core.index)));
        }
        proj = df_source(m.index_curve(core.index));
    }
    const DiscountCurves discount = m.discount(core.currency, collateral);
    return core.non_started_pv(proj ? &*proj : nullptr, discount.get());
}

// ── Swaps (_swap_pv.get_leg_pv / get_swap_mtm, _simplified_swap_pv) ────────

// _swap_pv.get_leg_pv for one leg of a Swap.
double swap_leg_pv(nb::handle swap, nb::handle leg, nb::handle collateral, const MarketView& m,
                   nb::handle currency) {
    const Leg l = leg_of(leg);
    // Swap.leg_settlement_currency(leg)
    nb::object settlement = nb::none();
    if (!nb::cast<bool>(swap.attr("is_deliverable"))) settlement = swap.attr("payment_currency");
    if (settlement.is_none()) settlement = settlement_of(l.core->currency);
    // _swap_pv._leg_pv_non_deliverable
    const double pv = same_currency(l.core->currency, settlement)
                          ? leg_pv(leg, collateral, m, false)
                          : leg_pv_non_deliverable(leg, collateral, m, settlement);
    return m.convert(pv, settlement, currency);
}

double swap_mtm(nb::handle swap, const MarketView& m, nb::handle currency) {
    const nb::object collateral = swap.attr("collateral_index");
    const double receive = swap_leg_pv(swap, swap.attr("receive_leg"), collateral, m, currency);
    const double pay = swap_leg_pv(swap, swap.attr("pay_leg"), collateral, m, currency);
    return receive - pay;
}

double swap_leg_pv_of_side(nb::handle swap, bool receive, nb::handle market,
                           nb::handle currency) {
    const MarketView m(market);
    return swap_leg_pv(swap, swap.attr(receive ? "receive_leg" : "pay_leg"),
                       swap.attr("collateral_index"), m, currency);
}

double simplified_swap_mtm(nb::handle swap, const MarketView& m, nb::handle currency) {
    const nb::object collateral = swap.attr("collateral_index");
    const nb::object receive = swap.attr("receive_leg"), pay = swap.attr("pay_leg");
    const double r = m.convert(leg_pv(receive, collateral, m, true),
                               leg_of(receive).core->currency, currency);
    const double p =
        m.convert(leg_pv(pay, collateral, m, true), leg_of(pay).core->currency, currency);
    return r - p;
}

double non_started_swap_mtm(nb::handle swap, const MarketView& m, nb::handle currency) {
    const nb::object effective = swap.attr("effective_date");
    if (m.t() > ordinal_of(effective)) {
        throw_py(PyExc_ValueError,
              fmt_str("A NonStartedSimplifiedSwap effective {} cannot be valued on {}, after it "
                     "starts: use a SimplifiedSwap or a Swap.",
                     effective, m.t_date()));
    }
    const nb::object collateral = swap.attr("collateral_index");
    const nb::object receive = swap.attr("receive_leg"), pay = swap.attr("pay_leg");
    const double r = m.convert(non_started_leg_pv(receive, collateral, m),
                               leg_of(receive).core->currency, currency);
    const double p = m.convert(non_started_leg_pv(pay, collateral, m),
                               leg_of(pay).core->currency, currency);
    return r - p;
}

// _swap_risk.get_leg_duration of the receive (True) or pay leg.
double leg_duration(nb::handle swap, bool receive, nb::handle market) {
    const MarketView m(market);
    const nb::object leg = swap.attr(receive ? "receive_leg" : "pay_leg");
    const Leg l = leg_of(leg);
    const PySwapLeg& core = *l.core;
    const std::int64_t t = m.t();
    if (!core.has_unpaid(t)) return 0.0;
    const Projection p = projection(leg, l, m);
    const std::vector<double> unpaid = core.unpaid_flows(t, p.get(), p.known());
    std::vector<std::int64_t> dates;
    std::vector<double> flows;
    core.rate_sensitive(t, unpaid, dates, flows);
    if (dates.empty()) return 0.0;
    const DiscountCurves discount = m.discount(core.currency, swap.attr("collateral_index"));
    std::vector<double> dfs(dates.size());
    discount.get()(dates, dfs);
    return swaps::duration(t, dates, flows, dfs);
}

// ── Rate accruals (Market.accrue_rates_*) ──────────────────────────────────

// One step of the accrual loops: accrued += rate.get_accrued_interest(notional +
// accrued, start, end). False for a rate left to Python, or a result that is not
// finite (where Python may raise).
bool accrue(nb::handle rate, double notional, std::int64_t start, std::int64_t end,
            double& accrued, bool& exponential) {
    const auto terms = dt::rate_terms(rate);
    if (!terms) return false;
    const double interest = (notional + accrued) * (terms->wealth_factor(start, end) - 1.0);
    if (!std::isfinite(interest)) return false;
    accrued += interest;
    exponential = exponential || terms->interest == rates::Interest::Exponential;
    return true;
}

// The loop of Market.accrue_rates_reset_business_days, from start < end: (accrued,
// whether a rate was exponential, so the Python result is an np.float64), or None to
// leave it to Python.
nb::object accrue_business_days(nb::handle market, double notional, nb::handle name,
                                std::int64_t start, std::int64_t end, std::int64_t lag,
                                const dates::CalendarCore& calendar, bool closest) {
    const MarketView m(market);
    double accrued = 0.0;
    bool exponential = false;
    for (std::int64_t t = start; t < end;) {
        const std::int64_t reset = calendar.add_business_days(t, -lag);
        const nb::object rate = m.rate(reset, name, closest);
        const std::int64_t next = calendar.add_business_days(t, 1);
        if (!accrue(rate, notional, t, std::min(next, end), accrued, exponential)) {
            return nb::none();
        }
        t = calendar.add_business_days(t, 1);
    }
    return nb::make_tuple(accrued, exponential);
}

// The loop of Market.accrue_rates_custom_reset_days, on the sorted reset dates.
nb::object accrue_custom_days(nb::handle market, double notional, nb::handle name,
                              std::int64_t start, std::int64_t end, nb::handle reset_dates,
                              bool closest) {
    const auto resets = dt::exact_ordinals(reset_dates);
    if (!resets || !PyList_CheckExact(reset_dates.ptr())) return nb::none();
    const MarketView m(market);
    double accrued = 0.0;
    bool exponential = false;
    std::size_t k = 0;
    for (std::int64_t t = start; t < end; ++k) {
        const nb::object rate = m.rate((*resets)[k], name, closest);
        std::int64_t next = k == resets->size() - 1 ? end : (*resets)[k + 1];
        if (next > end) next = end;
        if (!accrue(rate, notional, t, next, accrued, exponential)) return nb::none();
        t = next;
    }
    return nb::make_tuple(accrued, exponential);
}

// ── Forwards (_native_fx.forward_mtm / ndf_mtm, with the lookups here) ─────

double forward_mtm(nb::handle forward, const MarketView& m, nb::handle currency) {
    const nb::object pair = forward.attr("currency_pair");
    const std::int64_t pay = ordinal_of(forward.attr("payment_date"));
    if (pay <= m.t()) return 0.0;
    const nb::object collateral = forward.attr("collateral_index");
    const double spot = m.fx_rate(m.t_date(), pair);
    const nb::object quote_ccy = pair.attr("quote_currency");
    const DiscountCurves quote = m.discount(quote_ccy, collateral);
    const DiscountCurves base = m.discount(pair.attr("base_currency"), collateral);
    const double to_currency = m.fx_rate(m.t_date(), currency_pair(quote_ccy, currency));
    return fx::forward_mtm(nb::cast<int>(forward.attr("sign")),
                           nb::cast<double>(forward.attr("notional")),
                           nb::cast<double>(forward.attr("strike")), spot, pay, quote.get(),
                           base.get(), to_currency);
}

// _native_fx._settlement_fx.
double settlement_fx(nb::handle ndf, const MarketView& m, nb::handle pair) {
    const nb::object fixing = ndf.attr("fixing_date");
    const std::int64_t fixing_ordinal = ordinal_of(fixing);
    if (fixing_ordinal <= m.t()) return m.fx_rate(fixing, pair);
    const nb::object known_until = m.fx_known_until(pair);
    const std::int64_t last_known = ordinal_of(known_until);
    const nb::object base = pair.attr("base_currency"), quote = pair.attr("quote_currency");
    // Market.get_future_fx_rate: published, or 1 on a pair of one currency.
    if (fixing_ordinal <= last_known) return m.fx_rate(fixing, pair);
    if (same_currency(base, quote)) return 1.0;
    const double anchor = m.fx_rate(known_until, pair);
    const DfSource base_curve = df_source(m.discount_curve(base));
    const DfSource quote_curve = df_source(m.discount_curve(quote));
    const double rate =
        fx::projected_fx(anchor, last_known, fixing_ordinal, base_curve, quote_curve);
    if (!(rate > 0)) {  // FX_Rate(cp, rate) rejects it with its own message
        return nb::cast<double>(nb::handle(refs().fx_rate)(pair, rate).attr("value"));
    }
    return rate;
}

double ndf_mtm(nb::handle ndf, const MarketView& m, nb::handle currency) {
    const nb::object pair = ndf.attr("currency_pair");
    const std::int64_t pay = ordinal_of(ndf.attr("payment_date"));
    if (pay <= m.t()) return 0.0;
    const double s_t = settlement_fx(ndf, m, pair);
    const nb::object quote_ccy = pair.attr("quote_currency");
    const DiscountCurves quote = m.discount(quote_ccy, ndf.attr("collateral_index"));
    const double to_currency = m.fx_rate(m.t_date(), currency_pair(quote_ccy, currency));
    return fx::ndf_mtm(nb::cast<int>(ndf.attr("sign")), nb::cast<double>(ndf.attr("notional")),
                       nb::cast<double>(ndf.attr("strike")), s_t, pay, quote.get(),
                       to_currency);
}

// ── Batches (_batch.ValuationBatch) ────────────────────────────────────────

// What Calculator.valuate runs for each instrument kind, in one market call.
enum class Kind : std::int64_t { Swap = 1, Simplified = 2, NonStarted = 3, Forward = 4, Ndf = 5 };

using Valuation = double (*)(nb::handle, const MarketView&, nb::handle);

template <Valuation F>
double on_new_view(nb::handle instrument, nb::handle market, nb::handle currency) {
    return F(instrument, MarketView(market), currency);
}

double valuate_kind(nb::handle instrument, std::int64_t kind, const MarketView& m,
                    nb::handle currency) {
    switch (static_cast<Kind>(kind)) {
        case Kind::Swap: return swap_mtm(instrument, m, currency);
        case Kind::Simplified: return simplified_swap_mtm(instrument, m, currency);
        case Kind::NonStarted: return non_started_swap_mtm(instrument, m, currency);
        case Kind::Forward: return forward_mtm(instrument, m, currency);
        case Kind::Ndf: return ndf_mtm(instrument, m, currency);
    }
    throw_py(PyExc_ValueError, fmt_str("Unknown valuation kind {}.", kind));
}

// The MTMs of `instruments` (a list), each valued as its `kinds` entry says, in order, on
// one view of `market`: the values, and errors, of one call per instrument.
DoublesOut valuate_many(nb::handle instruments, DaysIn kinds, nb::handle market,
                        nb::handle currency) {
    if (!PyList_CheckExact(instruments.ptr())) {
        throw_py(PyExc_TypeError, nb::str("instruments must be a list."));
    }
    const auto n = static_cast<std::size_t>(PyList_GET_SIZE(instruments.ptr()));
    if (kinds.shape(0) != n) throw_py(PyExc_ValueError, nb::str("One kind per instrument."));
    const MarketView m(market);
    auto [out, mtm] = new_array<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
        const nb::handle instrument =
            PyList_GET_ITEM(instruments.ptr(), static_cast<Py_ssize_t>(i));
        mtm[i] = valuate_kind(instrument, kinds.data()[i], m, currency);
    }
    return out;
}

}  // namespace

// ── For the curve builder's solver (py_market.hpp) ─────────────────────────

void valuate_subset(nb::handle instruments, std::span<const std::int64_t> kinds,
                    std::span<const std::int64_t> subset, nb::handle market,
                    nb::handle currency, double* out) {
    const MarketView m(market);
    for (std::size_t i = 0; i < subset.size(); ++i) {
        const auto k = static_cast<std::size_t>(subset[i]);
        const nb::handle instrument =
            PyList_GET_ITEM(instruments.ptr(), static_cast<Py_ssize_t>(k));
        out[i] = valuate_kind(instrument, kinds[k], m, currency);
    }
}

namespace {

// ValuationBatch._compile / _compile_leg / _compile_ndf / _freeze, and
// _Jacobian._source_terms. A value is referred to by (source, index in the source) until
// every source is laid out.
class BatchCompiler {
public:
    BatchCompiler(const MarketView& m, nb::handle currency, CompiledBatch& out)
        : m_(m), currency_(currency), out_(out) {}

    bool instrument(nb::handle instrument, std::int64_t kind, std::int64_t row) {
        if (static_cast<Kind>(kind) == Kind::Ndf) return ndf(instrument, row);
        if (static_cast<Kind>(kind) != Kind::Swap) return false;
        const nb::object collateral = instrument.attr("collateral_index");
        // Swap.leg_settlement_currency(leg), as swap_leg_pv reads it.
        nb::object payment = nb::none();
        if (!nb::cast<bool>(instrument.attr("is_deliverable"))) {
            payment = instrument.attr("payment_currency");
        }
        // Compiled whole or not at all: a failed leg leaves its swap's entries behind,
        // and the caller drops the batch.
        return leg(instrument.attr("receive_leg"), collateral, payment, row, 1.0) &&
               leg(instrument.attr("pay_leg"), collateral, payment, row, -1.0);
    }

    // Lay the sources end to end, resolve the positions, add the curve terms.
    void finish() {
        std::vector<std::size_t> offset(out_.sources.size());
        std::size_t n = 0;
        for (std::size_t i = 0; i < out_.sources.size(); ++i) {
            offset[i] = n;
            n += out_.sources[i].o0.size();
        }
        out_.n_values = n;
        auto at = [&](const Pos& p) {
            return static_cast<std::int64_t>(p.source == kOne ? n : offset[p.source] + p.index);
        };
        for (const Pos& p : fx_pos_) out_.fx_at.push_back(at(p));
        for (const Pos& p : df_pos_) out_.df_at.push_back(at(p));
        for (const Pos& p : accrual_pos_) out_.accrual_at.push_back(at(p));
        for (const Pos& p : head_pos_) out_.head_at.push_back(at(p));
        for (std::size_t i = 0; i < out_.ndfs.size(); ++i) {
            out_.ndfs[i].df = at(ndf_pos_[i].first);
            out_.ndfs[i].fx = ndf_pos_[i].second ? at(*ndf_pos_[i].second) : -1;
        }
        out_.log_shift.assign(n + 1, 0.0);
        for (std::size_t i = 0; i < out_.sources.size(); ++i) {
            const BatchSource& s = out_.sources[i];
            if (s.kind == BatchSource::Accrual) {
                std::fill_n(out_.log_shift.begin() + static_cast<std::ptrdiff_t>(offset[i]),
                            s.o0.size(), 1.0);
            }
            curve_terms(s, static_cast<std::int64_t>(offset[i]));
        }
    }

private:
    static constexpr std::size_t kOne = static_cast<std::size_t>(-1);  // the trailing 1
    struct Pos {
        std::size_t source, index;
    };

    std::size_t source(int kind, nb::handle a, nb::handle b) {
        for (std::size_t i = 0; i < out_.sources.size(); ++i) {
            const BatchSource& s = out_.sources[i];
            if (s.kind == kind && s.a.is(a) && s.b.is(b)) return i;
        }
        BatchSource s;
        s.kind = kind;
        s.a = nb::borrow(a);
        s.b = nb::borrow(b);
        out_.sources.push_back(std::move(s));
        return out_.sources.size() - 1;
    }

    // The values of `src` at o0 (and o1): the position of the first.
    Pos ref(std::size_t src, std::span<const std::int64_t> o0,
            std::span<const std::int64_t> o1 = {}) {
        BatchSource& s = out_.sources[src];
        const Pos p{src, s.o0.size()};
        s.o0.insert(s.o0.end(), o0.begin(), o0.end());
        s.o1.insert(s.o1.end(), o1.begin(), o1.end());
        return p;
    }

    static Pos shifted(Pos p, std::size_t i) { return p.source == kOne ? p : Pos{p.source, p.index + i}; }

    void add_terms(std::span<const double> flows, Pos fx, Pos df, std::int64_t row, double scale) {
        for (std::size_t i = 0; i < flows.size(); ++i) {
            out_.flows.push_back(flows[i]);
            fx_pos_.push_back(shifted(fx, i));
            df_pos_.push_back(shifted(df, i));
            out_.term_row.push_back(row);
            out_.term_scale.push_back(scale);
        }
    }

    // _compile_leg: false when a coupon is already running or fixed.
    bool leg(nb::handle leg_obj, nb::handle collateral, nb::handle payment, std::int64_t row,
             double sign) {
        const Leg l = leg_of(leg_obj);
        const PySwapLeg& core = *l.core;
        const nb::object settlement =
            payment.is_none() ? settlement_of(core.currency) : nb::borrow(payment);
        const std::int64_t t = m_.t();
        const double scale = sign * m_.fx_rate(m_.t_date(), currency_pair(settlement, currency_));
        const bool deliverable = same_currency(core.currency, settlement);
        if (deliverable && !l.fixed() && core.par_eligible() && core.nothing_realised(t) &&
            m_.discounted_on_index_curve(core.currency, core.index, collateral)) {
            // _par_leg_pv: residual_0 * df(start_0), plus the spreads on the end dfs.
            const std::size_t index = source(BatchSource::Index, core.index, nb::none());
            head_pos_.push_back(ref(index, core.start().first(1)));
            out_.head_residuals.push_back(core.residual()[0]);
            out_.head_scale.push_back(scale);
            out_.head_row.push_back(row);
            if (core.has_spread()) {
                add_terms(core.spreads(), Pos{kOne, 0}, ref(index, core.end()), row, scale);
            }
            return true;
        }
        const std::size_t k = core.first_unpaid(t);
        if (k == core.size()) return true;
        if (core.has_realised(t)) return false;  // _Uncompiled
        const std::size_t first = out_.flows.size();
        const std::size_t m = core.size() - k;
        const auto flows = core.flows().subspan(k);
        std::vector<double> zeros;
        std::span<const double> static_flows = flows;
        if (!l.fixed()) {  // projected: residual * accrual + fixed flows
            zeros.assign(m, 0.0);
            static_flows = zeros;
            const Pos accrual = ref(source(BatchSource::Accrual, core.index, nb::none()),
                                    core.start().subspan(k), core.end().subspan(k));
            for (std::size_t i = 0; i < m; ++i) {
                out_.projected_pos.push_back(static_cast<std::int64_t>(first + i));
                out_.residuals.push_back(core.residual()[k + i]);
                out_.fixed.push_back(flows[i]);
                accrual_pos_.push_back(shifted(accrual, i));
            }
        }
        const auto pay = core.pay().subspan(k);
        if (deliverable) {
            const Pos df = ref(source(BatchSource::Discount, core.currency, collateral), pay);
            add_terms(static_flows, Pos{kOne, 0}, df, row, scale);
        } else {
            const nb::handle pair = currency_pair(core.currency, settlement);
            const Pos fx = ref(source(BatchSource::Fx, pair, nb::none()),
                               core.fx_fixing().subspan(k));
            const Pos df = ref(source(BatchSource::Discount, settlement, collateral), pay);
            add_terms(static_flows, fx, df, row, scale);
        }
        return true;
    }

    // _compile_ndf.
    bool ndf(nb::handle ndf_obj, std::int64_t row) {
        const nb::object cp = ndf_obj.attr("currency_pair");
        const std::int64_t pay = ordinal_of(ndf_obj.attr("payment_date"));
        const std::int64_t t = m_.t();
        if (pay <= t) return true;  // paid: nothing
        const nb::object fixing = ndf_obj.attr("fixing_date");
        const std::int64_t fixing_ordinal = ordinal_of(fixing);
        const nb::object quote = cp.attr("quote_currency");
        CompiledBatch::Ndf entry{row,
                                 nb::cast<double>(ndf_obj.attr("sign")) *
                                     nb::cast<double>(ndf_obj.attr("notional")),
                                 nb::cast<double>(ndf_obj.attr("strike")),
                                 std::numeric_limits<double>::quiet_NaN(),
                                 -1,
                                 -1,
                                 m_.fx_rate(m_.t_date(), currency_pair(quote, currency_))};
        std::optional<Pos> fx;
        if (fixing_ordinal <= t || fixing_ordinal <= ordinal_of(m_.fx_known_until(cp))) {
            entry.s_t = m_.fx_rate(fixing, cp);
        } else {
            const nb::handle pair = currency_pair(cp.attr("base_currency"), quote);
            const std::int64_t at[] = {fixing_ordinal};
            fx = ref(source(BatchSource::Fx, pair, nb::none()), at);
        }
        const std::int64_t at[] = {pay};
        const Pos df = ref(
            source(BatchSource::Discount, quote, ndf_obj.attr("collateral_index")), at);
        out_.ndfs.push_back(entry);
        ndf_pos_.emplace_back(df, fx);
        return true;
    }

    void term(nb::handle curve, std::int64_t row, std::int64_t ordinal, double sign) {
        if (std::none_of(out_.curves.begin(), out_.curves.end(),
                         [&](const nb::object& c) { return c.is(curve); })) {
            out_.curves.push_back(nb::borrow(curve));
        }
        out_.curve_terms.push_back({curve.ptr(), row, ordinal, sign});
    }

    // _Jacobian._source_terms of one source: its values as products of curve dfs.
    void curve_terms(const BatchSource& s, std::int64_t offset) {
        const std::size_t n = s.o0.size();
        std::vector<std::pair<nb::object, double>> curves;
        switch (s.kind) {
            case BatchSource::Discount:  // Market._dfs_at_ordinals
                if (s.b.is_none()) {
                    curves.emplace_back(m_.discount_curve(s.a), 1.0);
                } else if (same_currency(s.b.attr("currency"), s.a)) {
                    curves.emplace_back(m_.index_curve(s.b), 1.0);
                } else {
                    curves.emplace_back(m_.discount_curve(s.a), 1.0);
                    curves.emplace_back(m_.index_curve(s.b), 1.0);
                    curves.emplace_back(m_.discount_curve(s.b.attr("currency")), -1.0);
                }
                for (const auto& [curve, sign] : curves) {
                    for (std::size_t i = 0; i < n; ++i) term(curve, offset + static_cast<std::int64_t>(i), s.o0[i], sign);
                }
                return;
            case BatchSource::Index: {
                const nb::object curve = m_.index_curve(s.a);
                for (std::size_t i = 0; i < n; ++i) term(curve, offset + static_cast<std::int64_t>(i), s.o0[i], 1.0);
                return;
            }
            case BatchSource::Accrual: {  // 1 + accrual = df(start) / df(end)
                const nb::object curve = m_.index_curve(s.a);
                for (std::size_t i = 0; i < n; ++i) {
                    term(curve, offset + static_cast<std::int64_t>(i), s.o0[i], 1.0);
                    term(curve, offset + static_cast<std::int64_t>(i), s.o1[i], -1.0);
                }
                return;
            }
            case BatchSource::Fx: {  // anchor * df ratios after the last known fixing
                const nb::object base = s.a.attr("base_currency");
                const nb::object quote = s.a.attr("quote_currency");
                if (same_currency(base, quote)) return;
                const std::int64_t last = ordinal_of(m_.fx_known_until(s.a));
                const nb::object base_curve = m_.discount_curve(base);
                const nb::object quote_curve = m_.discount_curve(quote);
                for (std::size_t i = 0; i < n; ++i) {
                    if (s.o0[i] <= last) continue;
                    const std::int64_t row = offset + static_cast<std::int64_t>(i);
                    term(base_curve, row, s.o0[i], 1.0);
                    term(base_curve, row, last, -1.0);
                    term(quote_curve, row, s.o0[i], -1.0);
                    term(quote_curve, row, last, 1.0);
                }
                return;
            }
        }
    }

    const MarketView& m_;
    nb::handle currency_;
    CompiledBatch& out_;
    std::vector<Pos> fx_pos_, df_pos_, accrual_pos_, head_pos_;
    std::vector<std::pair<Pos, std::optional<Pos>>> ndf_pos_;
};

}  // namespace

bool compile_batch(nb::handle instruments, std::span<const std::int64_t> kinds,
                   nb::handle market, nb::handle currency, CompiledBatch& out) {
    out = CompiledBatch{};
    const MarketView m(market);
    BatchCompiler compiler(m, currency, out);
    for (std::size_t k = 0; k < kinds.size(); ++k) {
        const nb::handle instrument =
            PyList_GET_ITEM(instruments.ptr(), static_cast<Py_ssize_t>(k));
        if (!compiler.instrument(instrument, kinds[k], static_cast<std::int64_t>(k))) {
            return false;
        }
    }
    compiler.finish();
    return true;
}

void evaluate_sources(nb::handle market, std::span<const BatchSource> sources, double* out) {
    const MarketView m(market);
    std::vector<double> starts;
    for (const BatchSource& s : sources) {
        const std::size_t n = s.o0.size();
        const std::span<double> into(out, n);
        switch (s.kind) {
            case BatchSource::Discount:  // Market._dfs_at_ordinals
                m.discount(s.a, s.b).get()(s.o0, into);
                break;
            case BatchSource::Index:  // the index curve's _dfs_at_ordinals
                df_source(m.index_curve(s.a))(s.o0, into);
                break;
            case BatchSource::Accrual: {  // IndexCurve._accrual_at_ordinals
                const swaps::DfSource curve = df_source(m.index_curve(s.a));
                starts.resize(n);
                curve(s.o0, starts);
                curve(s.o1, into);
                for (std::size_t i = 0; i < n; ++i) out[i] = 1.0 / (out[i] / starts[i]) - 1;
                break;
            }
            case BatchSource::Fx: {  // Market._future_fx_rates_at_ordinals
                const DoublesOut fx = m.future_fx(s.a, s.o0);
                std::copy(fx.data(), fx.data() + n, out);
                break;
            }
        }
        out += n;
    }
}

namespace {

// ── Pillar dates (curve_builder._keys._maturities) ─────────────────────────

// Kinds of maturity_tables: what _keys._maturities tells apart by isinstance.
constexpr std::int64_t kTableForward = 1, kTableNdf = 2, kTableSwap = 4,
                       kTableReceiveFloats = 8, kTablePayFloats = 16;

// Largest value of a leg's ordinal array (the max() of its dates); false when empty.
bool last_ordinal(nb::handle leg, const char* name, std::int64_t& out) {
    const auto ordinals = nb::cast<DaysIn>(leg.attr(name));
    if (ordinals.shape(0) == 0) return false;
    out = *std::max_element(ordinals.data(), ordinals.data() + ordinals.shape(0));
    return true;
}

// _keys._funding_key.
nb::handle funding_key(nb::handle rfi, nb::handle rfi_currency, nb::handle currency) {
    return same_currency(currency, rfi_currency) ? rfi : currency;
}

// One instrument's ({curve key: pillar date}, last date), or None for Python to do it.
nb::object maturity_table(nb::handle instrument, std::int64_t kind, nb::handle rfi,
                          nb::handle rfi_currency) {
    nb::dict ordinals;  // curve key -> latest pillar ordinal (keys compare by name)
    auto add = [&](nb::handle key, std::int64_t ordinal) {
        if (ordinals.contains(key)) {
            ordinal = std::max(ordinal, nb::cast<std::int64_t>(ordinals[key]));
        }
        ordinals[key] = ordinal;
    };
    auto finish = [&](std::int64_t last) {
        nb::dict out;
        for (auto [key, ordinal] : ordinals) out[key] = date_of(nb::cast<std::int64_t>(ordinal));
        return nb::make_tuple(out, date_of(last));
    };
    if (kind == kTableForward || kind == kTableNdf) {
        const nb::object pay_date = instrument.attr("payment_date");
        if (pay_date.is_none()) return nb::none();
        const nb::object pair = instrument.attr("currency_pair");
        const std::int64_t pay = ordinal_of(pay_date);
        std::int64_t pinned = pay, last = pay;
        if (kind == kTableNdf) {
            const nb::object fixing = instrument.attr("fixing_date");
            if (fixing.is_none()) return nb::none();
            pinned = ordinal_of(fixing);
            last = std::max(pay, pinned);
        }
        add(funding_key(rfi, rfi_currency, pair.attr("base_currency")), pinned);
        add(funding_key(rfi, rfi_currency, pair.attr("quote_currency")), pinned);
        return finish(last);
    }
    if ((kind & kTableSwap) == 0) return nb::none();
    std::int64_t last = std::numeric_limits<std::int64_t>::min();
    const nb::object settlement_of_leg = instrument.attr("leg_settlement_currency");
    for (int side = 0; side < 2; ++side) {
        const nb::object leg = instrument.attr(side == 0 ? "receive_leg" : "pay_leg");
        const bool floats = (kind & (side == 0 ? kTableReceiveFloats : kTablePayFloats)) != 0;
        const nb::object pay_ccy = settlement_of_leg(leg);
        const nb::object leg_ccy = leg.attr("currency");
        std::int64_t pay = 0, end = 0, fx = 0;
        if (!last_ordinal(leg, "_payment_ordinals", pay)) return nb::none();  // max() raises
        add(funding_key(rfi, rfi_currency, pay_ccy), pay);
        if (!same_currency(pay_ccy, leg_ccy)) {
            if (!last_ordinal(leg, "_fx_fixing_ordinals", fx)) return nb::none();
            add(funding_key(rfi, rfi_currency, leg_ccy), fx);
        }
        last = std::max(last, pay);
        if (floats) {
            if (!last_ordinal(leg, "_end_ordinals", end)) return nb::none();
            add(leg.attr("index"), end);
            last = std::max(last, end);
        }
    }
    return finish(last);
}

nb::list maturity_tables(nb::list instruments, DaysIn kinds, nb::handle rfi) {
    if (kinds.shape(0) != nb::len(instruments)) {
        throw_py(PyExc_ValueError, nb::str("One kind per instrument."));
    }
    const nb::object rfi_currency = rfi.attr("currency");
    nb::list out;
    for (std::size_t i = 0; i < kinds.shape(0); ++i) {
        out.append(maturity_table(instruments[i], kinds.data()[i], rfi, rfi_currency));
    }
    return out;
}

}  // namespace

void bind_market(nb::module_& m) {
    PyDateTime_IMPORT;
    if (PyDateTimeAPI == nullptr) throw nb::python_error();

    m.def("maturity_tables", &maturity_tables, nb::arg("instruments"), nb::arg("kinds"),
          nb::arg("risk_free_index"),
          "curve_builder._keys._maturities of each instrument: ({curve key: pillar "
          "date}, last date), or None where Python must do it. Kinds: 0 other, 1 "
          "forward, 2 NDF, 4 swap, plus 8 / 16 when its receive / pay leg floats.");

    m.def("market_swap_mtm", &on_new_view<swap_mtm>, nb::arg("swap"), nb::arg("market"), nb::arg("currency"),
          "_swap_pv.get_swap_mtm with the market lookups in C++.");
    m.def("market_swap_leg_pv", &swap_leg_pv_of_side, nb::arg("swap"), nb::arg("receive"),
          nb::arg("market"), nb::arg("currency"),
          "_swap_pv.get_leg_pv of the receive (True) or pay leg.");
    m.def("market_simplified_swap_mtm", &on_new_view<simplified_swap_mtm>, nb::arg("swap"),
          nb::arg("market"), nb::arg("currency"));
    m.def("market_non_started_swap_mtm", &on_new_view<non_started_swap_mtm>, nb::arg("swap"),
          nb::arg("market"), nb::arg("currency"));
    m.def("market_forward_mtm", &on_new_view<forward_mtm>, nb::arg("forward"), nb::arg("market"),
          nb::arg("currency"));
    m.def("market_ndf_mtm", &on_new_view<ndf_mtm>, nb::arg("ndf"), nb::arg("market"),
          nb::arg("currency"));
    m.def("market_valuate_many", &valuate_many, nb::arg("instruments"), nb::arg("kinds"),
          nb::arg("market"), nb::arg("currency"),
          "The MTMs of a list of instruments on one market view: kinds 1 swap, 2 simplified "
          "swap, 3 non-started simplified swap, 4 forward, 5 NDF (the market_* calls).");

    m.def("market_leg_duration", &leg_duration, nb::arg("swap"), nb::arg("receive"),
          nb::arg("market"), "_swap_risk.get_leg_duration of the receive (True) or pay leg.");
    m.def(
        "market_dfs",
        [](nb::handle market, nb::handle currency, DaysIn ordinals, nb::handle collateral,
           nb::handle risk_free_index) {
            return MarketView(market, risk_free_index).dfs(currency, view(ordinals), collateral);
        },
        nb::arg("market"), nb::arg("currency"), nb::arg("ordinals"),
        nb::arg("collateral").none(), nb::arg("risk_free_index").none(),
        "Market._dfs_at_ordinals.");
    m.def(
        "market_future_fx",
        [](nb::handle market, nb::handle pair, DaysIn ordinals) {
            return MarketView(market).future_fx(pair, view(ordinals));
        },
        nb::arg("market"), nb::arg("pair"), nb::arg("ordinals"),
        "Market._future_fx_rates_at_ordinals.");

    m.def("market_accrue_business_days", &accrue_business_days, nb::arg("market"),
          nb::arg("notional"), nb::arg("name"), nb::arg("start"), nb::arg("end"),
          nb::arg("fixing_lag"), nb::arg("calendar"), nb::arg("closest"),
          "The loop of Market.accrue_rates_reset_business_days: (accrued, exponential) "
          "or None.");
    m.def("market_accrue_custom_days", &accrue_custom_days, nb::arg("market"),
          nb::arg("notional"), nb::arg("name"), nb::arg("start"), nb::arg("end"),
          nb::arg("reset_dates"), nb::arg("closest"),
          "The loop of Market.accrue_rates_custom_reset_days: (accrued, exponential) "
          "or None.");
    m.def(
        "market_rate",
        [](nb::handle market, std::int64_t t, nb::handle name, bool closest) {
            return MarketView(market).rate(t, name, closest);
        },
        nb::arg("market"), nb::arg("t"), nb::arg("name"), nb::arg("closest"),
        "Market.get_rate at the ordinal t.");

    // The lookups alone, for tests.
    m.def(
        "market_index_curve",
        [](nb::handle market, nb::handle index) { return MarketView(market).index_curve(index); },
        nb::arg("market"), nb::arg("index"));
    m.def(
        "market_discount_curve",
        [](nb::handle market, nb::handle currency) {
            return MarketView(market).discount_curve(currency);
        },
        nb::arg("market"), nb::arg("currency"));
    m.def(
        "market_fx_rate",
        [](nb::handle market, nb::handle t, nb::handle pair) {
            return MarketView(market).fx_rate(t, pair);
        },
        nb::arg("market"), nb::arg("t"), nb::arg("pair"));
    m.def(
        "market_fx_known_until",
        [](nb::handle market, nb::handle pair) {
            return MarketView(market).fx_known_until(pair);
        },
        nb::arg("market"), nb::arg("pair"));
    m.def(
        "market_convert",
        [](nb::handle market, double value, nb::handle from, nb::handle to) {
            return MarketView(market).convert(value, from, to);
        },
        nb::arg("market"), nb::arg("value"), nb::arg("from_currency"),
        nb::arg("to_currency"));
}

}  // namespace fts::py
