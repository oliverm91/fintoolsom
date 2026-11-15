// The market lookups of the valuation, in C++ (fintoolsom._native.market_*).
//
// A MarketView reads a Python fintoolsom.market.Market where it stands: its curves,
// FX histories, valuation date and risk-free index are the Python containers
// themselves, looked up through the C API, so the view can never go stale (callers
// mutate those dicts directly). Each lookup is the Python method's, in the same order,
// with the same errors and messages: Market.get_index_curve, get_discount_curve,
// _dfs_at_ordinals (its curves), get_fx_rate, get_fx_known_until and
// convert_value_currency. On top of it, a whole swap, simplified swap, forward or NDF is
// valued in one call, as the calculator functions do.
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fx/fx_derivatives.hpp"
#include "py_curve.hpp"

#include <datetime.h>  // after Python.h

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
};

const Refs& refs() {
    static const Refs r = [] {
        auto get = [](const char* module, const char* name) {
            return nb::object(nb::module_::import_(module).attr(name)).release().ptr();
        };
        const char* currencies = "fintoolsom.market.currencies";
        const char* glue = "fintoolsom.derivatives.calculator._native_swap_pv";
        return Refs{get("fintoolsom.rates", "IndexCurve"), get(currencies, "Currency"),
                    get(currencies, "CurrencyPair"),    get(currencies, "FX_Rate"),
                    get(glue, "_core"),                 get(glue, "_projection")};
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
    explicit MarketView(nb::handle market)
        : market_(market),
          curves_(market.attr("curves")),
          fx_history_(market.attr("fx_history")),
          t_date_(market.attr("t")),
          rfi_(market.attr("risk_free_index")),
          t_(ordinal_of(t_date_)) {}

    [[nodiscard]] nb::handle market() const { return market_; }
    [[nodiscard]] std::int64_t t() const { return t_; }
    [[nodiscard]] nb::handle t_date() const { return t_date_; }

    // Market.get_index_curve.
    [[nodiscard]] nb::object index_curve(nb::handle index) const {
        const nb::object name = index.attr("name");
        PyObject* curve = PyDict_GetItemWithError(curves_.ptr(), name.ptr());
        if (curve == nullptr && PyErr_Occurred()) throw nb::python_error();
        if (curve == nullptr || !PyObject_IsInstance(curve, refs().index_curve_type)) {
            throw_py(PyExc_KeyError, fmt_str("No index curve for '{}' in market.", name));
        }
        return nb::borrow(curve);
    }

    // Market.get_discount_curve, under the market's risk-free index.
    [[nodiscard]] nb::object discount_curve(nb::handle currency) const {
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
    [[nodiscard]] double convert(double value, nb::handle from, nb::handle to) const {
        return value * fx_rate(t_date_, currency_pair(from, to));
    }

private:
    nb::handle market_;
    nb::object curves_, fx_history_, t_date_, rfi_;
    std::int64_t t_;
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

// _native_swap_pv._projection: the index curve, then (in Python) the realised flows
// of the coupons started or fixed by t.
struct Projection {
    std::optional<DfSource> curve;
    nb::object keep;  // the Python (proj, index, flows) when there are realised flows
    Known known;
    [[nodiscard]] const DfSource* get() const { return curve ? &*curve : nullptr; }
};

Projection projection(nb::handle leg, const Leg& l, const MarketView& m) {
    Projection p;
    if (l.fixed()) return p;
    nb::object proj = m.index_curve(l.core->index);
    if (!l.core->has_realised(m.t())) {
        p.curve = df_source(proj);
        return p;
    }
    p.keep = nb::handle(refs().projection)(leg, l.core_obj, m.market(), m.t());
    const auto index = nb::cast<DaysIn>(nb::object(p.keep[1]));
    const auto flows = nb::cast<DoublesIn>(nb::object(p.keep[2]));
    p.curve = df_source(nb::object(p.keep[0]));
    p.known = Known{view(index), view(flows)};
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
    return core.pv(t, p.get(), p.known, discount.get());
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
    return core.pv_non_deliverable(t, p.get(), p.known, fx, discount.get());
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

double swap_mtm(nb::handle swap, nb::handle market, nb::handle currency) {
    const MarketView m(market);
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

double simplified_swap_mtm(nb::handle swap, nb::handle market, nb::handle currency) {
    const MarketView m(market);
    const nb::object collateral = swap.attr("collateral_index");
    const nb::object receive = swap.attr("receive_leg"), pay = swap.attr("pay_leg");
    const double r = m.convert(leg_pv(receive, collateral, m, true),
                               leg_of(receive).core->currency, currency);
    const double p =
        m.convert(leg_pv(pay, collateral, m, true), leg_of(pay).core->currency, currency);
    return r - p;
}

double non_started_swap_mtm(nb::handle swap, nb::handle market, nb::handle currency) {
    const MarketView m(market);
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

// ── Forwards (_native_fx.forward_mtm / ndf_mtm, with the lookups here) ─────

double forward_mtm(nb::handle forward, nb::handle market, nb::handle currency) {
    const MarketView m(market);
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

double ndf_mtm(nb::handle ndf, nb::handle market, nb::handle currency) {
    const MarketView m(market);
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

}  // namespace

void bind_market(nb::module_& m) {
    PyDateTime_IMPORT;
    if (PyDateTimeAPI == nullptr) throw nb::python_error();

    m.def("market_swap_mtm", &swap_mtm, nb::arg("swap"), nb::arg("market"), nb::arg("currency"),
          "_swap_pv.get_swap_mtm with the market lookups in C++.");
    m.def("market_swap_leg_pv", &swap_leg_pv_of_side, nb::arg("swap"), nb::arg("receive"),
          nb::arg("market"), nb::arg("currency"),
          "_swap_pv.get_leg_pv of the receive (True) or pay leg.");
    m.def("market_simplified_swap_mtm", &simplified_swap_mtm, nb::arg("swap"),
          nb::arg("market"), nb::arg("currency"));
    m.def("market_non_started_swap_mtm", &non_started_swap_mtm, nb::arg("swap"),
          nb::arg("market"), nb::arg("currency"));
    m.def("market_forward_mtm", &forward_mtm, nb::arg("forward"), nb::arg("market"),
          nb::arg("currency"));
    m.def("market_ndf_mtm", &ndf_mtm, nb::arg("ndf"), nb::arg("market"), nb::arg("currency"));

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
