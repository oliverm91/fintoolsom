// Bindings of the FX forward, NDF and option kernels, and of ndtr / ndtri.
#include <cstdint>

#include <nanobind/stl/pair.h>

#include "fx/fx_derivatives.hpp"
#include "py_curve.hpp"

namespace fts::py {

void bind_fx(nb::module_& m) {
    m.def("ndtr", &special::ndtr, nb::arg("x"), "Standard normal cdf, 0.5 * erfc(-x / sqrt(2)) (as _normal.ndtr).");
    m.def("ndtri", &special::ndtri, nb::arg("y"), "Inverse of ndtr, the Cephes port of _normal.ndtri.");

    m.def(
        "projected_fx",
        [](double anchor, std::int64_t last_known, std::int64_t fixing, nb::handle base,
           nb::handle quote) {
            return fx::projected_fx(anchor, last_known, fixing, df_source(base),
                                    df_source(quote));
        },
        nb::arg("anchor"), nb::arg("last_known"), nb::arg("fixing"), nb::arg("base"),
        nb::arg("quote"), "FX of a fixing after last_known, projected on the funding curves.");

    m.def(
        "forward_mtm",
        [](int sign, double notional, double strike, double spot, std::int64_t pay,
           nb::handle quote_a, nb::handle quote_b, nb::handle quote_c, nb::handle base_a,
           nb::handle base_b, nb::handle base_c, double to_currency) {
            const DiscountCurves quote(quote_a, quote_b, quote_c);
            const DiscountCurves base(base_a, base_b, base_c);
            return fx::forward_mtm(sign, notional, strike, spot, pay, quote.get(), base.get(),
                                   to_currency);
        },
        nb::arg("sign"), nb::arg("notional"), nb::arg("strike"), nb::arg("spot"),
        nb::arg("pay"), nb::arg("quote_a"), nb::arg("quote_b").none(),
        nb::arg("quote_c").none(), nb::arg("base_a"), nb::arg("base_b").none(),
        nb::arg("base_c").none(), nb::arg("to_currency"),
        "Deliverable forward MtM: quote and base flows discounted, converted at spot.");

    m.def(
        "ndf_mtm",
        [](int sign, double notional, double strike, double settlement_fx, std::int64_t pay,
           nb::handle a, nb::handle b, nb::handle c, double to_currency) {
            const DiscountCurves quote(a, b, c);
            return fx::ndf_mtm(sign, notional, strike, settlement_fx, pay, quote.get(),
                               to_currency);
        },
        nb::arg("sign"), nb::arg("notional"), nb::arg("strike"), nb::arg("settlement_fx"),
        nb::arg("pay"), nb::arg("a"), nb::arg("b").none(), nb::arg("c").none(),
        nb::arg("to_currency"), "NDF MtM given the settlement FX of its fixing date.");

    m.def("option_log_moneyness", &fx::log_moneyness, nb::arg("strike"), nb::arg("spot"),
          nb::arg("df_r"), nb::arg("df_q"));
    m.def(
        "option_d1",
        [](double spot, double strike, double vol, double yf, double r, double q) {
            const auto [d1, vol_sqrt_yf] = fx::option_d1(spot, strike, vol, yf, r, q);
            return std::make_pair(d1, vol_sqrt_yf);
        },
        nb::arg("spot"), nb::arg("strike"), nb::arg("vol"), nb::arg("yf"), nb::arg("r"),
        nb::arg("q"), "(d1, vol * sqrt(yf)).");
    m.def("option_mtm", &fx::option_mtm, nb::arg("sign"), nb::arg("notional"),
          nb::arg("strike"), nb::arg("spot"), nb::arg("vol"), nb::arg("yf"), nb::arg("r"),
          nb::arg("q"));
    m.def("option_delta", &fx::option_delta, nb::arg("sign"), nb::arg("notional"),
          nb::arg("strike"), nb::arg("spot"), nb::arg("vol"), nb::arg("yf"), nb::arg("r"),
          nb::arg("q"));
    m.def("option_gamma", &fx::option_gamma, nb::arg("notional"), nb::arg("strike"),
          nb::arg("spot"), nb::arg("vol"), nb::arg("yf"), nb::arg("r"), nb::arg("q"));
    m.def("option_vega", &fx::option_vega, nb::arg("notional"), nb::arg("strike"),
          nb::arg("spot"), nb::arg("vol"), nb::arg("yf"), nb::arg("r"), nb::arg("q"));
    m.def("strike_from_delta", &fx::strike_from_delta, nb::arg("delta"), nb::arg("spot"),
          nb::arg("vol"), nb::arg("yf"), nb::arg("r"), nb::arg("q"), nb::arg("df_q"),
          nb::arg("sign"));
}

}  // namespace fts::py
