#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>

#include "discount.hpp"
#include "fx/special.hpp"

// FX forwards, NDFs and Garman-Kohlhagen options: the arithmetic of
// fintoolsom.derivatives.calculator (_forwards, _options), with the same operations in
// the same order. Market lookups, the volatility surface and rate conventions stay in
// Python; exp, log and sqrt are the C library's, as CPython's math module.
namespace fts::fx {

// Market._future_fx_rates_at_ordinals for one fixing after the last known date:
// anchor * (df_base(fixing) / df_base(L)) / (df_quote(fixing) / df_quote(L)).
inline double projected_fx(double anchor, std::int64_t last_known, std::int64_t fixing,
                           const DfSource& base, const DfSource& quote) {
    const std::int64_t nodes[2] = {last_known, fixing};
    double b[2], q[2];
    base(nodes, b);
    quote(nodes, q);
    return anchor * (b[1] / b[0]) / (q[1] / q[0]);
}

// _forwards.get_forward_mtm, after the paid check: in the quote currency, times
// `to_currency` (the spot rate to the reporting currency).
inline double forward_mtm(int sign, double notional, double strike, double spot,
                          std::int64_t pay, const Discount& quote, const Discount& base,
                          double to_currency) {
    const double df_d = quote.at(pay);
    const double df_f = base.at(pay);
    const double notional_leg_vp = spot * notional * df_f;
    const double strike_leg_vp = strike * notional * df_d;
    const double mtm = sign * (notional_leg_vp - strike_leg_vp);
    return mtm * to_currency;
}

// _forwards.get_ndf_mtm, given the settlement FX of the fixing date.
inline double ndf_mtm(int sign, double notional, double strike, double settlement_fx,
                      std::int64_t pay, const Discount& quote, double to_currency) {
    const double fv = sign * notional * (settlement_fx - strike);
    const double mtm = fv * quote.at(pay);
    return mtm * to_currency;
}

inline const double kSqrt2Pi = std::sqrt(2 * std::numbers::pi);  // _options._SQRT_2PI

// Standard normal density, _options._norm_pdf.
inline double norm_pdf(double x) { return std::exp(-x * x / 2) / kSqrt2Pi; }

// log(strike / forward), _options.get_option_log_moneyness.
inline double log_moneyness(double strike, double spot, double df_r, double df_q) {
    const double fwd_price = spot * df_q / df_r;
    return std::log(strike / fwd_price);
}

struct D1 {
    double d1;
    double vol_sqrt_yf;
};

// _options._get_option_d1.
inline D1 option_d1(double spot, double strike, double vol, double yf, double r, double q) {
    const double vol_sqrt_yf = vol * std::sqrt(yf);
    const double d1 = (std::log(spot / strike) + (r - q + vol * vol / 2) * yf) / vol_sqrt_yf;
    return {d1, vol_sqrt_yf};
}

// _options.get_option_mtm.
inline double option_mtm(int sign, double notional, double strike, double spot, double vol,
                         double yf, double r, double q) {
    const auto [d1, vol_sqrt_yf] = option_d1(spot, strike, vol, yf, r, q);
    const double d2 = d1 - vol_sqrt_yf;
    return sign * notional *
           (spot * std::exp(-q * yf) * special::ndtr(sign * d1) -
            strike * std::exp(-r * yf) * special::ndtr(sign * d2));
}

// _options.get_option_delta.
inline double option_delta(int sign, double notional, double strike, double spot, double vol,
                           double yf, double r, double q) {
    const double d1 = option_d1(spot, strike, vol, yf, r, q).d1;
    return sign * notional * std::exp(-q * yf) * special::ndtr(sign * d1);
}

// _options.get_option_gamma.
inline double option_gamma(double notional, double strike, double spot, double vol, double yf,
                           double r, double q) {
    const auto [d1, vol_sqrt_yf] = option_d1(spot, strike, vol, yf, r, q);
    return notional * std::exp(-q * yf) * norm_pdf(d1) / (spot * vol_sqrt_yf);
}

// _options.get_option_vega.
inline double option_vega(double notional, double strike, double spot, double vol, double yf,
                          double r, double q) {
    const double d1 = option_d1(spot, strike, vol, yf, r, q).d1;
    return notional * spot * std::exp(-q * yf) * std::sqrt(yf) * norm_pdf(d1);
}

// _options.get_strike_from_delta.
inline double strike_from_delta(double delta, double spot, double vol, double yf, double r,
                                double q, double df_q, int sign) {
    return spot * std::exp(-(sign * special::ndtri(sign * delta * (1 / df_q)) * vol *
                                 std::sqrt(yf) -
                             (r - q + vol * vol / 2) * yf));
}

}  // namespace fts::fx
