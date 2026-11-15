#pragma once

#include <cmath>
#include <stdexcept>

// Interest conventions (fintoolsom.rates.Rates): the wealth factor of a rate over a
// time fraction, with the Python operations (``1 + r * t``, ``(1 + r) ** t``,
// ``exp(r * t)``) and their inverses.
namespace fts::rates {

// The InterestConventionBase subclasses, by kind.
enum class Interest { Linear = 0, Compounded = 1, Exponential = 2 };

inline Interest interest_of(int kind) {
    if (kind < 0 || kind > 2) throw std::invalid_argument("Unknown interest convention.");
    return static_cast<Interest>(kind);
}

// get_wf_from_rate.
inline double wealth_factor(Interest kind, double rate, double time_fraction) {
    switch (kind) {
        case Interest::Linear:
            return 1.0 + rate * time_fraction;
        case Interest::Compounded:
            return std::pow(1.0 + rate, time_fraction);
        case Interest::Exponential:
            return std::exp(rate * time_fraction);
    }
    throw std::invalid_argument("Unknown interest convention.");
}

// get_rate_from_wf.
inline double rate_from_wf(Interest kind, double wf, double time_fraction) {
    switch (kind) {
        case Interest::Linear:
            return (wf - 1.0) / time_fraction;
        case Interest::Compounded:
            return std::pow(wf, 1.0 / time_fraction) - 1.0;
        case Interest::Exponential:
            return std::log(wf) / time_fraction;
    }
    throw std::invalid_argument("Unknown interest convention.");
}

// What a Rate method returns from its wealth factor.
enum class Factor { Wealth = 0, Discount = 1, Accrued = 2 };

// get_wealth_factor, get_discount_factor (1 / wf) or get_accrued_interest
// (notional * (wf - 1)).
inline double factor(Factor what, double wf, double notional) {
    switch (what) {
        case Factor::Wealth:
            return wf;
        case Factor::Discount:
            return 1.0 / wf;
        case Factor::Accrued:
            return notional * (wf - 1.0);
    }
    throw std::invalid_argument("Unknown rate factor.");
}

}  // namespace fts::rates
