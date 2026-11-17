#pragma once
// The market valuation of bind_market.cpp, for the curve builder's solver
// (bind_solver.cpp): MTMs of instruments and the values a ValuationBatch reads.

#include <cstdint>
#include <span>
#include <vector>

#include "py_curve.hpp"

namespace fts::py {

// One curve lookup of a ValuationBatch (_batch._Source), at its ordinals.
struct BatchSource {
    enum Kind : int { Discount = 0, Index = 1, Accrual = 2, Fx = 3 };
    int kind = Discount;
    nb::object a, b;  // discount: currency, collateral; index, accrual: index; fx: pair
    std::vector<std::int64_t> o0, o1;  // o1: the accrual ends
};

// ValuationBatch(instruments, market, currency, compile_all=True) laid out for its
// gradient: the values one call reads (`sources` end to end, then a constant 1), the
// terms of the MTMs in them, and the log df terms of every value by curve
// (_solver._Jacobian._source_terms).
struct CompiledBatch {
    std::vector<BatchSource> sources;
    std::size_t n_values = 0;  // the trailing 1 excluded
    std::vector<double> log_shift;  // per value (1 for an accrual, else 0), then 0
    // Leg terms flow * fx * df, in MTM row `term_row` with `term_scale` (sign * spot); fx
    // at the trailing 1 for a flow in the leg currency.
    std::vector<double> flows, term_scale;
    std::vector<std::int64_t> fx_at, df_at, term_row;
    // Projected flows: flows[projected_pos] = residual * value[accrual_at] + fixed.
    std::vector<std::int64_t> projected_pos, accrual_at;
    std::vector<double> residuals, fixed;
    // Par legs' heads residual_0 * df(start_0).
    std::vector<double> head_residuals, head_scale;
    std::vector<std::int64_t> head_at, head_row;
    // NDFs a * (s_t - strike) * df * spot; s_t the value at fx when fx >= 0.
    struct Ndf {
        std::int64_t row;
        double a, strike, s_t;
        std::int64_t fx, df;
        double spot;
    };
    std::vector<Ndf> ndfs;
    // log(value[row] + shift) = sum of sign * log df(ordinal) of `curve`.
    struct CurveTerm {
        PyObject* curve;  // kept alive by `curves`
        std::int64_t row, ordinal;
        double sign;
    };
    std::vector<CurveTerm> curve_terms;
    std::vector<nb::object> curves;
};

// Compile `instruments` (a list; kinds 1, a Swap of exactly that type, or 5, an NDF):
// false when one is not compiled (a coupon already running or fixed, another kind).
bool compile_batch(nb::handle instruments, std::span<const std::int64_t> kinds,
                   nb::handle market, nb::handle currency, CompiledBatch& out);

// market_valuate_many of instruments[subset[i]] (instruments a list, kinds per
// instrument) into out[i], on one view of `market`.
void valuate_subset(nb::handle instruments, std::span<const std::int64_t> kinds,
                    std::span<const std::int64_t> subset, nb::handle market,
                    nb::handle currency, double* out);

// _Source.evaluate of every source, end to end, into out.
void evaluate_sources(nb::handle market, std::span<const BatchSource> sources, double* out);

void bind_solver(nb::module_& m);

}  // namespace fts::py
