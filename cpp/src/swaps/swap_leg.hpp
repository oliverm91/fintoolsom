#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "discount.hpp"

// Swap leg valuation: the curve-dependent arithmetic of
// fintoolsom.derivatives.calculator (_swap_pv, _simplified_swap_pv), with the same
// operations in the same order. Market lookups and realised fixings stay in Python.
namespace fts::swaps {

// SwapLeg subclasses: a fixed leg, or a floating leg projected from its index curve
// (overnight and cross-currency legs from the start date, term-rate legs from the
// fixing date).
enum class LegKind : int { Fixed = 0, Overnight = 1, TermRate = 2 };

using fts::DfSource;
using fts::Discount;

// FX rates of non-deliverable flows, as Market._future_fx_rates_at_ordinals: a fixing
// on or before `last_known` is a known rate (in `known`, in coupon order); a later one
// is projected from `anchor` on the funding curves `base` and `quote`.
struct FxProjection {
    std::int64_t last_known = 0;
    std::span<const double> known;
    double anchor = 0.0;
    const DfSource* base = nullptr;
    const DfSource* quote = nullptr;
};

// Flows of unpaid coupons known without projection (running overnight coupons, set
// term-rate coupons): coupon indices and their flows.
struct Known {
    std::span<const std::int64_t> index;
    std::span<const double> flow;
};

// math.fsum: the correctly rounded sum (CPython's algorithm, special values included).
double fsum(std::span<const double> values);

// One swap leg's per-coupon data as flat arrays (structure of arrays).
class SwapLegCore {
public:
    SwapLegCore(LegKind kind, std::span<const std::int64_t> start,
                std::span<const std::int64_t> end, std::span<const std::int64_t> pay,
                std::span<const std::int64_t> fx_fixing, std::span<const std::int64_t> fixing,
                std::span<const double> residual, std::span<const double> flows,
                std::span<const double> spreads, bool par_eligible, bool has_spread);

    [[nodiscard]] LegKind kind() const noexcept { return kind_; }
    [[nodiscard]] std::size_t size() const noexcept { return n_; }
    [[nodiscard]] bool par_eligible() const noexcept { return par_eligible_; }
    // First coupon paid after t: payment dates are non-decreasing.
    [[nodiscard]] std::size_t first_unpaid(std::int64_t t) const noexcept;
    [[nodiscard]] bool has_unpaid(std::int64_t t) const noexcept { return first_unpaid(t) < n_; }
    // _swap_pv._nothing_realised: the first coupon has not started (fixed, term rate).
    [[nodiscard]] bool nothing_realised(std::int64_t t) const noexcept;
    // Whether an unpaid coupon has started (overnight) or fixed (term rate) by t.
    [[nodiscard]] bool has_realised(std::int64_t t) const noexcept;
    // FX fixing ordinals of the unpaid coupons fixed on or before last_known, in order;
    // any_unknown tells whether some fix later.
    [[nodiscard]] std::vector<std::int64_t> known_fx_fixings(std::int64_t t,
                                                             std::int64_t last_known,
                                                             bool& any_unknown) const;

    // _swap_pv._leg_flows: flows of the coupons paid after t.
    [[nodiscard]] std::vector<double> unpaid_flows(std::int64_t t, const DfSource* proj,
                                                   Known known) const;
    // _swap_pv._par_leg_pv.
    [[nodiscard]] double par_pv(const DfSource& curve) const;
    // _swap_pv._flows_pv.
    [[nodiscard]] double pv(std::int64_t t, const DfSource* proj, Known known,
                            const Discount& discount) const;
    // _swap_pv._leg_pv_non_deliverable, for a leg settled in another currency.
    [[nodiscard]] double pv_non_deliverable(std::int64_t t, const DfSource* proj, Known known,
                                            const FxProjection& fx,
                                            const Discount& discount) const;
    // _simplified_swap_pv._non_started_leg_pv, off the par shortcut.
    [[nodiscard]] double non_started_pv(const DfSource* proj, const Discount& discount) const;

private:
    // Flows of coupons [lo, n): fixed flows, or projected over [start, end].
    void projected(std::size_t lo, const DfSource& proj, double* out) const;

    LegKind kind_;
    std::size_t n_;
    std::vector<std::int64_t> start_, end_, pay_, fx_fixing_, fixing_;
    std::vector<double> residual_;
    std::vector<double> flows_;  // fixed: interest + amortization; floating: fixed_flows
    std::vector<double> spreads_;
    bool par_eligible_;
    bool has_spread_;
};

}  // namespace fts::swaps
