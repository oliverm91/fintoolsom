#include "swaps/swap_leg.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fts::swaps {

namespace {

using Ordinals = std::span<const std::int64_t>;

// sum(a[i] * b[i]) as math.fsum((a * b).tolist()): products first, then the sum.
double fsum_products(std::span<const double> a, std::span<const double> b) {
    std::vector<double> products(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) products[i] = a[i] * b[i];
    return fsum(products);
}

// Projected flows residual * (1 / (df(end) / df(start)) - 1) + fixed_flow of the
// coupons whose (start, end) ordinals and data are given, as _projected_flows over
// IndexCurve._accrual_at_ordinals (dfs of the ends first, then of the starts).
void project(const DfSource& proj, Ordinals starts, Ordinals ends,
             std::span<const double> residual, std::span<const double> fixed_flows,
             double* out) {
    const std::size_t m = starts.size();
    std::vector<double> df_end(m), df_start(m);
    proj(ends, df_end);
    proj(starts, df_start);
    // Branch-free and loop-invariant: vectorizes.
    for (std::size_t i = 0; i < m; ++i) {
        const double accrual = 1.0 / (df_end[i] / df_start[i]) - 1;
        out[i] = residual[i] * accrual + fixed_flows[i];
    }
}

}  // namespace

double fsum(std::span<const double> values) {
    // CPython Modules/mathmodule.c math_fsum (Shewchuk's exact partials).
    std::vector<double> p;
    p.reserve(32);
    double special_sum = 0.0;
    double inf_sum = 0.0;
    for (double x : values) {
        const double xsave = x;
        std::size_t i = 0;
        for (std::size_t j = 0; j < p.size(); ++j) {
            double y = p[j];
            if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
            const double hi = x + y;
            const double yr = hi - x;
            const double lo = y - yr;
            if (lo != 0.0) p[i++] = lo;
            x = hi;
        }
        p.resize(i);
        if (x != 0.0) {
            if (!std::isfinite(x)) {
                // Intermediate overflow, or a nan or inf among the values.
                if (std::isfinite(xsave)) throw std::overflow_error("intermediate overflow in fsum");
                if (std::isinf(xsave)) inf_sum += xsave;
                special_sum += xsave;
                p.clear();
            } else {
                p.push_back(x);
            }
        }
    }
    if (special_sum != 0.0) {
        if (std::isnan(inf_sum)) throw std::domain_error("-inf + inf in fsum");
        return special_sum;
    }
    double hi = 0.0;
    std::size_t n = p.size();
    if (n > 0) {
        hi = p[--n];
        double lo = 0.0;
        // Sum the partials from the top, stopping when the sum becomes inexact.
        while (n > 0) {
            const double x = hi;
            const double y = p[--n];
            hi = x + y;
            const double yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        // Half-even rounding across partials.
        if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
            const double y = lo * 2.0;
            const double x = hi + y;
            const double yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    return hi;
}

SwapLegCore::SwapLegCore(LegKind kind, Ordinals start, Ordinals end, Ordinals pay,
                         Ordinals fx_fixing, Ordinals fixing, std::span<const double> residual,
                         std::span<const double> flows, std::span<const double> spreads,
                         bool par_eligible, bool has_spread)
    : kind_(kind),
      n_(start.size()),
      start_(start.begin(), start.end()),
      end_(end.begin(), end.end()),
      pay_(pay.begin(), pay.end()),
      fx_fixing_(fx_fixing.begin(), fx_fixing.end()),
      fixing_(fixing.begin(), fixing.end()),
      residual_(residual.begin(), residual.end()),
      flows_(flows.begin(), flows.end()),
      spreads_(spreads.begin(), spreads.end()),
      par_eligible_(par_eligible),
      has_spread_(has_spread) {
    const auto same = [this](std::size_t m) { return m == n_; };
    const bool floating = kind != LegKind::Fixed;
    if (!same(end.size()) || !same(pay.size()) || !same(fx_fixing.size()) ||
        !same(residual.size()) || !same(flows.size()) ||
        (floating && !same(spreads.size())) ||
        (kind == LegKind::TermRate && !same(fixing.size()))) {
        throw std::invalid_argument("SwapLegCore: per-coupon arrays differ in length.");
    }
}

std::size_t SwapLegCore::first_unpaid(std::int64_t t) const noexcept {
    return static_cast<std::size_t>(std::upper_bound(pay_.begin(), pay_.end(), t) -
                                    pay_.begin());
}

bool SwapLegCore::nothing_realised(std::int64_t t) const noexcept {
    const auto& first = kind_ == LegKind::TermRate ? fixing_ : start_;
    return !first.empty() && first.front() > t;
}

bool SwapLegCore::has_realised(std::int64_t t) const noexcept {
    if (kind_ == LegKind::Fixed) return false;
    const auto& on = kind_ == LegKind::TermRate ? fixing_ : start_;
    for (std::size_t i = first_unpaid(t); i < n_; ++i) {
        if (on[i] <= t) return true;
    }
    return false;
}

std::vector<std::int64_t> SwapLegCore::known_fx_fixings(std::int64_t t, std::int64_t last_known,
                                                        bool& any_unknown) const {
    std::vector<std::int64_t> known;
    any_unknown = false;
    for (std::size_t i = first_unpaid(t); i < n_; ++i) {
        if (fx_fixing_[i] <= last_known) {
            known.push_back(fx_fixing_[i]);
        } else {
            any_unknown = true;
        }
    }
    return known;
}

void SwapLegCore::projected(std::size_t lo, const DfSource& proj, double* out) const {
    const std::size_t m = n_ - lo;
    project(proj, Ordinals(start_).subspan(lo, m), Ordinals(end_).subspan(lo, m),
            std::span(residual_).subspan(lo, m), std::span(flows_).subspan(lo, m), out);
}

std::vector<double> SwapLegCore::unpaid_flows(std::int64_t t, const DfSource* proj,
                                              Known known) const {
    const std::size_t k = first_unpaid(t);
    const std::size_t m = n_ - k;
    std::vector<double> out(m);
    if (m == 0) return out;
    if (kind_ == LegKind::Fixed) {
        std::copy(flows_.begin() + static_cast<std::ptrdiff_t>(k), flows_.end(), out.begin());
        return out;
    }
    if (known.index.empty()) {  // every unpaid coupon projects
        projected(k, *proj, out.data());
        return out;
    }
    // Project the coupons not known, then place the known flows.
    std::vector<char> is_known(m, 0);
    for (const std::int64_t i : known.index) is_known[static_cast<std::size_t>(i) - k] = 1;
    std::vector<std::int64_t> starts, ends;
    std::vector<double> residual, fixed;
    std::vector<std::size_t> at;
    for (std::size_t j = 0; j < m; ++j) {
        if (is_known[j]) continue;
        const std::size_t i = k + j;
        starts.push_back(start_[i]);
        ends.push_back(end_[i]);
        residual.push_back(residual_[i]);
        fixed.push_back(flows_[i]);
        at.push_back(j);
    }
    if (!at.empty()) {
        std::vector<double> projected_flows(at.size());
        project(*proj, starts, ends, residual, fixed, projected_flows.data());
        for (std::size_t q = 0; q < at.size(); ++q) out[at[q]] = projected_flows[q];
    }
    for (std::size_t q = 0; q < known.index.size(); ++q) {
        out[static_cast<std::size_t>(known.index[q]) - k] = known.flow[q];
    }
    return out;
}

double SwapLegCore::par_pv(const DfSource& curve) const {
    double df_start = 0.0;
    curve(Ordinals(start_).first(1), std::span(&df_start, 1));
    double pv = residual_[0] * df_start;
    if (has_spread_) {
        std::vector<double> df_end(n_);
        curve(end_, df_end);
        pv += fsum_products(spreads_, df_end);
    }
    return pv;
}

double SwapLegCore::pv(std::int64_t t, const DfSource* proj, Known known,
                       const Discount& discount) const {
    const std::size_t k = first_unpaid(t);
    if (k == n_) return 0.0;
    const std::vector<double> flows = unpaid_flows(t, proj, known);
    std::vector<double> dfs(n_ - k);
    discount(Ordinals(pay_).subspan(k), dfs);
    return fsum_products(flows, dfs);
}

double SwapLegCore::pv_non_deliverable(std::int64_t t, const DfSource* proj, Known known,
                                       const FxProjection& fx, const Discount& discount) const {
    const std::size_t k = first_unpaid(t);
    if (k == n_) return 0.0;
    const std::size_t m = n_ - k;
    const std::vector<double> flows = unpaid_flows(t, proj, known);

    // FX at each unpaid fixing: known, or anchor * (b / b0) / (q / q0).
    std::vector<double> rates(m);
    std::vector<std::int64_t> nodes{fx.last_known};
    std::vector<std::size_t> projected_at;
    std::size_t next_known = 0;
    for (std::size_t j = 0; j < m; ++j) {
        const std::int64_t fixing = fx_fixing_[k + j];
        if (fixing <= fx.last_known) {
            rates[j] = fx.known[next_known++];
        } else {
            nodes.push_back(fixing);
            projected_at.push_back(j);
        }
    }
    if (!projected_at.empty()) {
        std::vector<double> df_base(nodes.size()), df_quote(nodes.size());
        (*fx.base)(nodes, df_base);
        (*fx.quote)(nodes, df_quote);
        for (std::size_t q = 0; q < projected_at.size(); ++q) {
            rates[projected_at[q]] =
                fx.anchor * (df_base[q + 1] / df_base[0]) / (df_quote[q + 1] / df_quote[0]);
        }
    }

    std::vector<double> dfs(m);
    discount(Ordinals(pay_).subspan(k), dfs);
    // np.sum(flows * fx * dfs): (flows * fx) * dfs, then the sum.
    std::vector<double> products(m);
    for (std::size_t j = 0; j < m; ++j) products[j] = flows[j] * rates[j] * dfs[j];
    return fsum(products);
}

double SwapLegCore::non_started_pv(const DfSource* proj, const Discount& discount) const {
    std::vector<double> flows(n_);
    if (kind_ == LegKind::Fixed) {
        flows = flows_;
    } else if (n_ > 0) {
        projected(0, *proj, flows.data());
    }
    std::vector<double> dfs(n_);
    discount(pay_, dfs);
    return fsum_products(flows, dfs);
}

namespace {

// np.isclose(x, y, rtol=1e-12, atol=0.0): numpy's formula, infinities included.
bool isclose(double x, double y) {
    return (std::fabs(x - y) <= 0.0 + 1e-12 * std::fabs(y) && std::isfinite(y)) || x == y;
}

}  // namespace

bool repays_at_par(std::span<const std::int64_t> pay, std::span<const std::int64_t> start,
                   std::span<const std::int64_t> end, std::span<const double> r,
                   std::span<const double> a) {
    const std::size_t n = pay.size();
    if (n == 0) return false;
    if (!std::equal(pay.begin(), pay.end(), end.begin(), end.end())) return false;
    if (!std::equal(start.begin() + 1, start.end(), end.begin(), end.end() - 1)) return false;
    for (std::size_t i = 1; i < n; ++i) {
        if (!isclose(r[i], r[i - 1] - a[i - 1])) return false;
    }
    return isclose(r[n - 1], a[n - 1]);
}

void SwapLegCore::rate_sensitive(std::int64_t t, std::span<const double> unpaid,
                                 std::vector<std::int64_t>& dates,
                                 std::vector<double>& flows) const {
    const std::size_t lo = first_unpaid(t);
    dates.clear();
    flows.clear();
    std::size_t first = n_;  // first unpaid coupon whose rate is not set
    if (kind_ != LegKind::Fixed) {
        const std::vector<std::int64_t>& set_by = kind_ == LegKind::TermRate ? fixing_ : start_;
        for (std::size_t i = lo; i < n_; ++i) {
            if (!(set_by[i] <= t)) {
                first = i;
                break;
            }
        }
    }
    for (std::size_t i = lo; i < first; ++i) {
        dates.push_back(pay_[i]);
        flows.push_back(unpaid[i - lo]);
    }
    if (first == n_) return;
    dates.push_back(start_[first]);
    flows.push_back(residual_[first]);
    for (std::size_t i = first; i < n_; ++i) {
        dates.push_back(pay_[i]);
        flows.push_back(spreads_[i]);
    }
}

double duration(std::int64_t t, std::span<const std::int64_t> dates,
                std::span<const double> flows, std::span<const double> dfs) {
    std::vector<double> pvs(dates.size()), weighted(dates.size());
    for (std::size_t i = 0; i < dates.size(); ++i) {
        pvs[i] = flows[i] * dfs[i];
        const double tau = static_cast<double>(dates[i] - t) / 365.0;
        weighted[i] = tau * pvs[i];
    }
    const double total = fsum(pvs);
    return total != 0.0 ? fsum(weighted) / total : 0.0;
}

}  // namespace fts::swaps
