// The curve builder's group solve in C++ (fintoolsom._native.GroupSolver).
//
// curve_builder._solver lays out a group's unknowns (the forwards of its free pillars)
// and hands the solve here when every curve has a C++ core and every residual
// instrument is valued in one C++ market call. The solve then runs without Python per
// iteration: the unknowns are set on the curves' cores in place, the residuals are
// market_valuate_many's, and
//
// * sweep: ascending date blocks solved by Newton steps, Broyden-updated: _solver._newton
//   and _solver._solve_small line for line (math.fsum included), so the same steps;
// * joint: Newton steps on every unknown at once with the analytic Jacobian of
//   _solver._Jacobian (the batch gradient times d log df / d forwards, read off private
//   copies of the cores), with backtracking. It replaces scipy's hybr: the root is the
//   same, reached by another path.
//
// What they do not settle returns to Python: a block Newton gives up on is solved by
// hybr there (as the Python sweep does) and the sweep resumes; a joint solve that stops
// short, or an error raised by a valuation, has Python redo it its own way.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "py_market.hpp"

namespace fts::py {

namespace {

// _solver's constants.
constexpr double kNewtonBump = 1e-7, kNewtonXtol = 1e-12, kNewtonNoise = 1e-10;
constexpr int kNewtonSteps = 8;
// Joint Newton: at most kJointSteps steps, each halved at most kJointHalvings times.
constexpr int kJointSteps = 40, kJointHalvings = 12;

using Matrix = std::vector<std::vector<double>>;

// math.fsum (CPython's algorithm: exact partials, then the correctly rounded total).
double fsum(std::span<const double> xs) {
    std::vector<double> p;
    p.reserve(8);
    double naive = 0.0;
    bool finite = true;
    for (double x : xs) {
        naive += x;
        if (!std::isfinite(x)) {
            finite = false;
            continue;
        }
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
        if (x != 0.0) p.push_back(x);
    }
    if (!finite) return naive;  // inf or nan, as fsum's special sum
    std::size_t n = p.size();
    double hi = 0.0, lo = 0.0;
    if (n > 0) {
        hi = p[--n];
        while (n > 0) {
            const double x = hi;
            const double y = p[--n];
            hi = x + y;
            const double yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
            const double y = lo * 2.0;
            const double x = hi + y;
            const double yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    return hi;
}

// Python's max(map(abs, v)): the first element, replaced by any later one greater.
double max_abs(std::span<const double> v) {
    double m = std::fabs(v[0]);
    for (std::size_t i = 1; i < v.size(); ++i) {
        const double a = std::fabs(v[i]);
        if (a > m) m = a;
    }
    return m;
}

// _solver._solve_small: a x = b by Gaussian elimination with partial pivoting.
std::optional<std::vector<double>> solve_small(const Matrix& a, std::span<const double> b) {
    const std::size_t n = b.size();
    if (n == 1) {
        if (a[0][0] == 0.0) return std::nullopt;
        return std::vector<double>{b[0] / a[0][0]};
    }
    Matrix m(n);
    for (std::size_t i = 0; i < n; ++i) {
        m[i] = a[i];
        m[i].push_back(b[i]);
    }
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t p = c;
        for (std::size_t i = c + 1; i < n; ++i) {
            if (std::fabs(m[i][c]) > std::fabs(m[p][c])) p = i;
        }
        if (m[p][c] == 0.0) return std::nullopt;
        std::swap(m[c], m[p]);
        const std::vector<double>& pivot = m[c];
        for (std::size_t i = c + 1; i < n; ++i) {
            const double f = m[i][c] / pivot[c];
            if (f != 0.0) {  // Python's `if f:` (nan is true)
                std::vector<double>& row = m[i];
                for (std::size_t k = c; k <= n; ++k) row[k] -= f * pivot[k];
            }
        }
    }
    std::vector<double> x(n, 0.0), terms;
    for (std::size_t i = n; i-- > 0;) {
        const std::vector<double>& row = m[i];
        terms.clear();
        for (std::size_t k = i + 1; k < n; ++k) terms.push_back(row[k] * x[k]);
        x[i] = (row[n] - fsum(terms)) / row[i];
    }
    return x;
}

bool all_finite(std::span<const double> v) {
    return std::all_of(v.begin(), v.end(), [](double d) { return std::isfinite(d); });
}

struct NewtonResult {
    std::vector<double> x;
    Matrix jac;
};

// _solver._newton. `residual(x, r)` returns false when the valuation raised.
template <class Residual>
std::optional<NewtonResult> newton(Residual&& residual, std::vector<double> x,
                                   const Matrix* jac0) {
    const std::size_t n = x.size();
    std::vector<double> r, r_new, bumped, terms;
    if (!residual(x, r) || !all_finite(r)) return std::nullopt;
    Matrix jac;
    if (jac0 == nullptr) {
        jac.assign(n, std::vector<double>(n, 0.0));
        for (std::size_t k = 0; k < n; ++k) {
            bumped = x;
            bumped[k] += kNewtonBump;
            if (!residual(bumped, r_new)) return std::nullopt;
            for (std::size_t i = 0; i < std::min(n, r_new.size()); ++i) {
                jac[i][k] = (r_new[i] - r[i]) / kNewtonBump;
            }
        }
    } else {
        jac = *jac0;
    }
    double norm = max_abs(r);
    std::vector<double> minus_r(n), x_new(n);
    for (int step_count = 0; step_count < kNewtonSteps; ++step_count) {
        for (std::size_t i = 0; i < n; ++i) minus_r[i] = -r[i];
        const auto dx = solve_small(jac, minus_r);
        if (!dx) return std::nullopt;
        const double step = max_abs(*dx);
        if (!std::isfinite(step)) return std::nullopt;
        for (std::size_t i = 0; i < n; ++i) x_new[i] = x[i] + (*dx)[i];
        if (step <= kNewtonXtol) return NewtonResult{x_new, std::move(jac)};
        if (!residual(x_new, r_new)) return std::nullopt;
        const double norm_new = max_abs(r_new);
        if (!std::isfinite(norm_new)) return std::nullopt;
        if (norm_new >= norm) {
            // Not lower: converged to the residuals' rounding, or diverging.
            if (step <= kNewtonNoise) return NewtonResult{x, std::move(jac)};
            return std::nullopt;
        }
        // Broyden: J += (dr - J dx) dx^T / (dx . dx)
        terms.resize(n);
        for (std::size_t k = 0; k < n; ++k) terms[k] = (*dx)[k] * (*dx)[k];
        const double dd = fsum(terms);
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<double>& row = jac[i];
            for (std::size_t k = 0; k < n; ++k) terms[k] = row[k] * (*dx)[k];
            const double u = (r_new[i] - r[i] - fsum(terms)) / dd;
            for (std::size_t k = 0; k < n; ++k) row[k] += u * (*dx)[k];
        }
        x = x_new;
        std::swap(r, r_new);
        norm = norm_new;
    }
    return std::nullopt;
}

// a x = b for a dense n x n row-major `a` (both overwritten); false if singular.
bool lu_solve(std::vector<double>& a, std::vector<double>& b, std::size_t n) {
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t p = c;
        for (std::size_t i = c + 1; i < n; ++i) {
            if (std::fabs(a[i * n + c]) > std::fabs(a[p * n + c])) p = i;
        }
        const double pivot = a[p * n + c];
        if (pivot == 0.0 || !std::isfinite(pivot)) return false;
        if (p != c) {
            std::swap_ranges(a.begin() + static_cast<std::ptrdiff_t>(c * n),
                             a.begin() + static_cast<std::ptrdiff_t>((c + 1) * n),
                             a.begin() + static_cast<std::ptrdiff_t>(p * n));
            std::swap(b[c], b[p]);
        }
        for (std::size_t i = c + 1; i < n; ++i) {
            const double f = a[i * n + c] / pivot;
            if (f == 0.0) continue;
            for (std::size_t k = c; k < n; ++k) a[i * n + k] -= f * a[c * n + k];
            b[i] -= f * b[c];
        }
    }
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t k = i + 1; k < n; ++k) s -= a[i * n + k] * b[k];
        b[i] = s / a[i * n + i];
    }
    return all_finite(b);
}

std::vector<double> doubles_of(nb::handle a) {
    const auto in = nb::cast<DoublesIn>(a);
    return {in.data(), in.data() + in.shape(0)};
}

std::vector<std::int64_t> ints_of(nb::handle a) {
    const auto in = nb::cast<DaysIn>(a);
    return {in.data(), in.data() + in.shape(0)};
}

// One curve solved: its core in the market, the fixed forwards then the free ones.
struct SolvedCurve {
    nb::object curve, core_obj;
    PyPillarCurve* core = nullptr;
    std::vector<double> forwards;
    std::size_t n_fixed = 0, start = 0, stop = 0;
    double tau0 = 0.0;
    std::optional<double> spot_log_df;  // IndexCurve with a spot segment: constant

    // PillarCurve._log_df_first_knot (IndexCurve's when it has a spot segment).
    [[nodiscard]] double log_df(std::span<const double> f) const {
        return spot_log_df ? *spot_log_df : -f[0] * tau0;
    }
};

// The log df terms of one curve in the batch values (_Jacobian._source_terms): value
// rows[t] gets signs[t] * log df(ordinals[at[t]]).
struct CurveTerms {
    std::vector<std::int64_t> ordinals, rows, at;
    std::vector<double> signs;
};

struct Block {
    std::vector<std::int64_t> unknowns, instruments;
    std::optional<Matrix> jac;  // Newton's last Jacobian of the block
};

class GroupSolver {
public:
    GroupSolver(nb::handle market, nb::handle currency, nb::list curves, nb::list instruments,
                DaysIn kinds)
        : market_(nb::borrow(market)),
          currency_(nb::borrow(currency)),
          instruments_(std::move(instruments)),
          kinds_(kinds.data(), kinds.data() + kinds.shape(0)) {
        if (kinds_.size() != nb::len(instruments_)) {
            throw std::invalid_argument("One kind per instrument.");
        }
        for (nb::handle item : curves) {
            // (curve, fixed forwards, start, stop, tau0, spot log df or None)
            const nb::tuple t = nb::cast<nb::tuple>(item);
            SolvedCurve c;
            c.curve = nb::borrow(t[0]);
            c.core_obj = c.curve.attr("_core");
            c.core = nb::cast<PyPillarCurve*>(c.core_obj);
            c.forwards = doubles_of(t[1]);
            c.n_fixed = c.forwards.size();
            c.start = nb::cast<std::size_t>(t[2]);
            c.stop = nb::cast<std::size_t>(t[3]);
            c.forwards.resize(c.n_fixed + c.stop - c.start, 0.0);
            c.tau0 = nb::cast<double>(t[4]);
            if (!t[5].is_none()) c.spot_log_df = nb::cast<double>(t[5]);
            n_unknowns_ = std::max(n_unknowns_, c.stop);
            curves_.push_back(std::move(c));
        }
        all_.resize(kinds_.size());
        for (std::size_t i = 0; i < all_.size(); ++i) all_[i] = static_cast<std::int64_t>(i);
    }

    // The date blocks of the sweeps, ascending: (unknowns, instruments) index arrays.
    void set_blocks(nb::list blocks) {
        blocks_.clear();
        for (nb::handle item : blocks) {
            const nb::tuple t = nb::cast<nb::tuple>(item);
            blocks_.push_back(Block{ints_of(t[0]), ints_of(t[1]), std::nullopt});
            if (blocks_.back().unknowns.size() != blocks_.back().instruments.size()) {
                throw std::invalid_argument("A block needs one instrument per unknown.");
            }
        }
    }

    // The sweeps by block Newton (_solve_unknowns), from block `block` of sweep `sweep`, at
    // most `max_sweeps` sweeps: (status, y, sweep, block). Status 0: every residual within
    // `tol`; 1: the sweeps ended above it; 2: Newton gave up on that block (y as before
    // it), which Python solves by hybr before resuming at the next block; 3: valuing every
    // instrument raised (Python redoes the solve, raising the calculator's error).
    nb::tuple sweep(DoublesIn y0, int max_sweeps, double tol, int sweep, std::size_t block) {
        std::vector<double> y(y0.data(), y0.data() + y0.shape(0));
        if (y.size() != n_unknowns_) throw std::invalid_argument("One value per unknown.");
        std::vector<double> r;
        auto out = [&](int status, int s, std::size_t b) {
            return nb::make_tuple(status, to_array<double>(y), s, b);
        };
        for (int s = sweep; s < max_sweeps; ++s, block = 0) {
            for (std::size_t k = block; k < blocks_.size(); ++k) {
                Block& b = blocks_[k];
                auto residual = [&](const std::vector<double>& x, std::vector<double>& res) {
                    for (std::size_t i = 0; i < x.size(); ++i) {
                        y[static_cast<std::size_t>(b.unknowns[i])] = x[i];
                    }
                    return residuals(y, b.instruments, res);
                };
                std::vector<double> x0(b.unknowns.size());
                for (std::size_t i = 0; i < x0.size(); ++i) {
                    x0[i] = y[static_cast<std::size_t>(b.unknowns[i])];
                }
                auto solved = newton(residual, x0, b.jac ? &*b.jac : nullptr);
                const std::vector<double>& x = solved ? solved->x : x0;
                for (std::size_t i = 0; i < x0.size(); ++i) {
                    y[static_cast<std::size_t>(b.unknowns[i])] = x[i];
                }
                if (!solved) return out(2, s, k);
                b.jac = std::move(solved->jac);
            }
            if (!residuals(y, all_, r)) return out(3, s, 0);
            if (max_abs(r) <= tol) return out(0, s, 0);
        }
        return out(1, max_sweeps, 0);
    }

    // Every unknown at once by Newton steps with the analytic Jacobian, backtracking;
    // None when they stop short of the residuals' rounding, or the batch does not
    // compile (Python then runs hybr). `linear`: log df is affine in the forwards on
    // every curve, so d log df / d y is read once, with the bump `bump`.
    nb::object joint(DoublesIn y0, bool linear, double bump) {
        if (!prepare_jacobian(linear, bump)) return nb::none();
        const std::size_t n = n_unknowns_;
        std::vector<double> x(y0.data(), y0.data() + y0.shape(0)), x_new(n), r, r_new, dx, jac;
        if (x.size() != n || kinds_.size() != n) return nb::none();
        if (!residuals(x, all_, r) || !all_finite(r)) return nb::none();
        double norm = max_abs(r);
        for (int it = 0; it < kJointSteps; ++it) {
            if (!jacobian(x, jac)) return nb::none();
            dx.resize(n);
            for (std::size_t i = 0; i < n; ++i) dx[i] = -r[i];
            if (!lu_solve(jac, dx, n)) return nb::none();
            const double step = max_abs(dx);
            if (step <= kNewtonXtol) {
                for (std::size_t i = 0; i < n; ++i) x[i] += dx[i];
                return nb::cast(to_array<double>(x));
            }
            double scale = 1.0;
            bool lower = false;
            for (int h = 0; h <= kJointHalvings; ++h, scale *= 0.5) {
                for (std::size_t i = 0; i < n; ++i) x_new[i] = x[i] + scale * dx[i];
                if (!residuals(x_new, all_, r_new)) return nb::none();
                const double norm_new = max_abs(r_new);
                if (std::isfinite(norm_new) && norm_new < norm) {
                    lower = true;
                    norm = norm_new;
                    break;
                }
                // Not lower: converged to the residuals' rounding, or a step too long.
                if (scale * step <= kNewtonNoise) return nb::cast(to_array<double>(x));
            }
            if (!lower) return nb::none();
            std::swap(x, x_new);
            std::swap(r, r_new);
        }
        return nb::none();
    }

    // The residuals at y (for tests), None when the valuation raised.
    nb::object residuals_at(DoublesIn y) {
        std::vector<double> x(y.data(), y.data() + y.shape(0)), r;
        if (x.size() != n_unknowns_ || !residuals(x, all_, r)) return nb::none();
        return nb::cast(to_array<double>(r));
    }

    // The joint Jacobian at y, instruments by unknowns (for tests), None on failure.
    nb::object jacobian_at(DoublesIn y, bool linear, double bump) {
        if (!prepare_jacobian(linear, bump)) return nb::none();
        std::vector<double> x(y.data(), y.data() + y.shape(0)), jac;
        if (x.size() != n_unknowns_ || !jacobian(x, jac)) return nb::none();
        const std::size_t rows = kinds_.size(), cols = n_unknowns_;
        auto* buffer = new double[rows * cols];
        std::copy(jac.begin(), jac.end(), buffer);
        nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<2>>(buffer, {rows, cols}, owner));
    }

    // Set the unknowns y on the cores (_GroupSolve.apply, the cores only).
    void apply(std::span<const double> y) {
        for (SolvedCurve& c : curves_) {
            std::copy(y.begin() + static_cast<std::ptrdiff_t>(c.start),
                      y.begin() + static_cast<std::ptrdiff_t>(c.stop),
                      c.forwards.begin() + static_cast<std::ptrdiff_t>(c.n_fixed));
            c.core->core.set_forwards(c.forwards, c.log_df(c.forwards));
        }
    }

private:
    // Compile the batch (once) and lay its log df terms out by solved curve.
    bool prepare_jacobian(bool linear, double bump) {
        if (compiled_ && linear == linear_ && bump == bump_) return prepared_;
        if (!compiled_) {
            compiled_ = true;
            try {
                prepared_ = compile_batch(instruments_, kinds_, market_, currency_, batch_);
            } catch (const nb::python_error&) {
                prepared_ = false;
            } catch (const std::exception&) {
                prepared_ = false;
            }
            if (prepared_) lay_out_curve_terms();
        }
        linear_ = linear;
        bump_ = bump;
        dg_.clear();
        return prepared_;
    }

    // _Jacobian._source_terms: the terms of the curves solved, each curve's ordinals
    // made unique (np.unique).
    void lay_out_curve_terms() {
        curve_terms_.assign(curves_.size(), CurveTerms{});
        for (std::size_t j = 0; j < curves_.size(); ++j) {
            CurveTerms& c = curve_terms_[j];
            const PyObject* curve = curves_[j].curve.ptr();
            for (const auto& t : batch_.curve_terms) {
                if (t.curve != curve) continue;
                c.rows.push_back(t.row);
                c.at.push_back(t.ordinal);  // an ordinal until made an index below
                c.signs.push_back(t.sign);
            }
            c.ordinals = c.at;
            std::sort(c.ordinals.begin(), c.ordinals.end());
            c.ordinals.erase(std::unique(c.ordinals.begin(), c.ordinals.end()), c.ordinals.end());
            for (std::int64_t& a : c.at) {
                a = std::lower_bound(c.ordinals.begin(), c.ordinals.end(), a) - c.ordinals.begin();
            }
        }
    }

    bool residuals(std::span<const double> y, std::span<const std::int64_t> subset,
                   std::vector<double>& out) {
        apply(y);
        out.resize(subset.size());
        try {
            valuate_subset(instruments_, kinds_, subset, market_, currency_, out.data());
        } catch (const nb::python_error&) {
            return false;
        } catch (const std::exception&) {
            return false;
        }
        return true;
    }

    // d log(v + shift) / d y (_Jacobian._curve_part), into dg_ (values by unknowns).
    bool curve_part(std::span<const double> y) {
        const std::size_t n = n_unknowns_;
        dg_.assign((batch_.n_values + 1) * n, 0.0);
        std::vector<double> base, bumped_dfs, forwards;
        try {
            for (std::size_t j = 0; j < curves_.size(); ++j) {
                const SolvedCurve& c = curves_[j];
                const CurveTerms& terms = curve_terms_[j];
                if (terms.ordinals.empty() || c.stop == c.start) continue;
                PyPillarCurve curve = *c.core;  // private: bumped freely
                forwards = c.forwards;
                std::copy(y.begin() + static_cast<std::ptrdiff_t>(c.start),
                          y.begin() + static_cast<std::ptrdiff_t>(c.stop),
                          forwards.begin() + static_cast<std::ptrdiff_t>(c.n_fixed));
                curve.core.set_forwards(forwards, c.log_df(forwards));
                const std::size_t m = terms.ordinals.size();
                base.resize(m);
                bumped_dfs.resize(m);
                curve.dfs_at_ordinals(terms.ordinals, base);
                for (double& b : base) b = std::log(b);
                for (std::size_t k = c.start; k < c.stop; ++k) {
                    std::vector<double> bumped = forwards;
                    bumped[c.n_fixed + k - c.start] += bump_;
                    curve.core.set_forwards(bumped, c.log_df(bumped));
                    curve.dfs_at_ordinals(terms.ordinals, bumped_dfs);
                    for (std::size_t t = 0; t < terms.rows.size(); ++t) {
                        const auto at = static_cast<std::size_t>(terms.at[t]);
                        const double dlog = (std::log(bumped_dfs[at]) - base[at]) / bump_;
                        dg_[static_cast<std::size_t>(terms.rows[t]) * n + k] +=
                            terms.signs[t] * dlog;
                    }
                }
            }
        } catch (const nb::python_error&) {
            return false;
        } catch (const std::exception&) {
            return false;
        }
        return true;
    }

    // d residuals / d y at y (_Jacobian.__call__), row-major instruments by unknowns.
    bool jacobian(std::span<const double> y, std::vector<double>& jac) {
        const std::size_t n = n_unknowns_;
        const CompiledBatch& b = batch_;
        apply(y);
        v_.resize(b.n_values + 1);
        try {
            evaluate_sources(market_, b.sources, v_.data());
        } catch (const nb::python_error&) {
            return false;
        } catch (const std::exception&) {
            return false;
        }
        v_[b.n_values] = 1.0;
        if (!linear_ || dg_.empty()) {
            if (!curve_part(y)) return false;
        }
        jac.assign(kinds_.size() * n, 0.0);
        // ValuationBatch.gradient's entries, each times (v + shift) d g / d y of its value.
        auto add = [&](std::int64_t row, std::int64_t col, double coeff) {
            const auto c = static_cast<std::size_t>(col);
            coeff *= v_[c] + b.log_shift[c];
            if (coeff == 0.0) return;
            const double* d = dg_.data() + c * n;
            double* out = jac.data() + static_cast<std::size_t>(row) * n;
            for (std::size_t k = 0; k < n; ++k) out[k] += coeff * d[k];
        };
        auto value = [&](std::int64_t at) { return v_[static_cast<std::size_t>(at)]; };
        flows_ = b.flows;
        for (std::size_t t = 0; t < b.projected_pos.size(); ++t) {
            flows_[static_cast<std::size_t>(b.projected_pos[t])] =
                b.residuals[t] * value(b.accrual_at[t]) + b.fixed[t];
        }
        for (std::size_t t = 0; t < flows_.size(); ++t) {
            const double fx = value(b.fx_at[t]), df = value(b.df_at[t]);
            add(b.term_row[t], b.df_at[t], flows_[t] * fx * b.term_scale[t]);
            add(b.term_row[t], b.fx_at[t], flows_[t] * df * b.term_scale[t]);
        }
        for (std::size_t t = 0; t < b.projected_pos.size(); ++t) {  // flow = r * accrual + f
            const auto p = static_cast<std::size_t>(b.projected_pos[t]);
            const double fx = value(b.fx_at[p]), df = value(b.df_at[p]);
            add(b.term_row[p], b.accrual_at[t], b.residuals[t] * fx * df * b.term_scale[p]);
        }
        for (std::size_t h = 0; h < b.head_at.size(); ++h) {
            add(b.head_row[h], b.head_at[h], b.head_residuals[h] * b.head_scale[h]);
        }
        for (const CompiledBatch::Ndf& d : b.ndfs) {
            const double s_t = d.fx >= 0 ? value(d.fx) : d.s_t;
            add(d.row, d.df, d.a * (s_t - d.strike) * d.spot);
            if (d.fx >= 0) add(d.row, d.fx, d.a * value(d.df) * d.spot);
        }
        return all_finite(jac);
    }

    nb::object market_, currency_;
    nb::list instruments_;
    std::vector<std::int64_t> kinds_, all_;
    std::vector<SolvedCurve> curves_;
    std::vector<Block> blocks_;
    std::size_t n_unknowns_ = 0;

    // The Jacobian: the compiled batch (once), the curves' terms in it, the bump.
    bool compiled_ = false, prepared_ = false, linear_ = false;
    double bump_ = 0.0;
    CompiledBatch batch_;
    std::vector<CurveTerms> curve_terms_;
    std::vector<double> dg_, v_, flows_;  // scratch; dg_ kept on linear curves
};

}  // namespace

void bind_solver(nb::module_& m) {
    nb::class_<GroupSolver>(m, "GroupSolver",
                            "One curve-group solve of the curve builder (_solver._GroupSolve).")
        .def(nb::init<nb::handle, nb::handle, nb::list, nb::list, DaysIn>(), nb::arg("market"),
             nb::arg("currency"), nb::arg("curves"), nb::arg("instruments"), nb::arg("kinds"),
             "curves: (curve, fixed forwards, start, stop, tau of the first knot, constant "
             "log df at the first knot or None) per curve, in the unknowns' order; "
             "instruments and kinds as market_valuate_many takes them.")
        .def("set_blocks", &GroupSolver::set_blocks, nb::arg("blocks"),
             "The sweeps' date blocks, ascending: (unknowns, instruments) index arrays.")
        .def("sweep", &GroupSolver::sweep, nb::arg("y0"), nb::arg("max_sweeps"), nb::arg("tol"),
             nb::arg("sweep"), nb::arg("block"),
             "The sweeps by block Newton from that block: (status, y, sweep, block); status "
             "0 within tol, 1 above it, 2 Newton gave up on that block, 3 a valuation raised.")
        .def("joint", &GroupSolver::joint, nb::arg("y0"), nb::arg("linear"), nb::arg("bump"),
             "Every unknown at once by Newton with the analytic Jacobian (instruments: "
             "Swaps of that exact type, kind 1, and NDFs); None if not solved.")
        .def("residuals", &GroupSolver::residuals_at, nb::arg("y"),
             "The residuals at y, None when the valuation raised.")
        .def("jacobian", &GroupSolver::jacobian_at, nb::arg("y"), nb::arg("linear"),
             nb::arg("bump"), "The analytic Jacobian at y, None when not compiled.")
        .def(
            "apply",
            [](GroupSolver& s, DoublesIn y) { s.apply(view(y)); }, nb::arg("y"),
            "Set the unknowns on the curves' cores.");
}

}  // namespace fts::py
