#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace fts {

// Discount factors of a curve at date ordinals: fills out[i] for ordinals[i].
using DfSource =
    std::function<void(std::span<const std::int64_t> ordinals, std::span<double> out)>;

// Discount factors of flows, as Market._dfs_at_ordinals: one curve, or a * b / c for
// collateral in another currency.
struct Discount {
    const DfSource* a = nullptr;
    const DfSource* b = nullptr;
    const DfSource* c = nullptr;

    void operator()(std::span<const std::int64_t> ordinals, std::span<double> out) const {
        (*a)(ordinals, out);
        if (b == nullptr) return;
        std::vector<double> db(ordinals.size()), dc(ordinals.size());
        (*b)(ordinals, db);
        (*c)(ordinals, dc);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = out[i] * db[i] / dc[i];
    }

    [[nodiscard]] double at(std::int64_t ordinal) const {
        double df = 0.0;
        (*this)(std::span(&ordinal, 1), std::span(&df, 1));
        return df;
    }
};

}  // namespace fts
