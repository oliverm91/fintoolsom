#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>

#include "schedule.hpp"

// Day counts (fintoolsom.dates.date_counts) and time fractions
// (fintoolsom.dates.time_fractions) on date ordinals, with the same rules and the same
// floating-point operations as the Python classes.
namespace fts::dates {

struct Ymd {
    int y;
    int m;
    int d;
};

inline Ymd split(Ordinal o) {
    const auto ymd = ymd_of(o);
    return {static_cast<int>(ymd.year()), static_cast<int>(static_cast<unsigned>(ymd.month())),
            static_cast<int>(static_cast<unsigned>(ymd.day()))};
}

inline bool is_leap(int year) { return std::chrono::year{year}.is_leap(); }

// calendar.monthrange(year, month)[1].
inline int month_days(int year, int month) {
    return static_cast<int>(static_cast<unsigned>(
        std::chrono::year_month_day_last{std::chrono::year{year} /
                                         std::chrono::month{static_cast<unsigned>(month)} /
                                         std::chrono::last}
            .day()));
}

inline std::int64_t thirty_360(const Ymd& a, int d1, const Ymd& b, int d2) {
    return 360LL * (b.y - a.y) + 30LL * (b.m - a.m) + (d2 - d1);
}

// The DayCountConventionBase subclasses, by kind.
enum class DayCount { Actual = 0, Days30A = 1, Days30U = 2, Days30E = 3, Days30EISDA = 4 };

inline std::int64_t day_count(DayCount kind, Ordinal start, Ordinal end) {
    if (kind == DayCount::Actual) return end - start;
    const Ymd a = split(start), b = split(end);
    int d1 = a.d, d2 = b.d;
    switch (kind) {
        case DayCount::Days30A:
            d1 = std::min(d1, 30);
            if (d1 > 29) d2 = std::min(d2, 30);
            break;
        case DayCount::Days30U: {
            const bool start_feb = a.m == 2 && d1 == month_days(a.y, a.m);
            const bool end_feb = b.m == 2 && d2 == month_days(b.y, b.m);
            if (start_feb && end_feb) d2 = 30;
            if (start_feb) d1 = 30;
            if (d2 == 31 && d1 >= 30) d2 = 30;
            if (d1 == 31) d1 = 30;
            break;
        }
        case DayCount::Days30E:
            d1 = std::min(d1, 30);
            d2 = std::min(d2, 30);
            break;
        case DayCount::Days30EISDA:
            if (d1 == month_days(a.y, a.m)) d1 = 30;
            if (d2 == month_days(b.y, b.m) && b.m != 2) d2 = 30;
            break;
        default:
            throw std::invalid_argument("Unknown day count convention.");
    }
    return thirty_360(a, d1, b, d2);
}

// _feb_29_count: 29 Februaries in (start, end].
inline std::int64_t feb_29_count(Ordinal start, Ordinal end) {
    std::int64_t n = 0;
    for (int y = year_of(start), last = year_of(end); y <= last; ++y) {
        if (!is_leap(y)) continue;
        const Ordinal f = ordinal_of(std::chrono::year{y} / 2 / 29);
        if (start < f && f <= end) ++n;
    }
    return n;
}

// The TimeFractionBase subclasses, by kind.
enum class TimeFraction {
    Thirty360BondBasis = 0,
    Thirty360E = 1,
    Thirty360EISDA = 2,
    Actual365 = 3,
    Actual360 = 4,
    Actual364 = 5,
    Actual30 = 6,
    Actual365_25 = 7,
    Actual365Long = 8,
    NL360 = 9,
    NL365 = 10,
    ActualActualISDA = 11,
    ActualActualICMA = 12,
};

// get_day_count_factor. `frequency` for Actual365Long and ActualActualICMA,
// [period_start, period_end] for ActualActualICMA.
inline double time_fraction(TimeFraction kind, Ordinal start, Ordinal end,
                            std::int64_t frequency, Ordinal period_start, Ordinal period_end) {
    const auto days = static_cast<double>(end - start);
    switch (kind) {
        case TimeFraction::Thirty360BondBasis: {
            const Ymd a = split(start), b = split(end);
            const int d1 = std::min(a.d, 30);
            const int d2 = b.d == 31 && d1 > 29 ? 30 : b.d;
            return static_cast<double>(thirty_360(a, d1, b, d2)) / 360.0;
        }
        case TimeFraction::Thirty360E: {
            const Ymd a = split(start), b = split(end);
            return static_cast<double>(thirty_360(a, std::min(a.d, 30), b, std::min(b.d, 30))) /
                   360.0;
        }
        case TimeFraction::Thirty360EISDA: {
            const Ymd a = split(start), b = split(end);
            const bool a_feb = a.m == 2 && a.d == month_days(a.y, a.m);
            const bool b_feb = b.m == 2 && b.d == month_days(b.y, b.m);
            const int d1 = a_feb || a.d == 31 ? 30 : a.d;
            const int d2 = b_feb || b.d == 31 ? 30 : b.d;
            return static_cast<double>(thirty_360(a, d1, b, d2)) / 360.0;
        }
        case TimeFraction::Actual365:
            return days / 365.0;
        case TimeFraction::Actual360:
            return days / 360.0;
        case TimeFraction::Actual364:
            return days / 364.0;
        case TimeFraction::Actual30:
            return days / 30.0;
        case TimeFraction::Actual365_25:
            return days / 365.25;
        case TimeFraction::Actual365Long: {
            const bool long_year = frequency == 1 ? feb_29_count(start, end) != 0
                                                  : is_leap(year_of(end));
            return days / (long_year ? 366.0 : 365.0);
        }
        case TimeFraction::NL360:
            return static_cast<double>(end - start - feb_29_count(start, end)) / 360.0;
        case TimeFraction::NL365:
            return static_cast<double>(end - start - feb_29_count(start, end)) / 365.0;
        case TimeFraction::ActualActualISDA: {
            double factor = 0.0;
            for (Ordinal t = start; t < end;) {
                const int y = year_of(t);
                const Ordinal next = std::min(ordinal_of(std::chrono::year{y + 1} / 1 / 1), end);
                factor += static_cast<double>(next - t) / (is_leap(y) ? 366.0 : 365.0);
                t = next;
            }
            return factor;
        }
        case TimeFraction::ActualActualICMA: {
            const std::int64_t denominator = frequency * (period_end - period_start);
            if (denominator == 0) throw std::domain_error("division by zero");
            return days / static_cast<double>(denominator);
        }
    }
    throw std::invalid_argument("Unknown time fraction.");
}

}  // namespace fts::dates
