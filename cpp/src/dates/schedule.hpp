#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

// Business-day calendars, date adjustment conventions and schedule generation:
// fintoolsom.dates (Calendar, the *Convention classes, Tenor, ScheduleGenerator) with
// the same rules, on date ordinals (date.toordinal()).
namespace fts::dates {

using Ordinal = std::int64_t;

inline constexpr Ordinal kOrdinal1970 = 719163;  // date(1970, 1, 1).toordinal()

inline std::chrono::year_month_day ymd_of(Ordinal o) {
    return std::chrono::year_month_day{
        std::chrono::sys_days{std::chrono::days{o - kOrdinal1970}}};
}

inline Ordinal ordinal_of(std::chrono::year_month_day ymd) {
    return std::chrono::sys_days{ymd}.time_since_epoch().count() + kOrdinal1970;
}

inline int year_of(Ordinal o) { return static_cast<int>(ymd_of(o).year()); }
inline unsigned month_of(Ordinal o) { return static_cast<unsigned>(ymd_of(o).month()); }

// date.weekday(): Monday is 0 (date(1, 1, 1), ordinal 1, is a Monday).
inline int weekday_of(Ordinal o) { return static_cast<int>(((o + 6) % 7 + 7) % 7); }

// date + relativedelta(months=months): the same day in the target month, clamped to
// its last day.
inline Ordinal add_months(Ordinal o, std::int64_t months) {
    const auto ymd = ymd_of(o);
    const std::int64_t index = static_cast<int>(ymd.year()) * 12LL +
                               (static_cast<unsigned>(ymd.month()) - 1) + months;
    const std::int64_t q = index >= 0 ? index / 12 : -((-index + 11) / 12);
    const std::chrono::year year{static_cast<int>(q)};
    const std::chrono::month month{static_cast<unsigned>(index - q * 12 + 1)};
    const auto last = std::chrono::year_month_day_last{year / month / std::chrono::last}.day();
    const auto day = std::min(ymd.day(), last);
    return ordinal_of(std::chrono::year_month_day{year, month, day});
}

// One calendar's days over a window of whole years, from `load(first, last)` (years
// inclusive): per day, bit 0 a holiday of the `holidays` object, bit 1 one of its
// weekend days, bit 2 one of its weekend working days. `weekend_mask` holds
// Calendar._weekend_weekdays. The first date asked for loads [year - 5, year + 65] in
// one call; a date outside the window extends it by at least 20 years, in one call.
class CalendarCore {
public:
    using Loader = std::function<std::vector<std::uint8_t>(int first_year, int last_year)>;

    static constexpr int kBefore = 5, kAfter = 65, kExtend = 20;

    CalendarCore(Loader load, unsigned weekend_mask)
        : load_(std::move(load)), weekend_mask_(weekend_mask) {}

    // Calendar.is_holiday: a holiday, or a weekday in Calendar._weekend_weekdays.
    [[nodiscard]] bool is_holiday(Ordinal o) const {
        return (flags(o) & 1) != 0 || ((weekend_mask_ >> weekday_of(o)) & 1) != 0;
    }

    // HolidayBase.is_working_day.
    [[nodiscard]] bool is_working_day(Ordinal o) const {
        const std::uint8_t f = flags(o);
        return (f & 2) != 0 ? (f & 4) != 0 : (f & 1) == 0;
    }

    // Calendar.add_business_days: HolidayBase.get_nth_working_day.
    [[nodiscard]] Ordinal add_business_days(Ordinal o, std::int64_t n) const {
        const Ordinal step = n >= 0 ? 1 : -1;
        const std::int64_t count = n == 0 ? 1 : (n < 0 ? -n : n);
        for (std::int64_t i = 0; i < count; ++i) {
            if (n != 0) o += step;
            while (!is_working_day(o)) o += step;
        }
        return o;
    }

    // Years loaded, for tests: [first_year, last_year], or (0, -1) before any.
    [[nodiscard]] std::pair<int, int> years() const { return {first_year_, last_year_}; }

private:
    [[nodiscard]] std::uint8_t flags(Ordinal o) const {
        if (o < kMin || o >= kEnd) throw std::out_of_range("CalendarCore: date out of range.");
        if (o < first_ || o >= first_ + static_cast<Ordinal>(days_.size())) extend(year_of(o));
        return days_[static_cast<std::size_t>(o - first_)];
    }

    static Ordinal jan1(int year) { return ordinal_of(std::chrono::year{year} / 1 / 1); }
    // The years a core holds, 1..9998: populating a year's neighbours needs 9999.
    static constexpr Ordinal kMin = 1, kEnd = 3651695;  // date(9999, 1, 1).toordinal()

    std::vector<std::uint8_t> load(int first_year, int last_year) const {
        first_year = std::max(first_year, 1);
        last_year = std::min(last_year, 9998);
        std::vector<std::uint8_t> days = load_(first_year, last_year);
        if (days.size() != static_cast<std::size_t>(jan1(last_year + 1) - jan1(first_year))) {
            throw std::runtime_error("CalendarCore: the loader returned a wrong number of days.");
        }
        return days;
    }

    void extend(int year) const {
        if (days_.empty()) {
            const int first = std::max(year - kBefore, 1), last = std::min(year + kAfter, 9998);
            days_ = load(first, last);
            first_year_ = first;
            last_year_ = last;
            first_ = jan1(first);
        } else if (year < first_year_) {
            const int first = std::max(std::min(year, first_year_ - kExtend), 1);
            std::vector<std::uint8_t> before = load(first, first_year_ - 1);
            before.insert(before.end(), days_.begin(), days_.end());
            days_ = std::move(before);
            first_year_ = first;
            first_ = jan1(first);
        } else {
            const int last = std::min(std::max(year, last_year_ + kExtend), 9998);
            const std::vector<std::uint8_t> after = load(last_year_ + 1, last);
            days_.insert(days_.end(), after.begin(), after.end());
            last_year_ = last;
        }
    }

    Loader load_;
    unsigned weekend_mask_;
    mutable std::vector<std::uint8_t> days_;
    mutable Ordinal first_ = 0;
    mutable int first_year_ = 0, last_year_ = -1;
};

enum class Adjustment { Following = 0, ModifiedFollowing = 1, Preceding = 2, ModifiedPreceding = 3 };

struct Convention {
    Adjustment kind;
    const CalendarCore* calendar;

    [[nodiscard]] Ordinal following(Ordinal o) const {
        return calendar->is_holiday(o) ? calendar->add_business_days(o, 1) : o;
    }
    [[nodiscard]] Ordinal preceding(Ordinal o) const {
        return calendar->is_holiday(o) ? calendar->add_business_days(o, -1) : o;
    }
    // The adjust method of each convention class.
    [[nodiscard]] Ordinal adjust(Ordinal o) const {
        switch (kind) {
            case Adjustment::Following:
                return following(o);
            case Adjustment::Preceding:
                return preceding(o);
            case Adjustment::ModifiedFollowing: {
                const Ordinal d = following(o);
                return month_of(d) == month_of(o) ? d : preceding(o);
            }
            case Adjustment::ModifiedPreceding: {
                const Ordinal d = preceding(o);
                return month_of(d) == month_of(o) ? d : following(o);
            }
        }
        throw std::invalid_argument("Unknown adjustment convention.");
    }
};

// Tenor: a count of business days ('d', on its convention's calendar), weeks, months or
// years.
struct Tenor {
    char unit;  // 'd', 'w', 'm' or 'y'
    std::int64_t value;
    const CalendarCore* calendar;  // for 'd'

    // Tenor.get_unadjusted_maturity(t, forward, periods).
    [[nodiscard]] Ordinal unadjusted(Ordinal t, bool forward, std::int64_t periods) const {
        const std::int64_t sign = forward ? 1 : -1;
        switch (unit) {
            case 'd':
                return calendar->add_business_days(t, value * periods * sign);
            case 'w':
                return t + 7 * value * sign * periods;
            case 'm':
                return add_months(t, value * sign * periods);
            case 'y':
                return add_months(t, 12 * value * sign * periods);
        }
        throw std::invalid_argument("Unknown tenor unit.");
    }
};

// ScheduleGenerator.generate_schedule, or nullopt where the Python one raises because
// rolling never reaches the end (the caller reruns it in Python for its error).
inline std::optional<std::vector<Ordinal>> generate_schedule(
    Ordinal start, std::optional<Tenor> maturity, const Tenor& frequency, const Convention& adj,
    const Convention& maturity_adj, std::optional<Ordinal> end_date, bool stub_first,
    bool long_stub) {
    Ordinal umat, adj_mat;
    if (!end_date) {
        umat = maturity->unadjusted(start, true, 1);
        adj_mat = maturity_adj.adjust(umat);
    } else {
        umat = *end_date;
        adj_mat = *end_date;
        if (maturity) {
            const Ordinal tenor_umat = maturity->unadjusted(start, true, 1);
            if (maturity_adj.adjust(tenor_umat) == *end_date) umat = tenor_umat;
        }
    }
    // 10 + int((umat - start_date).days / 30)
    const auto max_iterations =
        10 + static_cast<std::int64_t>(static_cast<double>(umat - start) / 30.0);

    bool failed = false;
    const auto roll = [&](Ordinal anchor, Ordinal stop, bool forward) {
        std::vector<Ordinal> dates{anchor};
        std::int64_t periods = 0;
        while (forward ? dates.back() < stop : dates.back() > stop) {
            ++periods;
            dates.push_back(frequency.unadjusted(anchor, forward, periods));
            if (periods > max_iterations) {
                failed = true;
                break;
            }
        }
        return dates;
    };

    std::vector<Ordinal> u = roll(start, umat, true);
    if (failed) return std::nullopt;
    if (stub_first && u.back() != umat) {
        u = roll(umat, start, false);
        if (failed) return std::nullopt;
        std::sort(u.begin(), u.end());
    }
    if (u.front() != start || u.back() != umat) {
        if (stub_first) {
            u.erase(u.begin());
            u.insert(u.begin(), start);
            if (u.size() > 2 && long_stub) u.erase(u.begin() + 1);
        } else {
            u.back() = umat;
            if (u.size() > 2 && long_stub) u.erase(u.end() - 2);
        }
    }
    std::vector<Ordinal> out;
    if (u.size() >= 2) out.push_back(u.front());  // u_schedule[:-1][:1]
    for (std::size_t i = 1; i + 1 < u.size(); ++i) {
        const Ordinal d = adj.adjust(u[i]);
        if (out.back() < d && d < adj_mat) out.push_back(d);
    }
    out.push_back(adj_mat);
    return out;
}

}  // namespace fts::dates
