# fintoolsom.dates

Calendars, business-day adjustments, day counts, tenors and schedules. Pure date
arithmetic: nothing here reads market data.

## Modules

| Module | Contents |
|---|---|
| `calendars.py` | `Calendar` (country holidays from `holidays`, plus custom ones), `get_calendar(country)` |
| `holidays.py` | Holiday rules: `MonthDayRule`, `OrdinalWeekWeekdayRule`, `LastWeekWeekdayRule`, `FridayEasterRule`, `MondayEasterRule`, `ConsecutiveHolidaySandwichRule` |
| `adjustments.py` | `FollowingConvention`, `ModifiedFollowingConvention`, `PrecedingConvention`, `ModifiedPrecedingConvention` |
| `date_counts.py` | Day counts used by `Rate`: `ActualDayCountConvention`, `Days30A/U/E/EISDADayCountConvention` |
| `time_fractions.py` | `TF_*` year fractions used by swap coupons: Act/360, Act/365, 30/360 variants, Act/Act ISDA and ICMA, NL/365, ... |
| `term.py` | `Term(value, TermUnit, adj_convention)`: a tenor that adjusts its own end date |
| `schedules.py` | `Tenor("3M")` and `ScheduleGenerator` for coupon schedules with stubs |

## Example

```python
from datetime import date

from fintoolsom.dates import (
    ActualDayCountConvention, ModifiedFollowingConvention, ScheduleGenerator, Term, TermUnit, get_calendar,
)

cl = get_calendar("CL")
cl.is_holiday(date(2024, 9, 18))             # True: Independence Day
cl.add_business_days(date(2024, 9, 17), 1)   # 2024-09-23 (18-20 are holidays, then a weekend)

mf = ModifiedFollowingConvention("CL")       # a Calendar, a country code, or None (weekends only)
Term(6, TermUnit.M, mf).advance(date(2024, 3, 29))

ActualDayCountConvention.get_time_fraction(date(2024, 1, 1), date(2024, 7, 1), 360)  # 182/360

ScheduleGenerator().generate_schedule(date(2024, 1, 15), "2Y", "6M", mf)  # coupon end dates
```

## Notes

- **One calendar per country.** `get_calendar("CL")` builds the calendar once and
  returns the same object afterwards, so indexes, conventions and legs share it.
  A custom holiday added to it is seen everywhere; build a `Calendar(...)` of
  your own to keep one private.
- **Two day-count families.** `date_counts` classes are used as classes (static
  methods, passed to `RateConvention`); `time_fractions` are instances, held by
  swap legs and coupons.
- `Term` carries its own adjustment convention, so a quote's maturity and its
  schedule can roll differently.
- Indexes resolve their calendar from their currency's locality
  (`fintoolsom.market.localities`).
