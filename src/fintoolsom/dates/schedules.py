from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from dateutil.relativedelta import relativedelta

from .calendars import get_calendar
from .adjustments import AdjustmentDateConventionBase, ModifiedFollowingConvention

_def_cal = get_calendar()
_def_adj_conv = ModifiedFollowingConvention(_def_cal)


@dataclass(slots=True)
class Tenor:
    str_tenor: str
    adj_conv: AdjustmentDateConventionBase = field(default=None)

    tenor_unit: str = field(init=False)
    tenor_value: int = field(init=False)
    _rel_mapper: dict = field(init=False)
    _get_umaturity_func: Callable[[date], date] = field(init=False)

    def __post_init__(self):
        if len(self.str_tenor) < 2:
            raise ValueError(
                f"str_tenor length must be at least 2. str_tenor received was {self.str_tenor}"
            )

        if self.adj_conv is None:
            self.adj_conv = _def_adj_conv

        self.tenor_unit = self.str_tenor[-1].lower()
        self.tenor_value = int(self.str_tenor[:-1])
        self._rel_mapper = {}
        if self.tenor_unit == "d":

            def f(t: date, forward: bool = True, periods: int = 1) -> date:
                return self.adj_conv.calendar.add_business_days(
                    t, self.tenor_value * periods * (1 if forward else -1)
                )
        elif self.tenor_unit == "m":
            self._rel_mapper["kwarg"] = "months"
            self._rel_mapper["multiplier"] = 1
        elif self.tenor_unit == "y":
            self._rel_mapper["kwarg"] = "months"
            self._rel_mapper["multiplier"] = 12
        elif self.tenor_unit == "w":
            self._rel_mapper["kwarg"] = "days"
            self._rel_mapper["multiplier"] = 7
        else:
            raise ValueError(
                f"Last letter of str_tenor must be of D, W, M or Y. Got {self.tenor_unit}"
            )

        if self.tenor_unit != "d":
            step = self._rel_mapper["multiplier"] * self.tenor_value
            forward_step = relativedelta(**{self._rel_mapper["kwarg"]: step})
            backward_step = relativedelta(**{self._rel_mapper["kwarg"]: -step})

            def f(t: date, forward: bool = True, periods: int = 1) -> date:
                return t + (forward_step if forward else backward_step) * periods

        self._get_umaturity_func = f  # TEST THAT F LIVES AFTER INIT

    def get_unadjusted_maturity(
        self, t: date, forward: bool = True, periods: int = 1
    ) -> date:
        return self._get_umaturity_func(t, forward=forward, periods=periods)

    def get_adjusted_maturity(self, t: date, forward: bool = True) -> date:
        u_mat = self.get_unadjusted_maturity(t, forward=forward)
        return self.adj_conv.adjust(u_mat)


@dataclass
class ScheduleGenerator:
    @staticmethod
    def generate_schedule(
        start_date: date,
        maturity_tenor: str,
        frequency_tenor: str,
        adj_conv: AdjustmentDateConventionBase,
        maturity_adj_conv: AdjustmentDateConventionBase = None,
        end_date: date | None = None,
        stub_first: bool = True,
        long_stub: bool = True,
    ) -> list[date]:
        frequency_tenor = frequency_tenor.lower()
        accepted_frequency_tenors = ["0", 0, "0m", "0y", "1m", "3m", "6m", "1y"]
        if frequency_tenor not in accepted_frequency_tenors:
            raise ValueError(
                f"Frequency tenor accepted values are: {accepted_frequency_tenors}."
            )
        freq_tenor = Tenor(frequency_tenor, adj_conv)
        if maturity_adj_conv is None:
            maturity_adj_conv = adj_conv
        if end_date is None:
            tenor = Tenor(maturity_tenor, maturity_adj_conv)
            umat = tenor.get_unadjusted_maturity(start_date)
            adj_mat = maturity_adj_conv.adjust(umat)
        else:
            umat = end_date
            adj_mat = end_date
            if maturity_tenor:
                # An end date that is just the tenor's maturity after adjustment still
                # rolls from the unadjusted date; any other end date rolls from itself.
                tenor = Tenor(maturity_tenor, maturity_adj_conv)
                tenor_umat = tenor.get_unadjusted_maturity(start_date)
                if maturity_adj_conv.adjust(tenor_umat) == end_date:
                    umat = tenor_umat

        max_iterations = 10 + int((umat - start_date).days / 30)

        def roll(anchor: date, stop: date, forward: bool) -> list[date]:
            """`anchor` and ``anchor +/- k * frequency``, until reaching or passing `stop`.

            Every date comes from `anchor`, so a short month cannot shift the later ones.
            """
            dates = [anchor]
            periods = 0
            while dates[-1] < stop if forward else dates[-1] > stop:
                periods += 1
                dates.append(
                    freq_tenor.get_unadjusted_maturity(
                        anchor, forward=forward, periods=periods
                    )
                )
                if periods > max_iterations:
                    raise Exception(
                        f"Could not generate a schedule. Max iterations reached ({max_iterations}). start_date: {start_date}, maturity_tenor: {maturity_tenor}, frequency_tenor: {frequency_tenor}, stub_first: {stub_first}, long_stub: {long_stub}. Unadjusted Schedule list was. {dates}"
                    )
            return dates

        # A whole number of periods has no stub. Rolling from the start detects it even
        # when `umat` was clamped to a shorter month (31 Aug + 18M = 29 Feb).
        u_schedule = roll(start_date, umat, forward=True)
        if stub_first and u_schedule[-1] != umat:
            u_schedule = sorted(roll(umat, start_date, forward=False))

        if (
            u_schedule[0] != start_date or u_schedule[-1] != umat
        ):  # An exact hit: no stub.
            if stub_first:
                u_schedule = u_schedule[1:]  # removes date past starting_date
                u_schedule.insert(0, start_date)
                if (
                    len(u_schedule) > 2
                ):  # If it is 2. It should be starting_date and umat.
                    if long_stub:
                        u_schedule.pop(
                            1
                        )  # Removes second date so tenor between u_schedule[0] and u_schedule[1] is long (long_stub)
            else:
                u_schedule[-1] = umat  # replaces the date past umat by umat itself
                if (
                    len(u_schedule) > 2
                ):  # If it is 2. It should be starting_date and umat.
                    if long_stub:
                        u_schedule.pop(
                            -2
                        )  # Removes penultimate date so tenor between u_schedule[-2] and u_schedule[-1] is long (long_stub)

        # Adjust only now, so a holiday never moves the dates rolled from it. The start
        # date is kept as given.
        adj_schedule = u_schedule[:-1][:1]
        for adj_date in map(adj_conv.adjust, u_schedule[1:-1]):
            # Adjustment can land a date on its neighbour: keep one of them.
            if adj_schedule[-1] < adj_date < adj_mat:
                adj_schedule.append(adj_date)
        adj_schedule.append(adj_mat)

        return adj_schedule
