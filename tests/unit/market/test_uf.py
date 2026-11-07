"""Chilean UF / CPI mechanics (:mod:`fintoolsom.market.uf`) on real SII data.
"""

import math
from datetime import date
import pytest
from fintoolsom.market import Currency, Market, uf
from fintoolsom.market.currencies import CurrencyPair



# UF on the 9th of each month, CLP per UF (the reajuste boundaries).
UF_9 = {
    date(2026, 1, 9): 39762.52,
    date(2026, 2, 9): 39682.99,
    date(2026, 3, 9): 39841.72,
    date(2026, 4, 9): 39841.72,
    date(2026, 5, 9): 40240.14,
    date(2026, 6, 9): 40763.26,
    date(2026, 7, 9): 40844.79,
}


# Official monthly CPI (IPC) variation, as a decimal.
IPC_DEC_2025 = -0.002  # -0.2% (real deflation)


IPC_MAR_2026 = 0.010   # +1.0%


IPC_APR_2026 = 0.013   # +1.3%


# Real published daily UF over the 9-Apr -> 9-May 2026 reajuste period (IPC Mar = +1.0%).
UF_APR_MAY_2026 = {
    date(2026, 4, 13): 39894.61,
    date(2026, 4, 20): 39987.35,
    date(2026, 4, 30): 40120.20,
    date(2026, 5, 1): 40133.50,
    date(2026, 5, 9): 40240.14,
}


def test_inflation_from_real_uf_levels():
    # monthly_inflation(levels, year, m) = UF(9 of m+2) / UF(9 of m+1) - 1, which recovers the
    # official monthly IPC from the real SII UF series.
    levels = dict(UF_9)
    assert round(uf.monthly_inflation(levels, 2026, 3), 3) == IPC_MAR_2026  # +1.0%
    assert round(uf.monthly_inflation(levels, 2026, 4), 3) == IPC_APR_2026  # +1.3%


def test_inflation_reads_future_boundary_level():
    # The later boundary (9-Jun) is a future date relative to an early-May "today";
    # once published (loaded) monthly_inflation still resolves against it.
    levels = dict(UF_9)
    assert levels[date(2026, 6, 9)] == 40763.26  # future print is held
    assert round(uf.monthly_inflation(levels, 2026, 4), 3) == IPC_APR_2026


def test_inflation_rolls_year_for_december():
    # Real SII: Dec 2025 IPC = UF(9-Feb-2026) / UF(9-Jan-2026) - 1 = -0.2%, which
    # exercises the year rollover (m = Dec 2025 -> m+1 = Jan, m+2 = Feb 2026).
    levels = dict(UF_9)
    assert round(uf.monthly_inflation(levels, 2025, 12), 3) == IPC_DEC_2025


def test_inflation_missing_boundary_raises():
    levels = {date(2026, 4, 9): UF_9[date(2026, 4, 9)]}
    with pytest.raises(ValueError):
        uf.monthly_inflation(levels, 2026, 3)  # 9-May-2026 not loaded yet


def test_last_known_date_before_cpi_release():
    # 8-Jun-2026 is a business day (no Chilean holiday), so the release proxy is the
    # 8th. On the 3rd (before release) only levels up to the 9th of the current
    # month are known.
    levels = {date(2026, 2, 9): UF_9[date(2026, 2, 9)]}
    assert uf.last_published_date(date(2026, 6, 3)) == date(2026, 6, 9)


def test_last_known_date_after_cpi_release():
    # On/after the release, levels are known up to the 9th of next month.
    levels = {date(2026, 2, 9): UF_9[date(2026, 2, 9)]}
    assert uf.last_published_date(date(2026, 6, 25)) == date(2026, 7, 9)


def test_last_known_date_rolls_year_in_december():
    levels = {date(2026, 2, 9): UF_9[date(2026, 2, 9)]}
    assert uf.last_published_date(date(2026, 12, 20)) == date(2027, 1, 9)


def test_last_known_date_uses_chilean_calendar():
    # 8-Dec-2026 (Inmaculada Concepción) is a Chilean holiday, so the CPI-release
    # proxy (preceding business day of the 8th) is 7-Dec-2026 (Monday). On the 7th
    # we are already on/after release → horizon is next month. A weekends-only
    # calendar would place release on the 8th and keep us in the current month, so
    # this asserts the CL calendar (from CLP → Locality.CL) is in effect.
    levels = {date(2026, 2, 9): UF_9[date(2026, 2, 9)]}
    assert uf.last_published_date(date(2026, 12, 7)) == date(2027, 1, 9)


def test_extend_with_cpi_matches_real_published_uf():
    # From base UF(9-Apr-2026) with the March 2026 IPC (+1.0%), the projected daily
    # UF must reproduce SII's published levels to the cent across 10-Apr..9-May.
    levels = {date(2026, 4, 9): UF_9[date(2026, 4, 9)]}
    levels = uf.extend_with_cpi(levels, IPC_MAR_2026)
    for d, real_uf in UF_APR_MAY_2026.items():
        assert levels[d] == real_uf
    # Worked-example form: UF(13-Apr) = round(UF(9-Apr) * (1.01)^(4/30), 2).
    assert levels[date(2026, 4, 13)] == round(
        UF_9[date(2026, 4, 9)] * 1.01 ** (4 / 30), 2
    )


def test_extend_with_cpi_rounds_to_two_decimals():
    levels = {date(2026, 4, 9): UF_9[date(2026, 4, 9)]}
    levels = uf.extend_with_cpi(levels, IPC_MAR_2026)
    for level in levels.values():
        assert round(level, 2) == level  # every projected level is a 2-dp value


def test_extend_with_cpi_roundtrips_through_get_inflation():
    # Extending base 9-Apr with the March IPC fills through 9-May; March inflation
    # then reads back (UF's 2-dp rounding keeps it within a cent of the input CPI).
    levels = {date(2026, 4, 9): UF_9[date(2026, 4, 9)]}
    levels = uf.extend_with_cpi(levels, IPC_MAR_2026)
    assert round(uf.monthly_inflation(levels, 2026, 3), 3) == IPC_MAR_2026


def test_extend_with_cpi_does_not_overwrite_known_levels():
    sentinel = 99999.99
    levels = {date(2026, 4, 9): UF_9[date(2026, 4, 9)], date(2026, 4, 13): sentinel}
    levels = uf.extend_with_cpi(levels, IPC_MAR_2026)
    assert levels[date(2026, 4, 13)] == sentinel


def test_extend_with_cpi_requires_a_ninth_base():
    levels = {date(2026, 4, 10): UF_9[date(2026, 4, 9)]}
    with pytest.raises(ValueError):
        levels = uf.extend_with_cpi(levels, IPC_MAR_2026)


def test_extend_with_cpi_rejects_non_ninth_base():
    levels = {date(2026, 4, 9): UF_9[date(2026, 4, 9)]}
    with pytest.raises(ValueError):
        levels = uf.extend_with_cpi(levels, IPC_MAR_2026, base_date=date(2026, 4, 10))


# Extra real published UF levels inside the 9-Apr -> 9-May 2026 reajuste period.
UF_21_APR_2026 = 40000.61


UF_8_MAY_2026 = 40226.79


UF_20_MAY_2026 = 40424.99


def test_uf_convention_reajuste_matches_published_level():
    # round(UF(9-Apr) * 1.01^(4/30), 2) == real UF(13-Apr-2026).
    assert uf.UFConvention.reajuste(UF_9[date(2026, 4, 9)], IPC_MAR_2026, 4, 30) == 39894.61


def test_uf_convention_implied_monthly_cpi_full_period():
    cpi = uf.UFConvention.implied_monthly_cpi(UF_9[date(2026, 4, 9)], UF_9[date(2026, 5, 9)])
    assert round(cpi, 3) == IPC_MAR_2026


def test_implied_cpi_between_ninth_to_mid_month():
    # 9-Apr (base) to 21-Apr — partial period within March reajuste -> March IPC.
    cpi = uf.UFConvention.implied_monthly_cpi_between(
        date(2026, 4, 9), UF_9[date(2026, 4, 9)], date(2026, 4, 21), UF_21_APR_2026
    )
    assert round(cpi, 3) == IPC_MAR_2026


def test_implied_cpi_between_ninth_and_eighth_of_next_month():
    # 9-Apr and 8-May — still one reajuste period (boundary is the 9th) -> March IPC.
    cpi = uf.UFConvention.implied_monthly_cpi_between(
        date(2026, 4, 9), UF_9[date(2026, 4, 9)], date(2026, 5, 8), UF_8_MAY_2026
    )
    assert round(cpi, 3) == IPC_MAR_2026


def test_implied_cpi_between_full_period_equals_simple_ratio():
    # 9-Apr to 9-May is exactly one period: scaling exponent is 1, so it reduces
    # to end/start - 1.
    cpi = uf.UFConvention.implied_monthly_cpi_between(
        date(2026, 4, 9), UF_9[date(2026, 4, 9)], date(2026, 5, 9), UF_9[date(2026, 5, 9)]
    )
    assert math.isclose(
        cpi, UF_9[date(2026, 5, 9)] / UF_9[date(2026, 4, 9)] - 1.0, rel_tol=1e-12
    )


def test_implied_cpi_between_raises_across_reajuste_boundary():
    # 9-Apr (base) and 20-May cross the 9-May boundary: compounds two months' CPI.
    with pytest.raises(ValueError):
        uf.UFConvention.implied_monthly_cpi_between(
            date(2026, 4, 9), UF_9[date(2026, 4, 9)], date(2026, 5, 20), UF_20_MAY_2026
        )


def test_implied_cpi_between_requires_distinct_dates():
    with pytest.raises(ValueError):
        uf.UFConvention.implied_monthly_cpi_between(
            date(2026, 4, 9), UF_9[date(2026, 4, 9)], date(2026, 4, 9), UF_9[date(2026, 4, 9)]
        )


def test_implied_cpi_between_requires_base_on_ninth():
    with pytest.raises(ValueError, match="base_9 must be the 9th"):
        uf.UFConvention.implied_monthly_cpi_between(
            date(2026, 4, 13), 39894.61, date(2026, 4, 21), UF_21_APR_2026
        )


def test_extend_with_cpi_returns_a_copy():
    levels = {date(2026, 4, 9): UF_9[date(2026, 4, 9)]}
    filled = uf.extend_with_cpi(levels, IPC_MAR_2026)
    assert len(levels) == 1 and len(filled) == 31


def test_extend_with_cpi_missing_base_raises():
    with pytest.raises(KeyError):
        uf.extend_with_cpi({date(2026, 3, 9): 1.0}, IPC_MAR_2026, base_date=date(2026, 4, 9))


def test_uf_rate_data_loads_clf_like_any_currency():
    clfclp = CurrencyPair(Currency.CLF, Currency.CLP)
    market = Market(t=date(2026, 4, 9), fx_history={clfclp: uf.uf_rate_data(UF_APR_MAY_2026)})
    assert market.get_fx_rate(date(2026, 4, 13), clfclp).value == 39894.61
    assert market.get_fx_rate(date(2026, 4, 13), clfclp.invert()).value == pytest.approx(1 / 39894.61)
    assert market.get_fx_known_until(clfclp) == date(2026, 5, 9)
