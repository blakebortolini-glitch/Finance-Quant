"""Tests for the NYSE trading-day calendar."""

from __future__ import annotations

from datetime import date

from quant_agent import market_calendar as mc


def test_weekends_are_not_trading_days():
    assert mc.is_trading_day(date(2026, 6, 6)) is False   # Saturday
    assert mc.is_trading_day(date(2026, 6, 7)) is False   # Sunday
    assert mc.is_trading_day(date(2026, 6, 5)) is True     # Friday


def test_known_2026_holidays_are_closed():
    holidays = {
        date(2026, 1, 1),    # New Year's Day (Thu)
        date(2026, 1, 19),   # MLK Jr. Day
        date(2026, 2, 16),   # Washington's Birthday
        date(2026, 4, 3),    # Good Friday (Easter 2026 = Apr 5)
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth (Fri)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas (Fri)
    }
    assert holidays.issubset(mc.nyse_holidays(2026))
    for h in holidays:
        assert mc.is_trading_day(h) is False


def test_independence_day_2026_observed_friday():
    # July 4, 2026 is a Saturday -> observed Friday July 3.
    assert date(2026, 7, 3) in mc.nyse_holidays(2026)
    assert mc.is_trading_day(date(2026, 7, 3)) is False


def test_regular_trading_days_open():
    assert mc.is_trading_day(date(2026, 11, 27)) is True   # day after Thanksgiving
    assert mc.is_trading_day(date(2026, 1, 2)) is True      # day after New Year's


def test_juneteenth_not_a_holiday_before_2022():
    assert date(2019, 6, 19) not in mc.nyse_holidays(2019)
    assert date(2023, 6, 19) in mc.nyse_holidays(2023)


def test_nth_and_last_weekday_helpers():
    assert mc._nth_weekday(2026, 1, 0, 3) == date(2026, 1, 19)   # 3rd Mon Jan
    assert mc._last_weekday(2026, 5, 0) == date(2026, 5, 25)      # last Mon May
    assert mc._easter(2026) == date(2026, 4, 5)
