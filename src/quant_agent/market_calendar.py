"""NYSE trading-day calendar — weekday + US market holidays.

Used by the morning scheduler so the agent only runs on days the market is
actually open. Holidays are computed per year (no external dependency), with
the NYSE observed-day convention: a holiday on Saturday is observed the
preceding Friday, on Sunday the following Monday (New Year's is not pulled
back across the year boundary).
"""

from __future__ import annotations

from datetime import date, timedelta


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th ``weekday`` (Mon=0) of a month, e.g. 3rd Monday of January."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last ``weekday`` (Mon=0) of a month, e.g. last Monday of May."""
    d = date(year, month, 28)
    while d.month == month:
        nxt = d + timedelta(days=1)
        if nxt.month != month:
            break
        d = nxt
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Easter Sunday via the Anonymous Gregorian (Meeus/Jones/Butcher) algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date, *, pull_back: bool = True) -> date:
    """Apply the NYSE weekend-observance rule to a fixed-date holiday."""
    if d.weekday() == 5:  # Saturday -> preceding Friday
        return d - timedelta(days=1) if pull_back else d
    if d.weekday() == 6:  # Sunday -> following Monday
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[date]:
    """The set of NYSE market holidays for a given calendar year."""
    good_friday = _easter(year) - timedelta(days=2)
    holidays = {
        # New Year's: only roll Sunday -> Monday (don't pull back to prior Dec).
        _observed(date(year, 1, 1), pull_back=False),
        _nth_weekday(year, 1, 0, 3),       # MLK Jr. Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),       # Washington's Birthday (3rd Mon Feb)
        good_friday,
        _last_weekday(year, 5, 0),         # Memorial Day (last Mon May)
        _observed(date(year, 6, 19)),      # Juneteenth (since 2021)
        _observed(date(year, 7, 4)),       # Independence Day
        _nth_weekday(year, 9, 0, 1),       # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),      # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),     # Christmas
    }
    if year < 2022:
        holidays.discard(_observed(date(year, 6, 19)))
    return holidays


def is_trading_day(d: date | None = None) -> bool:
    """True if ``d`` (default today) is a weekday and not an NYSE holiday."""
    d = d or date.today()
    if d.weekday() >= 5:  # Saturday/Sunday
        return False
    return d not in nyse_holidays(d.year)


__all__ = ["is_trading_day", "nyse_holidays"]


if __name__ == "__main__":  # pragma: no cover
    today = date.today()
    print(f"{today} trading day? {is_trading_day(today)}")
    print(f"2026 NYSE holidays: {sorted(nyse_holidays(2026))}")
