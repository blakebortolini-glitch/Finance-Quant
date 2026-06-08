"""Parquet-based local cache, keyed by (ticker, date), TTL = 1 trading day.

Avoids refetching the same payload within a trading day. Each logical payload
(prices, fundamentals, options, macro) is stored as its own parquet file under
``CACHE_DIR`` with a key embedding ticker and as-of date. Because the file name
embeds the as-of date, a new trading day naturally misses the previous day's
file — the TTL is enforced by the date in the key plus an mtime sanity check.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from quant_agent.config import CACHE_DIR, CACHE_TTL_TRADING_DAYS

_SECONDS_PER_DAY = 86_400


def cache_path(key: str, as_of: date) -> Path:
    """Return the parquet path for a logical key on a given date."""
    return CACHE_DIR / f"{key}_{as_of.isoformat()}.parquet"


def get(key: str, as_of: date, ttl_days: int = CACHE_TTL_TRADING_DAYS) -> pd.DataFrame | None:
    """Return cached DataFrame if present and within TTL, else None.

    Args:
        key: logical payload key, e.g. ``"AAPL_prices"``.
        as_of: run date the payload was fetched for.
        ttl_days: max age in days before the entry is treated as stale.

    Returns:
        The cached DataFrame, or None on miss / stale / unreadable.
    """
    path = cache_path(key, as_of)
    if not path.exists():
        return None

    age_days = (time.time() - path.stat().st_mtime) / _SECONDS_PER_DAY
    if age_days > ttl_days:
        logger.debug(f"Cache stale ({age_days:.1f}d): {path.name}")
        return None

    try:
        df = pd.read_parquet(path)
        logger.debug(f"Cache hit: {path.name}")
        return df
    except Exception as exc:  # corrupt/partial file — treat as a miss
        logger.warning(f"Cache read failed for {path.name}: {exc}")
        return None


def put(key: str, as_of: date, df: pd.DataFrame) -> None:
    """Write a DataFrame to the parquet cache (creating CACHE_DIR if needed)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(key, as_of)
    try:
        df.to_parquet(path)
        logger.debug(f"Cache write: {path.name} ({len(df)} rows)")
    except Exception as exc:
        logger.warning(f"Cache write failed for {path.name}: {exc}")


__all__ = ["cache_path", "get", "put", "CACHE_DIR", "CACHE_TTL_TRADING_DAYS"]
