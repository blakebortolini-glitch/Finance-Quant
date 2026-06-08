"""Tests for the parquet cache round-trip and TTL behavior."""

from __future__ import annotations

import os
import time
from datetime import date

import pandas as pd
import pytest

from quant_agent.data import cache


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


def test_put_then_get_roundtrip(tmp_cache):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    cache.put("AAPL_prices", date(2026, 6, 4), df)
    out = cache.get("AAPL_prices", date(2026, 6, 4))
    assert out is not None
    pd.testing.assert_frame_equal(out, df)


def test_get_miss_returns_none(tmp_cache):
    assert cache.get("MSFT_prices", date(2026, 6, 4)) is None


def test_stale_file_is_a_miss(tmp_cache, monkeypatch):
    df = pd.DataFrame({"a": [1]})
    cache.put("AAPL_prices", date(2026, 6, 4), df)
    # Backdate the file mtime beyond the TTL.
    path = cache.cache_path("AAPL_prices", date(2026, 6, 4))
    old = time.time() - (cache.CACHE_TTL_TRADING_DAYS + 1) * 86_400
    os.utime(path, (old, old))
    assert cache.get("AAPL_prices", date(2026, 6, 4)) is None
