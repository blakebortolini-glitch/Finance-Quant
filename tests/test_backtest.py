"""Tests for the walk-forward backtest harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_agent.backtest import walk_forward as wf


def _synthetic_prices(n: int, mu: float, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(shocks))
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame({"Close": close}, index=idx)


def test_analytic_upside_prob_high_for_strong_uptrend():
    # Long window so the drift estimate is tight enough to be reliably bullish.
    prices = _synthetic_prices(2500, mu=0.35, sigma=0.18, seed=1)["Close"].to_numpy()
    p = wf._analytic_upside_prob(prices, threshold=1.05, horizon_years=90 / 252)
    assert 0.0 <= p <= 1.0
    assert p > 0.5  # strong drift -> likely above +5%


def test_analytic_upside_prob_monotonic_in_drift():
    horizon = 90 / 252
    low = wf._analytic_upside_prob(
        _synthetic_prices(2500, 0.02, 0.18, seed=5)["Close"].to_numpy(), 1.05, horizon
    )
    high = wf._analytic_upside_prob(
        _synthetic_prices(2500, 0.45, 0.18, seed=5)["Close"].to_numpy(), 1.05, horizon
    )
    assert high > low


def test_walk_forward_runs_and_reports_rates(monkeypatch):
    prices = _synthetic_prices(900, mu=0.30, sigma=0.22, seed=2)
    monkeypatch.setattr(wf, "fetch_prices", lambda ticker, as_of: prices)

    res = wf.walk_forward("TEST", horizon_days=90, step=21)
    assert res.n_signals > 0
    assert 0.0 <= res.base_rate <= 1.0
    assert res.ticker == "TEST"
    if res.n_bullish:
        assert 0.0 <= res.hit_rate <= 1.0


def test_walk_forward_raises_on_short_history(monkeypatch):
    prices = _synthetic_prices(200, mu=0.1, sigma=0.2, seed=3)  # < window + horizon
    monkeypatch.setattr(wf, "fetch_prices", lambda ticker, as_of: prices)
    with pytest.raises(ValueError):
        wf.walk_forward("TEST")


def test_walk_forward_watchlist_collects_rows(monkeypatch):
    prices = _synthetic_prices(900, mu=0.2, sigma=0.2, seed=4)
    monkeypatch.setattr(wf, "fetch_prices", lambda ticker, as_of: prices)
    df = wf.walk_forward_watchlist(["A", "B"], horizon_days=90)
    assert len(df) == 2
    assert {"ticker", "hit_rate", "base_rate", "edge"}.issubset(df.columns)
