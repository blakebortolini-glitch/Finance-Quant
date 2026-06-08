"""Tests for the GARCH(1,1) volatility forecast."""

from __future__ import annotations

import math

import numpy as np

from quant_agent.models import garch


def _garch_series(n: int, seed: int) -> np.ndarray:
    """Synthetic returns with volatility clustering (true GARCH-like DGP)."""
    rng = np.random.default_rng(seed)
    vol = 0.01 * np.ones(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        vol[t] = math.sqrt(
            1e-6 + 0.08 * (vol[t - 1] * eps[t - 1]) ** 2 + 0.90 * vol[t - 1] ** 2
        )
    return vol * eps


def test_forecast_returns_positive_annualized_vol():
    rets = _garch_series(1500, seed=1)
    vol = garch.forecast_volatility(rets)
    assert math.isfinite(vol)
    assert 0.0 < vol < 5.0  # sane annualized range


def test_forecast_too_few_returns_is_nan():
    assert math.isnan(garch.forecast_volatility(np.array([0.01, -0.01, 0.02])))


def test_forecast_handles_nans_in_input():
    rets = _garch_series(800, seed=2)
    rets[::50] = np.nan
    vol = garch.forecast_volatility(rets)
    assert math.isfinite(vol)
