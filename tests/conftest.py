"""Shared pytest fixtures and deterministic seeding for Monte Carlo tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_agent.config import GBM_RANDOM_SEED


@pytest.fixture
def synthetic_prices() -> np.ndarray:
    """A clean GBM-like price series for model unit tests."""
    rng = np.random.default_rng(GBM_RANDOM_SEED)
    mu, sigma, dt, n = 0.10, 0.25, 1 / 252, 756  # ~3y daily
    shocks = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    return 100.0 * np.exp(np.cumsum(shocks))
