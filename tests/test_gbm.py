"""Tests for GBM Monte Carlo, validated against Ito-derived analytical moments."""

from __future__ import annotations

import numpy as np
import pytest

from quant_agent.models import gbm, ito


def test_estimate_drift_volatility_recovers_synthetic_params():
    rng = np.random.default_rng(123)
    true_mu, true_sigma, S0 = 0.12, 0.30, 100.0
    dt = 1 / 252
    n = 252 * 10  # long sample for tight estimates
    drift = (true_mu - 0.5 * true_sigma**2) * dt
    shocks = drift + true_sigma * np.sqrt(dt) * rng.standard_normal(n)
    prices = S0 * np.exp(np.cumsum(shocks))

    mu, sigma = gbm.estimate_drift_volatility(prices, window=n)
    assert sigma == pytest.approx(true_sigma, abs=0.02)
    assert mu == pytest.approx(true_mu, abs=0.05)


def test_simulate_paths_shape_and_initial_column():
    paths = gbm.simulate_paths(100.0, 0.1, 0.2, T=1.0, steps=50, n_sims=1000, seed=1)
    assert paths.shape == (1000, 51)
    assert np.allclose(paths[:, 0], 100.0)
    assert np.all(paths > 0)  # GBM stays positive


def test_simulate_paths_is_deterministic_with_seed():
    a = gbm.simulate_paths(100.0, 0.1, 0.2, 1.0, 50, n_sims=500, seed=42)
    b = gbm.simulate_paths(100.0, 0.1, 0.2, 1.0, 50, n_sims=500, seed=42)
    assert np.array_equal(a, b)


def test_monte_carlo_matches_ito_analytical_moments():
    S0, mu, sigma, T = 100.0, 0.10, 0.25, 1.0
    paths = gbm.simulate_paths(S0, mu, sigma, T, steps=252, n_sims=200_000, seed=7)
    # Raises AssertionError on mismatch beyond tolerance.
    assert ito.analytical_vs_monte_carlo_check(paths, S0, mu, sigma, T, tol=0.02)


def test_path_statistics_percentile_ordering():
    paths = gbm.simulate_paths(100.0, 0.08, 0.22, 0.25, 63, n_sims=20_000, seed=3)
    stats = gbm.path_statistics(paths, 100.0)
    assert stats["p5"] < stats["p25"] < stats["p50"] < stats["p75"] < stats["p95"]
    assert 0.0 <= stats["prob_above_s0"] <= 1.0


def test_expected_value_of_call_matches_black_scholes():
    """E[max(S_T - K,0)] under the risk-neutral measure == discounted BS call."""
    from quant_agent.models.black_scholes import bs_call

    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.30, 1.0
    # Under risk-neutral drift mu = r, discounted expectation equals BS price.
    undiscounted = ito.expected_value_of_function(
        lambda s: max(s - K, 0.0), S0, mu=r, sigma=sigma, T=T
    )
    analytic = bs_call(S0, K, T, r, sigma) * np.exp(r * T)
    assert undiscounted == pytest.approx(analytic, rel=1e-3)


def test_estimate_drift_volatility_rejects_2d():
    with pytest.raises(ValueError):
        gbm.estimate_drift_volatility(np.ones((10, 2)))


def test_estimate_drift_volatility_rejects_single_price():
    with pytest.raises(ValueError):
        gbm.estimate_drift_volatility(np.array([100.0]))


@pytest.mark.parametrize("bad", [{"steps": 0}, {"n_sims": 0}])
def test_simulate_paths_rejects_nonpositive(bad):
    kwargs = {"steps": 10, "n_sims": 10, **bad}
    with pytest.raises(ValueError):
        gbm.simulate_paths(100.0, 0.1, 0.2, T=1.0, **kwargs)


def test_prob_above_matches_manual_fraction():
    paths = gbm.simulate_paths(100.0, 0.10, 0.20, 0.5, 50, n_sims=10_000, seed=5)
    p = gbm.prob_above(paths, 100.0)
    assert p == pytest.approx(float(np.mean(paths[:, -1] > 100.0)))


def test_upside_downside_ratio_positive_and_inf_cases():
    paths = gbm.simulate_paths(100.0, 0.15, 0.25, 1.0, 50, n_sims=10_000, seed=9)
    ratio = gbm.upside_downside_ratio(paths, 100.0)
    assert ratio > 0
    # All paths above S0 -> no downside -> inf.
    all_up = np.full((100, 2), 200.0)
    assert gbm.upside_downside_ratio(all_up, 100.0) == float("inf")
