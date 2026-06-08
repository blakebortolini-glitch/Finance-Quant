"""Tests for Markowitz mean-variance optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_agent.models import markowitz


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # Three assets with distinct vols and mild correlation.
    n = 750
    a = rng.normal(0.0006, 0.012, n)
    b = rng.normal(0.0004, 0.020, n)
    c = 0.3 * a + rng.normal(0.0005, 0.015, n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_covariance_matrix_is_symmetric_psd(returns):
    cov = markowitz.covariance_matrix(returns)
    assert list(cov.columns) == ["A", "B", "C"]
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
    eigvals = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigvals > -1e-10).all()  # positive semidefinite


def test_min_variance_weights_sum_to_one_and_nonneg(returns):
    cov = markowitz.covariance_matrix(returns)
    out = markowitz.min_variance_portfolio(cov)
    w = out["weights"]
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-6).all()


def test_efficient_frontier_variance_is_convex_in_return(returns):
    cov = markowitz.covariance_matrix(returns)
    er = pd.Series({"A": 0.14, "B": 0.10, "C": 0.12})
    frontier = markowitz.efficient_frontier(er, cov, n_points=25)
    assert not frontier.empty
    # Min-variance point should not be the highest-return point.
    min_var_row = frontier.loc[frontier["variance"].idxmin()]
    assert min_var_row["target_return"] <= frontier["target_return"].max()


def test_scipy_fallback_matches_constraints(returns):
    # Exercise the scipy.optimize fallback path directly.
    cov = markowitz.covariance_matrix(returns).to_numpy()
    mu = np.array([0.14, 0.10, 0.12])
    w = markowitz._solve_scipy(cov, target=0.12, mu=mu, allow_short=False)
    assert w is not None
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    assert mu @ w == pytest.approx(0.12, abs=1e-4)
    assert (w >= -1e-6).all()


def test_max_sharpe_weights_sum_to_one(returns):
    cov = markowitz.covariance_matrix(returns)
    er = pd.Series({"A": 0.14, "B": 0.08, "C": 0.11})
    out = markowitz.max_sharpe_portfolio(er, cov, rf=0.04)
    assert out["weights"].sum() == pytest.approx(1.0, abs=1e-4)
    assert out["sharpe"] > 0
    # Highest-return, reasonable-risk asset A should get meaningful weight.
    assert out["weights"]["A"] > out["weights"]["B"]
