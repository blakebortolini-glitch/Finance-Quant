"""Markowitz mean-variance optimization.

Program:  minimize w' Sigma w  s.t.  w'R = target,  sum(w) = 1,  w >= 0

Covariance via Ledoit-Wolf shrinkage. Quadratic program via cvxpy with a
scipy.optimize fallback. Short-selling controlled by the config flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.covariance import LedoitWolf

from quant_agent.config import (
    MARKOWITZ_ALLOW_SHORT,
    MARKOWITZ_N_FRONTIER_POINTS,
    TRADING_DAYS_PER_YEAR,
)


def covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Annualized Ledoit-Wolf shrinkage covariance from periodic returns.

    Args:
        returns_df: DataFrame of periodic (e.g. daily) returns, one col/ticker.

    Returns:
        Annualized covariance matrix as a DataFrame (tickers x tickers).
    """
    clean = returns_df.dropna()
    lw = LedoitWolf().fit(clean.to_numpy())
    cov = lw.covariance_ * TRADING_DAYS_PER_YEAR
    return pd.DataFrame(cov, index=returns_df.columns, columns=returns_df.columns)


def _solve_min_variance(
    cov: np.ndarray, target: float | None, mu: np.ndarray | None, allow_short: bool
) -> np.ndarray | None:
    """Solve min w'Σw s.t. sum(w)=1 (+ optional return target). cvxpy first."""
    n = cov.shape[0]
    try:
        import cvxpy as cp

        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1]
        if not allow_short:
            constraints.append(w >= 0)
        if target is not None and mu is not None:
            constraints.append(mu @ w == target)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints)
        prob.solve()
        if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        return np.asarray(w.value, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - fall back to scipy
        logger.debug(f"cvxpy solve failed ({exc}); using scipy fallback")
        return _solve_scipy(cov, target, mu, allow_short)


def _solve_scipy(
    cov: np.ndarray, target: float | None, mu: np.ndarray | None, allow_short: bool
) -> np.ndarray | None:
    from scipy.optimize import minimize

    n = cov.shape[0]
    x0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target is not None and mu is not None:
        constraints.append({"type": "eq", "fun": lambda w, t=target: mu @ w - t})
    bounds = None if allow_short else [(0.0, 1.0)] * n

    res = minimize(
        lambda w: w @ cov @ w,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x if res.success else None


def min_variance_portfolio(cov_matrix: pd.DataFrame) -> dict:
    """Global minimum-variance portfolio."""
    cov = cov_matrix.to_numpy()
    w = _solve_min_variance(cov, target=None, mu=None, allow_short=MARKOWITZ_ALLOW_SHORT)
    if w is None:
        raise RuntimeError("min-variance optimization failed")
    weights = pd.Series(w, index=cov_matrix.columns)
    return {"weights": weights, "variance": float(w @ cov @ w)}


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = MARKOWITZ_N_FRONTIER_POINTS,
) -> pd.DataFrame:
    """Trace the efficient frontier between min and max expected return.

    Returns a DataFrame with columns: target_return, variance, volatility, and
    one weight column per ticker.
    """
    cov = cov_matrix.to_numpy()
    mu = expected_returns.to_numpy()
    targets = np.linspace(mu.min(), mu.max(), n_points)

    rows = []
    for t in targets:
        w = _solve_min_variance(cov, target=float(t), mu=mu, allow_short=MARKOWITZ_ALLOW_SHORT)
        if w is None:
            continue
        var = float(w @ cov @ w)
        row = {"target_return": float(t), "variance": var, "volatility": float(np.sqrt(var))}
        row.update({tic: float(wi) for tic, wi in zip(expected_returns.index, w, strict=True)})
        rows.append(row)
    return pd.DataFrame(rows)


def max_sharpe_portfolio(
    expected_returns: pd.Series, cov_matrix: pd.DataFrame, rf: float
) -> dict:
    """Tangency (max-Sharpe) portfolio, found by scanning the frontier.

    Returns dict with weights (Series), expected_return, volatility, sharpe.
    """
    frontier = efficient_frontier(expected_returns, cov_matrix)
    if frontier.empty:
        raise RuntimeError("efficient frontier is empty; optimization failed")

    sharpe = (frontier["target_return"] - rf) / frontier["volatility"]
    best = frontier.loc[sharpe.idxmax()]
    weights = pd.Series(
        {tic: float(best[tic]) for tic in expected_returns.index}, index=expected_returns.index
    )
    return {
        "weights": weights,
        "expected_return": float(best["target_return"]),
        "volatility": float(best["volatility"]),
        "sharpe": float(sharpe.max()),
    }


__all__ = [
    "covariance_matrix",
    "efficient_frontier",
    "max_sharpe_portfolio",
    "min_variance_portfolio",
]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0.0005, 0.02, (500, 3)), columns=["A", "B", "C"])
    cov = covariance_matrix(rets)
    er = pd.Series({"A": 0.12, "B": 0.10, "C": 0.15})
    ms = max_sharpe_portfolio(er, cov, rf=0.04)
    print("max-Sharpe weights:\n", ms["weights"].round(3).to_string())
    print(f"sharpe={ms['sharpe']:.3f}")
