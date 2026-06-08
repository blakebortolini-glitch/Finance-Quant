"""Geometric Brownian Motion + vectorized Monte Carlo.

Model:  dS_t = mu * S_t * dt + sigma * S_t * dW_t

Exact discretization (no Euler error):
    S_{t+dt} = S_t * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z),  Z ~ N(0,1)

Drift feeds CAPM (actual expected return blend); win probability and
upside/downside ratio feed Kelly sizing. Validated against the analytical
Ito-derived log-normal moments in ``ito.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_agent.config import TRADING_DAYS_PER_YEAR

FloatArray = npt.NDArray[np.float64]


def estimate_drift_volatility(prices: FloatArray, window: int = 252) -> tuple[float, float]:
    """Annualized (mu, sigma) from log-returns over the trailing window.

    Drift is reported in the *arithmetic* (price-process) convention:
        mu = mean(log_returns) * N + sigma^2 / 2
    so that exp(mu * T) is the expected gross return, consistent with the GBM
    SDE drift term and with ``ito.log_return_distribution``.

    Args:
        prices: 1-D array of close prices, shape (n,).
        window: lookback length in trading days (uses the last ``window`` obs).

    Returns:
        (mu, sigma) annualized drift and volatility (decimals).
    """
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 1:
        raise ValueError(f"prices must be 1-D, got shape {prices.shape}")
    if prices.size < 2:
        raise ValueError("need at least 2 prices to compute a return")

    log_returns = np.diff(np.log(prices[-(window + 1):]))
    n = TRADING_DAYS_PER_YEAR
    sigma = float(np.std(log_returns, ddof=1) * np.sqrt(n))
    mu_log = float(np.mean(log_returns) * n)        # log-drift
    mu = mu_log + 0.5 * sigma**2                     # arithmetic drift
    return mu, sigma


def simulate_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> FloatArray:
    """Vectorized Monte Carlo via exact log-normal discretization.

    Args:
        S0: initial price.
        mu: annualized arithmetic drift (decimal).
        sigma: annualized volatility (decimal).
        T: horizon in years.
        steps: number of time steps over [0, T].
        n_sims: number of simulated paths.
        seed: RNG seed for reproducibility.

    Returns:
        Array of shape (n_sims, steps + 1); column 0 is S0 for every path.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")

    rng = np.random.default_rng(seed)
    dt = T / steps
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    z = rng.standard_normal(size=(n_sims, steps))
    log_increments = drift + diffusion * z          # (n_sims, steps)
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_sims, steps + 1), dtype=np.float64)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def path_statistics(paths: FloatArray, S0: float) -> dict[str, float]:
    """Terminal-price percentiles, P(S_T > S0), and expected return.

    Args:
        paths: shape (n_sims, steps + 1) from ``simulate_paths``.
        S0: initial price for the probability-above and return calcs.

    Returns:
        dict with keys p5, p25, p50, p75, p95, prob_above_s0, expected_return.
    """
    terminal = paths[:, -1]
    p5, p25, p50, p75, p95 = np.percentile(terminal, [5, 25, 50, 75, 95])
    return {
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "prob_above_s0": float(np.mean(terminal > S0)),
        "expected_return": float(np.mean(terminal) / S0 - 1.0),
    }


def prob_above(paths: FloatArray, level: float) -> float:
    """P(S_T > ``level``) from the terminal distribution of the paths."""
    return float(np.mean(paths[:, -1] > level))


def upside_downside_ratio(paths: FloatArray, S0: float) -> float:
    """Ratio of mean simulated upside to mean simulated downside vs S0.

    Used as Kelly's ``b`` (win/loss ratio). Returns ``inf`` if no losing path.
    """
    terminal = paths[:, -1]
    gains = terminal[terminal > S0] - S0
    losses = S0 - terminal[terminal <= S0]
    if losses.size == 0 or losses.sum() == 0:
        return float("inf")
    if gains.size == 0:
        return 0.0
    return float(gains.mean() / losses.mean())


__all__ = [
    "estimate_drift_volatility",
    "simulate_paths",
    "path_statistics",
    "prob_above",
    "upside_downside_ratio",
]


if __name__ == "__main__":  # pragma: no cover - self-test with synthetic data
    rng = np.random.default_rng(0)
    true_mu, true_sigma, S0 = 0.12, 0.30, 100.0
    dt = 1 / TRADING_DAYS_PER_YEAR
    shocks = (true_mu - 0.5 * true_sigma**2) * dt + true_sigma * np.sqrt(dt) * rng.standard_normal(
        2520
    )
    synthetic = S0 * np.exp(np.cumsum(shocks))

    mu_hat, sigma_hat = estimate_drift_volatility(synthetic, window=2520)
    paths = simulate_paths(S0, mu_hat, sigma_hat, T=0.25, steps=63, n_sims=50_000, seed=1)
    stats = path_statistics(paths, S0)
    print(f"estimated mu={mu_hat:.3f} (true {true_mu}), sigma={sigma_hat:.3f} (true {true_sigma})")
    print(f"90d P(S_T>S0)={stats['prob_above_s0']:.3f}, E[ret]={stats['expected_return']:.3%}")
