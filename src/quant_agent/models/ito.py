"""Ito's Lemma — analytical validator for the GBM Monte Carlo.

For GBM with arithmetic drift mu, applying Ito's lemma to f = ln(S) gives
    d(ln S) = (mu - sigma^2/2) dt + sigma dW
so
    ln(S_T / S_0) ~ Normal((mu - sigma^2/2) * T, sigma^2 * T).

Used to (a) provide closed-form moments to cross-check the simulator, and
(b) compute E[f(S_T)] analytically for arbitrary payoffs via the log-normal.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import integrate, stats

FloatArray = npt.NDArray[np.float64]


def log_return_distribution(S0: float, mu: float, sigma: float, T: float) -> stats.rv_frozen:
    """Return the analytical Normal distribution of ln(S_T / S_0).

    Args:
        S0: initial price (unused in the distribution of the *log return*,
            kept for signature symmetry / clarity).
        mu: annualized arithmetic drift.
        sigma: annualized volatility.
        T: horizon in years.

    Returns:
        A frozen scipy normal distribution with mean (mu - sigma^2/2)T and
        std sigma*sqrt(T).
    """
    mean = (mu - 0.5 * sigma**2) * T
    std = sigma * np.sqrt(T)
    return stats.norm(loc=mean, scale=std)


def terminal_price_moments(S0: float, mu: float, sigma: float, T: float) -> dict[str, float]:
    """Analytical mean and variance of S_T under the log-normal.

    E[S_T]   = S0 * exp(mu * T)
    Var[S_T] = S0^2 * exp(2 mu T) * (exp(sigma^2 T) - 1)
    """
    mean = S0 * np.exp(mu * T)
    var = S0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1.0)
    return {"mean": float(mean), "variance": float(var), "std": float(np.sqrt(var))}


def analytical_vs_monte_carlo_check(
    paths: FloatArray,
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    tol: float = 0.05,
) -> bool:
    """Assert simulated terminal moments match Ito-derived moments.

    Compares the Monte Carlo mean and std of S_T against the closed-form
    log-normal moments, requiring agreement within relative tolerance ``tol``.

    Returns:
        True if both moments agree within tolerance, else raises AssertionError.
    """
    analytic = terminal_price_moments(S0, mu, sigma, T)
    terminal = paths[:, -1]
    mc_mean = float(terminal.mean())
    mc_std = float(terminal.std(ddof=1))

    mean_err = abs(mc_mean - analytic["mean"]) / analytic["mean"]
    std_err = abs(mc_std - analytic["std"]) / analytic["std"]

    assert mean_err < tol, f"mean mismatch: MC={mc_mean:.4f} vs analytic={analytic['mean']:.4f}"
    assert std_err < tol, f"std mismatch: MC={mc_std:.4f} vs analytic={analytic['std']:.4f}"
    return True


def expected_value_of_function(
    f: Callable[[float], float], S0: float, mu: float, sigma: float, T: float
) -> float:
    """Compute E[f(S_T)] analytically via the log-normal density of S_T.

    Integrates f(S_T) against the log-normal pdf of S_T. ``f`` may be any
    measurable payoff (e.g. a call payoff max(S-K, 0)).
    """
    mean_log = np.log(S0) + (mu - 0.5 * sigma**2) * T
    std_log = sigma * np.sqrt(T)
    dist = stats.lognorm(s=std_log, scale=np.exp(mean_log))

    def integrand(s: float) -> float:
        return f(s) * dist.pdf(s)

    upper = dist.ppf(0.999999)
    value, _ = integrate.quad(integrand, 0.0, upper, limit=200)
    return float(value)


__all__ = [
    "log_return_distribution",
    "terminal_price_moments",
    "analytical_vs_monte_carlo_check",
    "expected_value_of_function",
]


if __name__ == "__main__":  # pragma: no cover
    from quant_agent.models.gbm import simulate_paths

    S0, mu, sigma, T = 100.0, 0.10, 0.25, 1.0
    paths = simulate_paths(S0, mu, sigma, T, steps=252, n_sims=100_000, seed=7)
    ok = analytical_vs_monte_carlo_check(paths, S0, mu, sigma, T, tol=0.02)
    print(f"Ito vs MC moment check passed: {ok}")
