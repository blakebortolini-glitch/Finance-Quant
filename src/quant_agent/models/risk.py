"""Risk metrics: Sharpe, Sortino, VaR, CVaR, maximum drawdown.

VaR/CVaR at 95% and 99%, both historical and Monte Carlo (parametric normal).
Maximum drawdown is reported with peak/trough indices and time-to-recovery.

Conventions: ``returns`` are periodic (daily) simple returns. Sharpe/Sortino
are annualized; VaR/CVaR are per-period and reported as negative numbers
(a loss), e.g. -0.032 = a 3.2% daily loss.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

from quant_agent.config import TRADING_DAYS_PER_YEAR

FloatArray = npt.NDArray[np.float64]


def _clean(returns: FloatArray) -> FloatArray:
    r = np.asarray(returns, dtype=np.float64)
    return r[np.isfinite(r)]


def sharpe_ratio(returns: FloatArray, rf: float) -> float:
    """Annualized Sharpe ratio. ``rf`` is the annualized risk-free rate."""
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    excess_daily = r.mean() - rf / TRADING_DAYS_PER_YEAR
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess_daily / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: FloatArray, rf: float) -> float:
    """Annualized Sortino ratio (downside-deviation denominator)."""
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    daily_rf = rf / TRADING_DAYS_PER_YEAR
    excess = r - daily_rf
    # Target semideviation: average squared shortfall over *all* observations,
    # which is <= total std, so Sortino >= Sharpe for a positive excess mean.
    downside = np.minimum(excess, 0.0)
    if not np.any(downside < 0):
        return float("inf")
    downside_dev = np.sqrt(np.mean(downside**2))
    if downside_dev == 0:
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR))


def value_at_risk(
    returns: FloatArray, confidence: float = 0.95, method: str = "historical"
) -> float:
    """Per-period Value at Risk (a negative number = loss).

    Args:
        returns: periodic simple returns.
        confidence: e.g. 0.95 for the 95% VaR.
        method: "historical" (empirical quantile) or "monte_carlo"
            (parametric normal fit).
    """
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    alpha = 1.0 - confidence
    if method == "historical":
        return float(np.percentile(r, alpha * 100))
    if method == "monte_carlo":
        mu, sigma = r.mean(), r.std(ddof=1)
        return float(stats.norm.ppf(alpha, loc=mu, scale=sigma))
    raise ValueError(f"unknown VaR method '{method}'")


def conditional_var(returns: FloatArray, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall: mean loss beyond the VaR quantile."""
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    var = value_at_risk(r, confidence, method="historical")
    tail = r[r <= var]
    if tail.size == 0:
        return float(var)
    return float(tail.mean())


def max_drawdown(prices: FloatArray) -> dict[str, float]:
    """Maximum drawdown with peak/trough indices and time-to-recovery.

    Args:
        prices: 1-D price series.

    Returns:
        dict with max_drawdown (negative decimal), peak_idx, trough_idx, and
        recovery_days (NaN if the series never recovers to the prior peak).
    """
    p = np.asarray(prices, dtype=np.float64)
    if p.size < 2:
        return {"max_drawdown": 0.0, "peak_idx": 0, "trough_idx": 0, "recovery_days": float("nan")}

    running_max = np.maximum.accumulate(p)
    drawdowns = p / running_max - 1.0
    trough_idx = int(np.argmin(drawdowns))
    max_dd = float(drawdowns[trough_idx])
    peak_idx = int(np.argmax(p[: trough_idx + 1])) if trough_idx > 0 else 0

    peak_value = p[peak_idx]
    recovery_days = float("nan")
    post = np.where(p[trough_idx:] >= peak_value)[0]
    if post.size > 0:
        recovery_days = float(post[0])

    return {
        "max_drawdown": max_dd,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "recovery_days": recovery_days,
    }


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "value_at_risk",
    "conditional_var",
    "max_drawdown",
]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.015, 756)
    prices = 100 * np.cumprod(1 + rets)
    print(f"Sharpe  {sharpe_ratio(rets, 0.04):.2f}")
    print(f"Sortino {sortino_ratio(rets, 0.04):.2f}")
    print(f"VaR95   {value_at_risk(rets, 0.95):.2%}/day")
    print(f"CVaR95  {conditional_var(rets, 0.95):.2%}/day")
    print(f"MaxDD   {max_drawdown(prices)['max_drawdown']:.2%}")
