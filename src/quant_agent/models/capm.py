"""Capital Asset Pricing Model — beta, required return, alpha signal.

Model:  E(R_i) = R_f + beta_i * (E(R_m) - R_f)

Beta via OLS (statsmodels) on weekly returns vs the market proxy.
"Actual expected return" = weighted blend of analyst 1y target, GBM drift,
and earnings-based forward yield (weights in config). Positive alpha
(actual ER > CAPM required ER) is an undervalued signal.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm

from quant_agent.config import (
    ER_BLEND_ANALYST_TARGET,
    ER_BLEND_EARNINGS_YIELD,
    ER_BLEND_GBM_DRIFT,
)

FloatArray = npt.NDArray[np.float64]


def estimate_beta(stock_returns: FloatArray, market_returns: FloatArray) -> dict[str, float]:
    """OLS beta regression of stock excess movement on market.

    Regresses stock returns on market returns (with intercept). The slope is
    beta, the intercept is (periodic) alpha, plus the regression R^2.

    Args:
        stock_returns: 1-D array of periodic returns, shape (n,).
        market_returns: 1-D array of matching market returns, shape (n,).

    Returns:
        dict with beta, alpha, r_squared.
    """
    y = np.asarray(stock_returns, dtype=np.float64)
    x = np.asarray(market_returns, dtype=np.float64)
    if y.shape != x.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {x.shape}")
    if y.size < 2:
        raise ValueError("need at least 2 observations for OLS")

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return {
        "beta": float(model.params[1]),
        "alpha": float(model.params[0]),
        "r_squared": float(model.rsquared),
    }


def expected_return(beta: float, rf: float, market_premium: float) -> float:
    """CAPM required return: R_f + beta * (E(R_m) - R_f)."""
    return float(rf + beta * market_premium)


def actual_expected_return(
    analyst_target_return: float | None,
    gbm_drift: float | None,
    earnings_yield: float | None,
) -> float:
    """Blend the three forward-return estimates using config weights.

    Missing components are dropped and the remaining weights renormalized so
    the blend stays a proper weighted average. Raises if all are missing.
    """
    components = [
        (analyst_target_return, ER_BLEND_ANALYST_TARGET),
        (gbm_drift, ER_BLEND_GBM_DRIFT),
        (earnings_yield, ER_BLEND_EARNINGS_YIELD),
    ]
    present = [(v, w) for v, w in components if v is not None and np.isfinite(v)]
    if not present:
        raise ValueError("no expected-return components available to blend")

    total_w = sum(w for _, w in present)
    return float(sum(v * w for v, w in present) / total_w)


def alpha_signal(actual_er: float, capm_required_er: float) -> float:
    """Alpha = actual ER - CAPM required ER. Positive -> undervalued."""
    return float(actual_er - capm_required_er)


__all__ = [
    "estimate_beta",
    "expected_return",
    "actual_expected_return",
    "alpha_signal",
]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    mkt = rng.normal(0.002, 0.02, 104)       # ~2y weekly
    stock = 0.001 + 1.3 * mkt + rng.normal(0, 0.01, 104)
    fit = estimate_beta(stock, mkt)
    req = expected_return(fit["beta"], rf=0.0521, market_premium=0.055)
    print(f"beta={fit['beta']:.3f} R2={fit['r_squared']:.3f} required ER={req:.3%}")
