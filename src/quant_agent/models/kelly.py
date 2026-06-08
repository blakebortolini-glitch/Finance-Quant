"""Kelly Criterion position sizing.

Discrete:    f* = (p(b + 1) - 1) / b
Continuous:  f* = (mu - r) / sigma^2

Inputs: p from GBM P(S_T > S_0); b from the ratio of average simulated upside
to average simulated downside (``gbm.upside_downside_ratio``). Hard-capped at
``KELLY_MAX_POSITION`` regardless of the raw Kelly output. Default to
fractional (quarter) Kelly.
"""

from __future__ import annotations

import math

from quant_agent.config import KELLY_FRACTION, KELLY_MAX_POSITION


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Vanilla discrete Kelly fraction f* = (p(b+1) - 1) / b.

    Args:
        win_prob: probability of a winning outcome, p in [0, 1].
        win_loss_ratio: payoff ratio b = avg_win / avg_loss (> 0).

    Returns:
        The raw Kelly fraction (may be negative -> do not bet long), not capped.
    """
    if not 0.0 <= win_prob <= 1.0:
        raise ValueError(f"win_prob must be in [0, 1], got {win_prob}")
    b = win_loss_ratio
    if not math.isfinite(b) or b <= 0:
        # Infinite/zero payoff ratio: no finite Kelly bet is well-defined.
        return 0.0
    return float((win_prob * (b + 1) - 1) / b)


def continuous_kelly(expected_return: float, variance: float, rf: float) -> float:
    """Continuous Kelly for stock sizing: f* = (mu - r) / sigma^2.

    Args:
        expected_return: annualized expected return mu (decimal).
        variance: annualized return variance sigma^2 (> 0).
        rf: annualized risk-free rate.

    Returns:
        The raw continuous Kelly fraction (uncapped).
    """
    if variance <= 0:
        raise ValueError("variance must be positive")
    return float((expected_return - rf) / variance)


def fractional_kelly(f_star: float, fraction: float = KELLY_FRACTION) -> float:
    """Scale Kelly by a fraction and apply the hard position cap.

    Negative raw Kelly is floored at 0 (no long position recommended); the
    scaled result is clamped to [0, KELLY_MAX_POSITION].
    """
    scaled = max(0.0, f_star) * fraction
    return float(min(scaled, KELLY_MAX_POSITION))


__all__ = ["kelly_fraction", "continuous_kelly", "fractional_kelly"]


if __name__ == "__main__":  # pragma: no cover
    f = kelly_fraction(win_prob=0.6, win_loss_ratio=1.5)
    print(f"discrete Kelly={f:.3f} -> quarter={fractional_kelly(f):.3f}")
    fc = continuous_kelly(expected_return=0.18, variance=0.30**2, rf=0.05)
    print(f"continuous Kelly={fc:.3f} -> quarter(capped)={fractional_kelly(fc):.3f}")
