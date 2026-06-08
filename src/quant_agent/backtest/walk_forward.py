"""Walk-forward validation harness.

Replays a *reconstructable* slice of the scoring engine on historical prices
and reports whether a bullish signal at time t was followed by a positive
forward return over the next ``horizon_days``.

Scope & honesty note
--------------------
A faithful replay of the full 7-signal composite is not possible from this
data source: point-in-time fundamentals (DCF, Piotroski, Altman) and historical
options chains (the vol signal) are not available from yfinance. The backtest
therefore exercises the **price-derivable** signal — the GBM upside probability,
computed from data available *up to t only* (no look-ahead) — which is the core
stochastic-model signal in the composite. Hit rate is compared against the base
rate (unconditional probability of a positive forward return) so the signal's
edge, if any, is visible above chance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from quant_agent.config import (
    GBM_DRIFT_WINDOW,
    GBM_HORIZON_DAYS,
    GBM_UPSIDE_THRESHOLD,
    TRADING_DAYS_PER_YEAR,
)
from quant_agent.data.fetcher import fetch_prices
from quant_agent.models import gbm


@dataclass
class BacktestResult:
    ticker: str
    n_signals: int           # total evaluation points
    n_bullish: int           # points flagged bullish
    hit_rate: float          # P(forward return > 0 | bullish)
    base_rate: float         # P(forward return > 0) unconditionally
    edge: float              # hit_rate - base_rate
    avg_forward_return: float        # mean forward return when bullish
    avg_forward_return_all: float    # mean forward return overall


def _analytic_upside_prob(window: np.ndarray, threshold: float, horizon_years: float) -> float:
    """P(S_T > S0 * threshold) under GBM, via the Ito log-normal (no MC).

    Uses the closed-form ln(S_T/S0) ~ Normal((mu - sigma^2/2)T, sigma^2 T).
    """
    mu, sigma = gbm.estimate_drift_volatility(window, window=GBM_DRIFT_WINDOW)
    if sigma <= 0:
        return 0.5
    mean = (mu - 0.5 * sigma**2) * horizon_years
    std = sigma * np.sqrt(horizon_years)
    log_threshold = np.log(threshold)
    return float(1.0 - stats.norm.cdf((log_threshold - mean) / std))


def walk_forward(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    horizon_days: int = GBM_HORIZON_DAYS,
    step: int = 21,
    bullish_prob: float = 0.5,
) -> BacktestResult:
    """Replay the GBM upside signal over history and score its forward hit rate.

    Args:
        ticker: symbol to backtest.
        start, end: optional ISO date bounds to restrict the price window.
        horizon_days: forward return horizon (trading days).
        step: spacing between evaluation points (trading days).
        bullish_prob: upside-probability threshold to call a point "bullish".

    Returns:
        A BacktestResult summarizing hit rate vs base rate.
    """
    prices = fetch_prices(ticker, date.today())["Close"]
    if start:
        prices = prices[prices.index >= pd.Timestamp(start)]
    if end:
        prices = prices[prices.index <= pd.Timestamp(end)]
    p = prices.to_numpy(dtype=np.float64)

    horizon_years = horizon_days / TRADING_DAYS_PER_YEAR
    first = GBM_DRIFT_WINDOW + 1
    last = len(p) - horizon_days
    if last <= first:
        raise ValueError(
            f"{ticker}: insufficient history ({len(p)} days) for a "
            f"{GBM_DRIFT_WINDOW}+{horizon_days} backtest"
        )

    bullish_fwd: list[float] = []
    all_fwd: list[float] = []
    for t in range(first, last, step):
        window = p[:t]
        fwd_return = p[t + horizon_days] / p[t] - 1.0
        all_fwd.append(fwd_return)
        prob = _analytic_upside_prob(window, GBM_UPSIDE_THRESHOLD, horizon_years)
        if prob > bullish_prob:
            bullish_fwd.append(fwd_return)

    n_signals = len(all_fwd)
    n_bullish = len(bullish_fwd)
    base_rate = float(np.mean(np.array(all_fwd) > 0)) if all_fwd else float("nan")
    hit_rate = float(np.mean(np.array(bullish_fwd) > 0)) if bullish_fwd else float("nan")
    avg_fwd = float(np.mean(bullish_fwd)) if bullish_fwd else float("nan")
    avg_fwd_all = float(np.mean(all_fwd)) if all_fwd else float("nan")

    result = BacktestResult(
        ticker=ticker,
        n_signals=n_signals,
        n_bullish=n_bullish,
        hit_rate=hit_rate,
        base_rate=base_rate,
        edge=hit_rate - base_rate if n_bullish else float("nan"),
        avg_forward_return=avg_fwd,
        avg_forward_return_all=avg_fwd_all,
    )
    logger.info(
        f"{ticker}: {n_bullish}/{n_signals} bullish, hit {hit_rate:.0%} "
        f"vs base {base_rate:.0%} (edge {result.edge:+.0%})"
    )
    return result


def walk_forward_watchlist(tickers: list[str], **kwargs) -> pd.DataFrame:
    """Run the backtest across many tickers; return a tidy summary DataFrame."""
    rows = []
    for ticker in tickers:
        try:
            rows.append(vars(walk_forward(ticker, **kwargs)))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{ticker}: backtest failed: {exc}")
    return pd.DataFrame(rows)


__all__ = ["BacktestResult", "walk_forward", "walk_forward_watchlist"]


if __name__ == "__main__":  # pragma: no cover
    print(walk_forward("AAPL"))
