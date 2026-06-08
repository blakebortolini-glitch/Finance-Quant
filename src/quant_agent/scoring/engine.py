"""Composite scoring engine: weighted signal fusion into a verdict.

Each raw model signal is normalized to a [-1, +1] contribution (positive =
bullish / undervalued), multiplied by its configured weight, and the weighted
average is scaled to a composite score in [-100, +100]. When only a subset of
signals is available (Milestone 1 ships CAPM alpha, GBM upside, and the
implied-vs-realized vol signal), the present weights are renormalized so the
composite still spans the full range.

Confidence reflects agreement across the *available* signals. Signal weights,
verdict cutoffs, and confidence thresholds all live in ``config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from quant_agent.config import (
    ALTMAN_Z_SAFE,
    SCORING_WEIGHTS,
    VERDICT_OVERVALUED,
    VERDICT_SLIGHT_OVER,
    VERDICT_SLIGHT_UNDER,
    VERDICT_UNDERVALUED,
)

# Normalization scales: the raw signal value that maps to a full +/-1 vote.
ALPHA_FULL_SCALE = 0.10        # +/-10% CAPM alpha -> +/-1
UPSIDE_NEUTRAL = 0.50          # P=0.5 is neutral; spans [0,1] -> [-1,+1]
VOL_GAP_FULL_SCALE = 0.10      # +/-10pp realized-minus-implied -> +/-1
DCF_FULL_SCALE = 0.30          # +/-30% intrinsic-vs-price gap -> +/-1
PIOTROSKI_MID = 4.5            # F-Score midpoint; 9 -> +1, 0 -> -1
PIOTROSKI_SCALE = 4.5
ALTMAN_FULL_SCALE = 3.0        # Z distance from the safe threshold for +/-1


class Verdict(StrEnum):
    UNDERVALUED = "UNDERVALUED"
    GROWTH = "GROWTH"
    FAIR = "FAIR"
    SLIGHT_OVER = "SLIGHTLY OVERVALUED"
    OVERVALUED = "OVERVALUED"


class Confidence(StrEnum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class SignalContribution:
    """One signal's normalized value, weight, and direction (-1/0/+1)."""

    name: str
    value: float          # normalized to [-1, +1]
    weight: float
    direction: int        # sign of the bullish/bearish vote


@dataclass
class ScoreResult:
    ticker: str
    composite: float                       # [-100, +100]
    verdict: Verdict
    confidence: Confidence
    signals: list[SignalContribution] = field(default_factory=list)
    n_agree: int = 0
    n_disagree: int = 0
    partial: bool = False                  # True when not all 7 signals present
    price: float | None = None             # last close, for reporting
    metrics: dict = field(default_factory=dict)  # raw figures for deeper tiers
    previous_verdict: str | None = None    # verdict at the prior run, if any
    changed: bool = False                  # verdict differs from the prior run
    in_watchlist: bool = True              # curated watchlist vs. on-demand coverage


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def normalize_alpha(alpha: float) -> float:
    """CAPM alpha (decimal) -> [-1, +1]. Positive alpha is bullish."""
    return _clip(alpha / ALPHA_FULL_SCALE)


def normalize_upside(prob_above: float) -> float:
    """P(S_T > S0 * threshold) in [0,1] -> [-1, +1] centered at 0.5."""
    return _clip((prob_above - UPSIDE_NEUTRAL) / UPSIDE_NEUTRAL)


def normalize_vol(gap: float) -> float:
    """IV-minus-realized gap -> [-1, +1]. Vol *discount* (gap<0) is bullish."""
    if not np.isfinite(gap):
        return 0.0
    return _clip(-gap / VOL_GAP_FULL_SCALE)


def normalize_dcf(intrinsic: float, price: float) -> float:
    """DCF intrinsic-vs-price gap -> [-1, +1]. Intrinsic > price is bullish."""
    if price <= 0 or not np.isfinite(intrinsic):
        return 0.0
    return _clip((intrinsic / price - 1.0) / DCF_FULL_SCALE)


def normalize_piotroski(score: int) -> float:
    """F-Score 0-9 -> [-1, +1], centered at the midpoint."""
    return _clip((score - PIOTROSKI_MID) / PIOTROSKI_SCALE)


def normalize_altman(z: float) -> float:
    """Altman Z -> [-1, +1], centered at the safe threshold (2.99)."""
    if not np.isfinite(z):
        return 0.0
    return _clip((z - ALTMAN_Z_SAFE) / ALTMAN_FULL_SCALE)


def normalize_markowitz(weight: float, n_assets: int) -> float:
    """Max-Sharpe weight -> [-1, +1] relative to the equal-weight baseline."""
    if n_assets <= 0:
        return 0.0
    equal = 1.0 / n_assets
    return _clip((weight - equal) / equal)


def normalize_insider(score: float) -> float:
    """Insider net-buy ratio is already in [-1, +1]; clamp for safety."""
    return _clip(score)


def normalize_institutional(score: float) -> float:
    """13F accumulation score is already in [-1, +1]; clamp for safety."""
    return _clip(score)


def classify_verdict(composite: float) -> Verdict:
    """Map composite score to a verdict using config thresholds."""
    if composite > VERDICT_UNDERVALUED:
        return Verdict.UNDERVALUED
    if composite > VERDICT_SLIGHT_UNDER:
        return Verdict.GROWTH
    if composite >= VERDICT_SLIGHT_OVER:
        return Verdict.FAIR
    if composite >= VERDICT_OVERVALUED:
        return Verdict.SLIGHT_OVER
    return Verdict.OVERVALUED


def _confidence(n_agree: int, n_total: int) -> Confidence:
    """Confidence from the fraction of available signals that agree."""
    if n_total == 0:
        return Confidence.LOW
    frac = n_agree / n_total
    if frac >= 0.8:
        return Confidence.HIGH
    if frac >= 0.5:
        return Confidence.MEDIUM
    return Confidence.LOW


# Map signal name -> config weight attribute.
_WEIGHT_OF = {
    "capm_alpha": SCORING_WEIGHTS.capm_alpha,
    "gbm_upside": SCORING_WEIGHTS.gbm_upside,
    "vol_premium": SCORING_WEIGHTS.vol_premium,
    "dcf_intrinsic": SCORING_WEIGHTS.dcf_intrinsic,
    "piotroski": SCORING_WEIGHTS.piotroski,
    "altman_z": SCORING_WEIGHTS.altman_z,
    "markowitz_weight": SCORING_WEIGHTS.markowitz_weight,
    # Smart Money add-ons — default weight 0.0 (excluded from the composite
    # until raised in config); included by the orchestrator only when > 0.
    "insider": SCORING_WEIGHTS.insider,
    "institutional": SCORING_WEIGHTS.institutional,
}

# "Partial" is judged against the seven CORE signals, not the optional add-ons.
_CORE_SIGNALS = frozenset({
    "capm_alpha", "gbm_upside", "vol_premium", "dcf_intrinsic",
    "piotroski", "altman_z", "markowitz_weight",
})
_TOTAL_SIGNALS = len(_CORE_SIGNALS)


def score_ticker(ticker: str, normalized_signals: dict[str, float]) -> ScoreResult:
    """Fuse already-normalized signals (each in [-1,+1]) into a ScoreResult.

    Args:
        ticker: the symbol being scored.
        normalized_signals: name -> normalized value in [-1, +1]. Only the
            signals present are used; weights are renormalized over them.

    Returns:
        A ScoreResult with composite score, verdict, confidence, and the
        per-signal contribution breakdown.
    """
    contributions: list[SignalContribution] = []
    present_weight = 0.0
    weighted_sum = 0.0

    for name, value in normalized_signals.items():
        if name not in _WEIGHT_OF:
            raise KeyError(f"unknown signal '{name}'")
        weight = _WEIGHT_OF[name]
        if weight == 0.0:
            continue  # zero-weight signal contributes nothing and casts no vote
        present_weight += weight
        weighted_sum += value * weight
        direction = int(np.sign(value)) if abs(value) > 1e-9 else 0
        contributions.append(SignalContribution(name, value, weight, direction))

    if present_weight == 0:
        raise ValueError("no signals provided to score")

    composite = 100.0 * weighted_sum / present_weight
    verdict = classify_verdict(composite)

    composite_sign = int(np.sign(composite)) if abs(composite) > 1e-9 else 0
    voting = [c for c in contributions if c.direction != 0]
    n_agree = sum(1 for c in voting if c.direction == composite_sign)
    n_disagree = len(voting) - n_agree
    confidence = _confidence(n_agree, len(voting))

    return ScoreResult(
        ticker=ticker,
        composite=round(composite, 1),
        verdict=verdict,
        confidence=confidence,
        signals=contributions,
        n_agree=n_agree,
        n_disagree=n_disagree,
        partial=sum(1 for n in normalized_signals if n in _CORE_SIGNALS) < _TOTAL_SIGNALS,
    )


__all__ = [
    "Verdict",
    "Confidence",
    "SignalContribution",
    "ScoreResult",
    "normalize_alpha",
    "normalize_upside",
    "normalize_vol",
    "normalize_dcf",
    "normalize_piotroski",
    "normalize_altman",
    "normalize_markowitz",
    "normalize_insider",
    "normalize_institutional",
    "score_ticker",
    "classify_verdict",
]
