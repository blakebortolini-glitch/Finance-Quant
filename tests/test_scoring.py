"""Tests for the composite scoring engine."""

from __future__ import annotations

import pytest

from quant_agent.scoring import engine
from quant_agent.scoring.engine import Confidence, Verdict


def test_normalizers_clip_to_unit_range():
    assert engine.normalize_alpha(1.0) == 1.0      # huge alpha clips to +1
    assert engine.normalize_alpha(-1.0) == -1.0
    assert engine.normalize_upside(1.0) == 1.0
    assert engine.normalize_upside(0.0) == -1.0
    assert engine.normalize_upside(0.5) == 0.0
    assert engine.normalize_vol(-0.10) == pytest.approx(1.0)   # vol discount bullish
    assert engine.normalize_vol(0.10) == pytest.approx(-1.0)   # vol premium bearish
    assert engine.normalize_vol(float("nan")) == 0.0


def test_new_signal_normalizers():
    assert engine.normalize_dcf(130.0, 100.0) == pytest.approx(1.0)   # +30% -> +1
    assert engine.normalize_dcf(70.0, 100.0) == pytest.approx(-1.0)   # -30% -> -1
    assert engine.normalize_dcf(100.0, 0.0) == 0.0                    # guard
    assert engine.normalize_piotroski(9) == pytest.approx(1.0)
    assert engine.normalize_piotroski(0) == pytest.approx(-1.0)
    assert engine.normalize_altman(5.99) == pytest.approx(1.0)        # 2.99 + 3.0
    assert engine.normalize_altman(float("nan")) == 0.0
    assert engine.normalize_markowitz(0.2, 10) == pytest.approx(1.0)  # 2x equal-weight
    assert engine.normalize_markowitz(0.0, 10) == pytest.approx(-1.0)
    assert engine.normalize_markowitz(0.1, 10) == pytest.approx(0.0)


def test_full_seven_signal_set_is_not_partial():
    norm = {
        "capm_alpha": 0.5, "gbm_upside": 0.3, "vol_premium": 0.1,
        "dcf_intrinsic": 0.4, "piotroski": 0.6, "altman_z": 0.2,
        "markowitz_weight": 0.5,
    }
    res = engine.score_ticker("TEST", norm)
    assert res.partial is False
    assert -100.0 <= res.composite <= 100.0


def test_classify_verdict_thresholds():
    assert engine.classify_verdict(50) == Verdict.UNDERVALUED
    assert engine.classify_verdict(20) == Verdict.GROWTH
    assert engine.classify_verdict(0) == Verdict.FAIR
    assert engine.classify_verdict(-20) == Verdict.SLIGHT_OVER
    assert engine.classify_verdict(-50) == Verdict.OVERVALUED


def test_score_ticker_all_bullish_signals():
    norm = {"capm_alpha": 1.0, "gbm_upside": 1.0, "vol_premium": 1.0}
    res = engine.score_ticker("TEST", norm)
    assert res.composite == pytest.approx(100.0)
    assert res.verdict == Verdict.UNDERVALUED
    assert res.confidence == Confidence.HIGH
    assert res.n_agree == 3 and res.n_disagree == 0
    assert res.partial is True  # only 3 of 7 signals


def test_score_ticker_renormalizes_partial_weights():
    # Single signal at +1 -> composite should be +100 (weight renormalized).
    res = engine.score_ticker("TEST", {"capm_alpha": 1.0})
    assert res.composite == pytest.approx(100.0)


def test_score_ticker_mixed_signals_lowers_confidence():
    norm = {"capm_alpha": 1.0, "gbm_upside": -1.0, "vol_premium": 0.2}
    res = engine.score_ticker("TEST", norm)
    # capm weight (0.25) > gbm (0.15), so composite leans positive but weakly.
    assert res.n_disagree >= 1
    assert res.confidence in (Confidence.MEDIUM, Confidence.LOW)


def test_score_ticker_unknown_signal_raises():
    with pytest.raises(KeyError):
        engine.score_ticker("TEST", {"bogus": 0.5})
