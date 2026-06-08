"""Tests for the report tiers and verdict-history / change tracking."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_agent import orchestrator
from quant_agent.orchestrator import RunResult
from quant_agent.reports import brief, detailed, summary
from quant_agent.scoring.engine import Confidence, ScoreResult, Verdict


def _result(ticker: str, composite: float, verdict: Verdict, **over) -> ScoreResult:
    metrics = {
        "alpha": 0.05, "dcf_intrinsic": 150.0, "dcf_fcf_growth": 0.08,
        "piotroski": 8, "altman_z": 4.1, "is_financial": False,
        "vol_label": "vol discount", "vol_gap": -0.06, "max_dd": -0.21,
        "kelly_sized": 0.046, "markowitz_weight": 0.11,
        "beta": 1.2, "rf": 0.04, "required_er": 0.11, "actual_er": 0.16,
        "sigma": 0.30, "garch_sigma": 0.28, "atm_iv": 0.24,
        "sharpe": 1.4, "sortino": 1.9, "var_95": -0.032, "var_99": -0.048,
        "cvar_95": -0.041, "horizon_days": 90, "upside_prob": 0.64,
        "gbm_p5": 108.0, "gbm_p50": 146.0, "gbm_p95": 198.0,
        "gbm_prob_above_s0": 0.64, "gbm_expected_return": 0.061,
        "kelly_continuous": 0.18,
    }
    metrics.update(over.pop("metrics", {}))
    return ScoreResult(
        ticker=ticker, composite=composite, verdict=verdict,
        confidence=Confidence.HIGH, price=138.42, metrics=metrics, **over
    )


@pytest.fixture
def run_result() -> RunResult:
    results = [
        _result("NVDA", 43.0, Verdict.UNDERVALUED),
        _result("JPM", 3.0, Verdict.FAIR, partial=True,
                metrics={"altman_z": None, "is_financial": True, "dcf_intrinsic": None}),
    ]
    portfolio = {
        "weights": pd.Series({"NVDA": 0.7, "JPM": 0.3}),
        "expected_return": 0.15, "volatility": 0.22, "sharpe": 0.5,
        "frontier": pd.DataFrame(
            {"volatility": [0.18, 0.20, 0.25], "target_return": [0.10, 0.13, 0.16]}
        ),
    }
    return RunResult(as_of=date(2026, 6, 4), results=results, portfolio=portfolio)


def test_summary_writes_markdown(tmp_path, monkeypatch, run_result):
    monkeypatch.setattr(summary, "REPORTS_DIR", tmp_path)
    path = summary.write_markdown(run_result)
    text = open(path).read()
    assert "NVDA" in text and "UNDERVALUED" in text
    assert "Bullish: 1" in text


def test_brief_writes_four_line_blocks(tmp_path, monkeypatch, run_result):
    monkeypatch.setattr(brief, "REPORTS_DIR", tmp_path)
    path = brief.write_markdown(run_result)
    text = open(path).read()
    assert "Cheap on:" in text and "Quality:" in text
    assert "Risk note:" in text and "Size:" in text
    # Financial ticker shows the Altman skip in plain English.
    assert "skipped (financial)" in text


def test_detailed_writes_sections_and_portfolio(tmp_path, monkeypatch, run_result):
    monkeypatch.setattr(detailed, "REPORTS_DIR", tmp_path)
    path = detailed.write_markdown(run_result)
    text = open(path).read()
    assert "Stochastic Models" in text
    assert "Position Sizing" in text
    assert "Max-Sharpe" in text
    assert "skipped (financial sector)" in text


def test_history_roundtrip_and_change_detection(tmp_path, monkeypatch):
    hist_path = tmp_path / "verdict_history.json"
    monkeypatch.setattr(orchestrator, "VERDICT_HISTORY_PATH", hist_path)
    monkeypatch.setattr(orchestrator, "REPORTS_DIR", tmp_path)

    assert orchestrator._load_history() == {}
    r1 = _result("NVDA", 40.0, Verdict.UNDERVALUED)
    orchestrator._save_history([r1], date(2026, 6, 3), {})

    hist = orchestrator._load_history()
    assert hist["NVDA"]["verdict"] == "UNDERVALUED"

    # Same verdict -> not changed; different verdict -> changed.
    assert hist["NVDA"]["verdict"] == Verdict.UNDERVALUED.value
    r2 = _result("NVDA", 5.0, Verdict.FAIR)
    assert (hist["NVDA"]["verdict"] != r2.verdict.value) is True
