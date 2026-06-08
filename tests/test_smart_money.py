"""Tests for the Smart Money layer: insider, institutional, EDGAR helpers."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from quant_agent.data import edgar
from quant_agent.models import insider, institutional
from quant_agent.scoring import engine

_TODAY = date(2026, 6, 5)
_RECENT = pd.Timestamp(_TODAY - timedelta(days=5))


def _insider_df(rows: list[dict]) -> pd.DataFrame:
    """Build a Form-4-style frame; defaults: recent date, not a 10b5-1 plan."""
    for r in rows:
        r.setdefault("Date", _RECENT)
        r.setdefault("FilingDate", _RECENT)
        r.setdefault("Is10b5_1", False)
        r.setdefault("Value", 100_000)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Insider (Form 4) — buy-cluster logic
# --------------------------------------------------------------------------- #
def test_insider_three_distinct_buyers_is_bullish():
    df = _insider_df([
        {"Code": "P", "Insider": "Alice", "Position": "CEO"},
        {"Code": "P", "Insider": "Bob", "Position": "CFO"},
        {"Code": "P", "Insider": "Carol", "Position": "Director"},
    ])
    out = insider.insider_signal(df, _TODAY)
    assert out["bull_cluster"] is True and out["score"] > 0
    assert out["n_cluster_buyers"] == 3 and out["has_signal"]


def test_insider_two_buyers_below_threshold_is_neutral():
    df = _insider_df([
        {"Code": "P", "Insider": "Alice", "Position": "CEO"},
        {"Code": "P", "Insider": "Bob", "Position": "CFO"},
    ])
    out = insider.insider_signal(df, _TODAY)
    assert out["bull_cluster"] is False and out["score"] == 0.0


def test_insider_role_weighting_csuite_beats_directors():
    csuite = _insider_df([
        {"Code": "P", "Insider": "A", "Position": "Chief Executive Officer"},
        {"Code": "P", "Insider": "B", "Position": "Chief Financial Officer"},
        {"Code": "P", "Insider": "C", "Position": "President"},
    ])
    directors = _insider_df([
        {"Code": "P", "Insider": "A", "Position": "Director"},
        {"Code": "P", "Insider": "B", "Position": "Director"},
        {"Code": "P", "Insider": "C", "Position": "Director"},
    ])
    csuite_score = insider.insider_signal(csuite, _TODAY)["score"]
    director_score = insider.insider_signal(directors, _TODAY)["score"]
    assert csuite_score > director_score


def test_insider_recency_decay_old_buys_weaker():
    old = pd.Timestamp(_TODAY - timedelta(days=120))
    recent = _insider_df([
        {"Code": "P", "Insider": "A", "Position": "CEO"},
        {"Code": "P", "Insider": "B", "Position": "CFO"},
        {"Code": "P", "Insider": "C", "Position": "COO"},
    ])
    stale = _insider_df([
        {"Code": "P", "Insider": "A", "Position": "CEO", "Date": old, "FilingDate": old},
        {"Code": "P", "Insider": "B", "Position": "CFO", "Date": old, "FilingDate": old},
        {"Code": "P", "Insider": "C", "Position": "COO", "Date": old, "FilingDate": old},
    ])
    assert insider.insider_signal(recent, _TODAY)["score"] > \
        insider.insider_signal(stale, _TODAY)["score"]


def test_insider_routine_selling_is_neutral_not_bearish():
    # Three directors selling -> not C-suite -> no bearish signal.
    df = _insider_df([
        {"Code": "S", "Insider": "A", "Position": "Director"},
        {"Code": "S", "Insider": "B", "Position": "Director"},
        {"Code": "S", "Insider": "C", "Position": "Director"},
    ])
    out = insider.insider_signal(df, _TODAY)
    assert out["bear_cluster"] is False and out["score"] == 0.0


def test_insider_csuite_offplan_sell_cluster_is_bearish():
    df = _insider_df([
        {"Code": "S", "Insider": "A", "Position": "CEO"},
        {"Code": "S", "Insider": "B", "Position": "CFO"},
        {"Code": "S", "Insider": "C", "Position": "COO"},
    ])
    out = insider.insider_signal(df, _TODAY)
    assert out["bear_cluster"] is True and out["score"] < 0


def test_insider_10b51_sells_excluded_from_bearish():
    df = _insider_df([
        {"Code": "S", "Insider": "A", "Position": "CEO", "Is10b5_1": True},
        {"Code": "S", "Insider": "B", "Position": "CFO", "Is10b5_1": True},
        {"Code": "S", "Insider": "C", "Position": "COO", "Is10b5_1": True},
    ])
    out = insider.insider_signal(df, _TODAY)
    assert out["bear_cluster"] is False and out["score"] == 0.0


def test_insider_excludes_grants_and_gifts():
    df = _insider_df([{"Code": c, "Insider": f"P{i}", "Position": "CEO"}
                      for i, c in enumerate("AGMF")])
    out = insider.insider_signal(df, _TODAY)
    assert out["has_signal"] is False and out["score"] == 0.0


def test_insider_empty_input():
    assert insider.insider_signal(pd.DataFrame(), _TODAY)["has_signal"] is False
    assert insider.insider_signal(None, _TODAY)["has_signal"] is False


# --------------------------------------------------------------------------- #
# Institutional (13F)
# --------------------------------------------------------------------------- #
def _fund(name, latest, prior):
    def mk(d):
        return None if d is None else pd.DataFrame(
            {"Ticker": list(d), "shares": list(d.values()), "value": [1] * len(d)}
        )
    return {"name": name, "latest": mk(latest), "prior": mk(prior)}


def test_institutional_new_and_added_are_bullish():
    funds = [
        _fund("A", {"NVDA": 100}, {}),              # new position
        _fund("B", {"NVDA": 200}, {"NVDA": 100}),   # added
    ]
    out = institutional.institutional_signal("NVDA", funds)
    assert out["n_new"] == 1 and out["n_added"] == 1
    assert out["score"] == pytest.approx(1.0)
    assert out["n_holders"] == 2


def test_institutional_exit_and_trim_are_bearish():
    funds = [
        _fund("A", {}, {"NVDA": 100}),              # exited
        _fund("B", {"NVDA": 50}, {"NVDA": 100}),    # trimmed
    ]
    out = institutional.institutional_signal("NVDA", funds)
    assert out["n_exited"] == 1 and out["n_trimmed"] == 1
    assert out["score"] == pytest.approx(-1.0)


def test_institutional_absent_ticker_is_neutral():
    funds = [_fund("A", {"MSFT": 100}, {"MSFT": 100})]
    out = institutional.institutional_signal("NVDA", funds)
    assert out["has_signal"] is False and out["score"] == 0.0


def test_institutional_small_change_within_threshold_is_flat():
    funds = [_fund("A", {"NVDA": 102}, {"NVDA": 100})]  # +2% < 5% threshold
    out = institutional.institutional_signal("NVDA", funds)
    assert out["n_added"] == 0 and out["n_trimmed"] == 0
    assert out["score"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# EDGAR helpers (no network)
# --------------------------------------------------------------------------- #
def test_week_key_anchors_to_monday():
    # 2026-06-04 is a Thursday -> Monday 2026-06-01.
    assert edgar._week_key(date(2026, 6, 4)) == date(2026, 6, 1)


def test_aggregate_holdings_groups_by_ticker_and_excludes_options():
    infotable = pd.DataFrame({
        "Ticker": ["AAPL", "AAPL", "MSFT", "AAPL"],
        "Value": [100, 50, 200, 30],
        "SharesPrnAmount": [10, 5, 20, 3],
        "PutCall": ["", "", "", "Put"],   # the Put row must be excluded
    })
    agg = edgar._aggregate_holdings(infotable, "2026-03-31")
    aapl = agg[agg["Ticker"] == "AAPL"].iloc[0]
    assert aapl["value"] == 150 and aapl["shares"] == 15   # Put row dropped
    assert set(agg["Ticker"]) == {"AAPL", "MSFT"}
    assert (agg["period"] == "2026-03-31").all()


# --------------------------------------------------------------------------- #
# Scoring integration
# --------------------------------------------------------------------------- #
def test_smart_money_normalizers_clip():
    assert engine.normalize_insider(2.0) == 1.0
    assert engine.normalize_institutional(-3.0) == -1.0


def test_smart_money_in_composite_at_small_weights_but_not_core():
    assert engine._WEIGHT_OF["insider"] == pytest.approx(0.06)
    assert engine._WEIGHT_OF["institutional"] == pytest.approx(0.03)
    assert "insider" not in engine._CORE_SIGNALS
    assert "institutional" not in engine._CORE_SIGNALS


def test_all_nine_weights_sum_to_one():
    from quant_agent.config import SCORING_WEIGHTS
    SCORING_WEIGHTS.validate()  # raises if not 1.0


def test_full_core_with_smart_money_not_partial():
    norm = {
        "capm_alpha": 0.2, "gbm_upside": 0.2, "vol_premium": 0.1,
        "dcf_intrinsic": 0.1, "piotroski": 0.3, "altman_z": 0.1,
        "markowitz_weight": 0.2, "insider": 0.5,
    }
    # insider now carries weight, but partial is judged on the 7 core signals.
    res = engine.score_ticker("X", norm)
    assert res.partial is False
    # insider contributes to the composite (non-zero weight) and casts a vote.
    assert any(s.name == "insider" for s in res.signals)
