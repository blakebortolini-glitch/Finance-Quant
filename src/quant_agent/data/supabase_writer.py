"""Optional Supabase sink for quant-agent run results.

Writes verdicts, signal breakdowns, smart-money activity, and the portfolio
snapshot to the web dashboard's database. Uses the SERVICE ROLE key (bypasses
RLS) for writes. No-ops gracefully when ``SUPABASE_SERVICE_KEY`` is absent so
local runs work without a database, and never raises into the pipeline — a
failed write is logged and swallowed.

Wired into ``orchestrator.run()`` and called with the full (unfiltered) result
set so the database always holds every watchlist ticker for the day.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

from loguru import logger

from quant_agent.config import SUPABASE_SERVICE_KEY, SUPABASE_URL
from quant_agent.models import black_scholes as bs
from quant_agent.scoring.engine import ScoreResult

_ATM_OPTION_DAYS = 30


def _enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def _num(x: Any) -> float | None:
    """Coerce to a JSON-safe finite float (NaN/inf/None -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int(x: Any) -> int | None:
    v = _num(x)
    return int(round(v)) if v is not None else None


def _weights_to_dict(weights: Any) -> dict[str, float | None]:
    """pandas Series of weights -> JSON dict {ticker: float}."""
    if weights is None:
        return {}
    try:
        return {str(k): _num(v) for k, v in weights.items()}
    except AttributeError:
        return {}


def _atm_greeks(price: float | None, rf: float | None, sigma: float | None) -> dict[str, float]:
    """ATM 30d call Greeks from forward vol (mirrors the detailed report)."""
    if not price or sigma is None or not math.isfinite(sigma) or sigma <= 0:
        return {}
    T = _ATM_OPTION_DAYS / 365.0
    return bs.greeks(price, price, T, rf or 0.0, sigma, "call")


def write_run_results(results: list[ScoreResult], portfolio: dict, as_of: date) -> bool:
    """Upsert a full run to Supabase. Returns True on success, False otherwise.

    Safe to call unconditionally: returns False (after logging) when Supabase is
    not configured or any error occurs, never propagating an exception.
    """
    if not _enabled():
        logger.info("Supabase not configured (no SUPABASE_SERVICE_KEY) — skipping DB write.")
        return False
    if not results:
        return False

    try:
        from supabase import create_client

        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        d = as_of.isoformat()
        _write_verdicts_and_signals(client, results, d)
        _write_smart_money(client, results, d)
        _write_portfolio(client, portfolio, d)
        _write_securities(client, results, d)
        logger.success(f"Supabase: wrote {len(results)} tickers for {d}.")
        return True
    except Exception as exc:  # noqa: BLE001 - never break the pipeline on a write
        logger.warning(f"Supabase write failed: {exc}")
        return False


def _write_verdicts_and_signals(client: Any, results: list[ScoreResult], d: str) -> None:
    verdict_rows = [
        {
            "date": d,
            "ticker": r.ticker,
            "verdict": r.verdict.value,
            "score": _int(r.composite),
            "confidence": r.confidence.value,
            "is_partial": bool(r.partial),
            "changed_from": r.previous_verdict if r.changed else None,
            "in_watchlist": bool(getattr(r, "in_watchlist", True)),
        }
        for r in results
    ]
    resp = client.table("verdicts").upsert(verdict_rows, on_conflict="date,ticker").execute()
    id_map = {(row["date"], row["ticker"]): row["id"] for row in (resp.data or [])}

    signal_rows = []
    for r in results:
        m = r.metrics
        sm = m.get("smart_money") if isinstance(m.get("smart_money"), dict) else {}
        ins = sm.get("insider", {}) or {}
        inst = sm.get("institutional", {}) or {}
        g = _atm_greeks(r.price, m.get("rf"), m.get("forward_sigma"))

        intrinsic = m.get("dcf_intrinsic")
        dcf_upside = _num(intrinsic / r.price - 1.0) if intrinsic is not None and r.price else None

        signal_rows.append({
            "verdict_id": id_map.get((d, r.ticker)),
            "date": d,
            "ticker": r.ticker,
            "capm_alpha": _num(m.get("alpha")),
            "capm_required_er": _num(m.get("required_er")),
            "capm_actual_er": _num(m.get("actual_er")),
            "capm_beta": _num(m.get("beta")),
            "gbm_upside_prob": _num(m.get("upside_prob")),
            "gbm_p5": _num(m.get("gbm_p5")),
            "gbm_p50": _num(m.get("gbm_p50")),
            "gbm_p95": _num(m.get("gbm_p95")),
            "gbm_expected_return": _num(m.get("gbm_expected_return")),
            "bs_realized_vol": _num(m.get("sigma")),
            "bs_implied_vol": _num(m.get("atm_iv")),
            "bs_vol_premium": _num(m.get("vol_gap")),
            "bs_delta": _num(g.get("delta")),
            "bs_gamma": _num(g.get("gamma")),
            "bs_theta": _num(g.get("theta")),
            "bs_vega": _num(g.get("vega")),
            "dcf_intrinsic": _num(intrinsic),
            "dcf_market_price": _num(r.price),
            "dcf_upside": dcf_upside,
            "piotroski_score": _int(m.get("piotroski")),
            "altman_z": _num(m.get("altman_z")),
            "altman_skipped": bool(m.get("altman_z") is None),
            "markowitz_weight": _num(m.get("markowitz_weight")),
            "kelly_fraction": _num(m.get("kelly_sized")),
            "sharpe": _num(m.get("sharpe")),
            "sortino": _num(m.get("sortino")),
            "max_drawdown": _num(m.get("max_dd")),
            "var_95": _num(m.get("var_95")),
            "cvar_95": _num(m.get("cvar_95")),
            "insider_signal": _num(ins.get("normalized")),
            "insider_cluster_detected": bool(ins.get("bull_cluster")),
            "institutional_signal": _num(inst.get("normalized")),
        })
    client.table("signals").upsert(signal_rows, on_conflict="date,ticker").execute()


def _write_smart_money(client: Any, results: list[ScoreResult], d: str) -> None:
    rows = []
    for r in results:
        sm = r.metrics.get("smart_money")
        if not isinstance(sm, dict) or not sm.get("enabled"):
            continue
        ins = sm.get("insider", {}) or {}
        inst = sm.get("institutional", {}) or {}
        rows.append({
            "date": d,
            "ticker": r.ticker,
            "insider_net_value": _num((ins.get("buy_value") or 0) - (ins.get("sell_value") or 0)),
            "insider_buy_count": _int(ins.get("n_buys")),
            "insider_sell_count": _int(ins.get("n_sells")),
            "insider_cluster_detected": bool(ins.get("bull_cluster")),
            "institutional_holders": {
                "holders": inst.get("holders", []),
                "n_holders": inst.get("n_holders", 0),
                "n_new": inst.get("n_new", 0),
                "n_added": inst.get("n_added", 0),
                "n_trimmed": inst.get("n_trimmed", 0),
                "n_exited": inst.get("n_exited", 0),
            },
            "institutional_signal": _num(inst.get("normalized")),
        })
    if rows:
        client.table("smart_money").upsert(rows, on_conflict="date,ticker").execute()


def _write_portfolio(client: Any, portfolio: dict, d: str) -> None:
    if not portfolio or "weights" not in portfolio:
        return

    frontier = portfolio.get("frontier")
    frontier_data: list[dict] = []
    if frontier is not None and not frontier.empty:
        for _, row in frontier.iterrows():
            frontier_data.append({
                "volatility": _num(row.get("volatility")),
                "return": _num(row.get("target_return")),
            })

    min_var = portfolio.get("min_variance")
    if not isinstance(min_var, dict):
        min_var = {}
    row = {
        "date": d,
        "max_sharpe_weights": _weights_to_dict(portfolio.get("weights")),
        "min_variance_weights": _weights_to_dict(min_var.get("weights")),
        "expected_return": _num(portfolio.get("expected_return")),
        "expected_volatility": _num(portfolio.get("volatility")),
        "sharpe_ratio": _num(portfolio.get("sharpe")),
        "frontier_data": frontier_data,
    }
    client.table("portfolio_snapshots").upsert(row, on_conflict="date").execute()


def _write_securities(client: Any, results: list[ScoreResult], d: str) -> None:
    """Reference price + sector for every scored ticker (powers personal portfolio)."""
    now_iso = datetime.now(UTC).isoformat()
    rows = []
    for r in results:
        m = r.metrics
        rows.append({
            "ticker": r.ticker,
            "name": m.get("name"),
            "sector": m.get("sector"),
            "latest_price": _num(r.price),
            "previous_close": _num(m.get("prev_close")),
            "as_of": d,
            "updated_at": now_iso,
        })
    if rows:
        client.table("securities").upsert(rows, on_conflict="ticker").execute()


__all__ = ["write_run_results"]
