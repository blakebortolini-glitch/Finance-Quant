"""Read-side Supabase helpers for the agent (service role, bypasses RLS).

Used to expand the daily scoring universe to cover every ticker any user holds
(Phase 4, "score the union"). No-ops to an empty list when Supabase isn't
configured, so local runs without a database still work.
"""

from __future__ import annotations

from loguru import logger

from quant_agent.config import SUPABASE_SERVICE_KEY, SUPABASE_URL


def fetch_held_tickers() -> list[str]:
    """Distinct, upper-cased tickers across all users' holdings (sorted)."""
    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        return []
    try:
        from supabase import create_client

        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        rows = client.table("user_holdings").select("ticker").execute().data or []
        tickers = {str(r["ticker"]).strip().upper() for r in rows if r.get("ticker")}
        if tickers:
            logger.info(f"Holdings coverage: {len(tickers)} distinct user-held tickers")
        return sorted(tickers)
    except Exception as exc:  # noqa: BLE001 - never break the run on a read failure
        logger.warning(f"Could not read user_holdings: {exc}")
        return []


__all__ = ["fetch_held_tickers"]
