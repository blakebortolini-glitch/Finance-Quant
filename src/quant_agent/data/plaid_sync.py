"""Nightly Plaid investment holdings sync.

Called automatically at the start of each orchestrator run to refresh all
users' Plaid-connected holdings before scoring, so the morning run always
uses the most up-to-date positions.

Per-user failures are caught and logged — one bad connection never stops the
others or the scoring run itself.

Required environment variables:
    PLAID_ENV          — "sandbox" or "production" (default: sandbox)
    PLAID_CLIENT_ID    — Plaid application client ID
    PLAID_SECRET       — Plaid secret key for the active environment
    SUPABASE_URL       — Supabase project URL (already set in config)
    SUPABASE_SERVICE_KEY — Supabase service-role key (already set in config)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import requests
from loguru import logger

from quant_agent.config import SUPABASE_SERVICE_KEY, SUPABASE_URL

# ---------------------------------------------------------------------------
# Plaid REST configuration (avoids adding the plaid-python SDK dependency)
# ---------------------------------------------------------------------------
_PLAID_ENV: str = os.getenv("PLAID_ENV", "sandbox")
_PLAID_CLIENT_ID: str | None = os.getenv("PLAID_CLIENT_ID")
_PLAID_SECRET: str | None = os.getenv("PLAID_SECRET")
_PLAID_BASE: str = f"https://{_PLAID_ENV}.plaid.com"

_TICKER_CLEAN = re.compile(r"[^A-Z.]")  # strip non-ticker characters


def _plaid_post(path: str, payload: dict) -> dict:
    """POST to Plaid's REST API, injecting credentials. Raises on HTTP errors."""
    r = requests.post(
        f"{_PLAID_BASE}{path}",
        json={
            "client_id": _PLAID_CLIENT_ID,
            "secret": _PLAID_SECRET,
            **payload,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sync_all_plaid_connections() -> dict:
    """Sync Plaid investment holdings for every connected user.

    Returns a summary dict::

        {"synced": 2, "failed": 0, "positions_upserted": 14}

    Skips gracefully if Supabase or Plaid credentials are absent (e.g., in
    local dev environments where those services aren't configured).
    """
    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        logger.debug("Plaid auto-sync skipped: Supabase not configured")
        return {"skipped": True, "reason": "supabase_not_configured"}

    if not (_PLAID_CLIENT_ID and _PLAID_SECRET):
        logger.debug("Plaid auto-sync skipped: PLAID_CLIENT_ID / PLAID_SECRET not set")
        return {"skipped": True, "reason": "plaid_not_configured"}

    try:
        from supabase import create_client  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("Plaid auto-sync skipped: supabase-py not installed")
        return {"skipped": True, "reason": "supabase_not_installed"}

    db = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # ── Fetch all connections ─────────────────────────────────────────────────
    rows = (
        db.table("plaid_connections")
        .select("user_id, access_token, institution_name")
        .execute()
        .data
        or []
    )

    if not rows:
        logger.info("Plaid auto-sync: no connections found — nothing to sync")
        return {"synced": 0, "failed": 0, "positions_upserted": 0}

    logger.info(f"Plaid auto-sync: processing {len(rows)} connection(s)")

    synced = 0
    failed = 0
    total_positions = 0

    for conn in rows:
        user_id: str = conn["user_id"]
        access_token: str | None = conn.get("access_token")
        institution: str = conn.get("institution_name") or "unknown"
        uid_short = user_id[:8] + "…"

        if not access_token:
            logger.warning(f"Plaid auto-sync: no access token for {uid_short} ({institution})")
            failed += 1
            continue

        try:
            _sync_one_user(db, user_id, access_token, institution, uid_short)
            synced += 1
        except Exception as exc:  # noqa: BLE001 — per-user isolation
            logger.error(
                f"Plaid auto-sync: failed for {uid_short} ({institution}): {exc}"
            )
            _write_sync_log(db, user_id, 0, "error", str(exc)[:500])
            failed += 1

    logger.info(
        f"Plaid auto-sync complete: {synced} succeeded, {failed} failed, "
        f"{total_positions} positions updated"
    )
    return {"synced": synced, "failed": failed, "positions_upserted": total_positions}


# ---------------------------------------------------------------------------
# Per-user sync
# ---------------------------------------------------------------------------

def _sync_one_user(db, user_id: str, access_token: str, institution: str, uid_short: str) -> int:
    """Fetch holdings for one user and upsert into user_holdings.

    Returns the number of positions upserted.
    """
    # Call Plaid investments/holdings/get
    data = _plaid_post("/investments/holdings/get", {"access_token": access_token})
    holdings = data.get("holdings") or []
    securities = data.get("securities") or []

    # Build security lookup
    sec_by_id: dict[str, dict] = {s["security_id"]: s for s in securities}

    upserts = []
    for h in holdings:
        sec = sec_by_id.get(h.get("security_id", ""))
        if not sec:
            continue

        # Only import equity positions with a ticker symbol
        if sec.get("type") != "equity":
            continue
        raw_ticker = sec.get("ticker_symbol") or ""
        ticker = _TICKER_CLEAN.sub("", raw_ticker.upper())
        if not ticker:
            continue

        quantity = float(h.get("quantity") or 0)
        if quantity <= 0:
            continue

        # cost_basis is total cost; divide by quantity for per-share average.
        # Fall back to institution_price if cost_basis isn't provided.
        cost_basis = h.get("cost_basis")
        inst_price = h.get("institution_price")
        if cost_basis:
            avg_cost = float(cost_basis) / quantity
        elif inst_price:
            avg_cost = float(inst_price)
        else:
            avg_cost = 0.0

        upserts.append({
            "user_id": user_id,
            "ticker": ticker,
            "shares": quantity,
            "avg_cost_basis": round(avg_cost, 4),
            "purchase_date": None,  # Plaid doesn't provide per-lot dates
            "plaid_synced": True,
        })

    if upserts:
        db.table("user_holdings").upsert(upserts, on_conflict="user_id,ticker").execute()

    # Stamp last_synced on the connection row
    db.table("plaid_connections").update(
        {"last_synced": datetime.now(tz=timezone.utc).isoformat()}
    ).eq("user_id", user_id).execute()

    n = len(upserts)
    logger.success(
        f"Plaid auto-sync: {institution} ({uid_short}) → {n} position(s) upserted"
    )

    # Write audit log entry
    _write_sync_log(db, user_id, n, "success", None)
    return n


def _write_sync_log(db, user_id: str, count: int, status: str, error: str | None) -> None:
    """Insert a row into plaid_sync_log (best-effort, never raises)."""
    try:
        db.table("plaid_sync_log").insert({
            "user_id": user_id,
            "tickers_imported": count,
            "status": status,
            "error": error,
        }).execute()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Plaid sync log write failed (non-fatal): {exc}")


__all__ = ["sync_all_plaid_connections"]
